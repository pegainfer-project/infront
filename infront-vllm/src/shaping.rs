//! What the HTTP layer does to a generate request before vLLM's routes see
//! it: refuse what the engine declared it does not serve, with a 400 naming
//! the parameter, and mark a chat prompt's history boundary when the engine
//! asked for it. The body is read and parsed once for both; a body that is
//! not JSON passes through untouched for the route to report.

use std::sync::Arc;

use axum::Error as AxumError;
use axum::Json;
use axum::Router;
use axum::body::Body;
use axum::body::Bytes;
use axum::body::to_bytes;
use axum::extract::Request;
use axum::extract::State;
use axum::http::HeaderValue;
use axum::http::Method;
use axum::http::StatusCode;
use axum::http::header::CONTENT_LENGTH;
use axum::middleware;
use axum::middleware::Next;
use axum::response::IntoResponse;
use axum::response::Response;
use http_body_util::LengthLimitError;
use log::debug;
use log::warn;
use serde_json::Value;
use serde_json::json;

use crate::history;
use crate::history::Boundary;

const COMPLETIONS_PATH: &str = "/v1/completions";
const CHAT_COMPLETIONS_PATH: &str = "/v1/chat/completions";

/// The body is buffered to be inspected, and an oversized request must not be
/// silently truncated.
const BODY_LIMIT: usize = 128 * 1024 * 1024;

const LOGPROBS: [&str; 3] = ["logprobs", "top_logprobs", "prompt_logprobs"];

/// Below this a temperature is greedy, as [`infront_engine::SamplingParams`]
/// reads it.
const GREEDY_BELOW: f64 = 1e-5;

/// What an engine serves beyond greedy generation. A request asking for more
/// is refused with a 400 naming the parameter before it reaches the engine.
/// The default serves everything.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Capabilities {
    /// `logprobs`, `top_logprobs` and `prompt_logprobs`.
    pub logprobs: bool,
    /// Drawing at a non-greedy temperature.
    pub sampling: bool,
}

impl Default for Capabilities {
    fn default() -> Self {
        Self {
            logprobs: true,
            sampling: true,
        }
    }
}

impl Capabilities {
    /// Why `body` asks for more than the engine serves, if it does.
    fn refusal(self, body: &Value) -> Option<String> {
        let logprobs = || {
            LOGPROBS
                .into_iter()
                .find(|key| body.get(key).is_some_and(asks))
                .map(|key| format!("{key} is not supported by this engine"))
        };
        let draw = || {
            draw_temperature(body).map(|t| {
                format!(
                    "temperature {t} asks for sampling, which this engine does not support; \
                     send temperature 0 for the most likely token"
                )
            })
        };
        (!self.logprobs)
            .then(logprobs)
            .flatten()
            .or_else(|| (!self.sampling).then(draw).flatten())
    }
}

/// Set to anything but `null` or `false`: a completion's `logprobs: 0` asks
/// for the sampled token's, as a chat's `logprobs: true` does.
fn asks(value: &Value) -> bool {
    !value.is_null() && *value != Value::Bool(false)
}

/// The temperature a request draws at, when it does: one above greedy, with no
/// `top_k` of 1 (one survivor is the most likely token). A request with no
/// temperature passes: the generation config's default is the route's to
/// apply.
fn draw_temperature(body: &Value) -> Option<f64> {
    let t = body
        .get("temperature")?
        .as_f64()
        .filter(|&t| t >= GREEDY_BELOW)?;
    (body.get("top_k").and_then(Value::as_i64) != Some(1)).then_some(t)
}

struct Shaping {
    capabilities: Capabilities,
    boundary: Option<Arc<Boundary>>,
}

/// Wrap a router with the shaping layer; with every capability served and no
/// boundary to measure there is nothing to do, and the router serves as is.
pub(crate) fn shape(
    router: Router,
    capabilities: Capabilities,
    boundary: Option<Boundary>,
) -> Router {
    if capabilities == Capabilities::default() && boundary.is_none() {
        return router;
    }
    let shaping = Arc::new(Shaping {
        capabilities,
        boundary: boundary.map(Arc::new),
    });
    router.layer(middleware::from_fn_with_state(shaping, shape_request))
}

async fn shape_request(
    State(shaping): State<Arc<Shaping>>,
    request: Request,
    next: Next,
) -> Response {
    let chat = match (request.method(), request.uri().path()) {
        (&Method::POST, CHAT_COMPLETIONS_PATH) => true,
        (&Method::POST, COMPLETIONS_PATH) => false,
        _ => return next.run(request).await,
    };

    let (mut parts, body) = request.into_parts();
    let bytes = match to_bytes(body, BODY_LIMIT).await {
        Ok(bytes) => bytes,
        Err(error) => return body_read_error(error),
    };
    let Ok(mut value) = serde_json::from_slice::<Value>(&bytes) else {
        return next
            .run(Request::from_parts(parts, Body::from(bytes)))
            .await;
    };
    if let Some(reason) = shaping.capabilities.refusal(&value) {
        return error_response(StatusCode::BAD_REQUEST, &reason);
    }

    let body = match shaping.boundary.clone().filter(|_| chat) {
        None => bytes,
        Some(boundary) => {
            // Rendering and tokenizing are CPU work; the route itself renders
            // on the blocking pool too.
            let marked = tokio::task::spawn_blocking(move || {
                let tokens = history::mark(&mut value, &boundary);
                (value, tokens)
            })
            .await;
            match marked {
                Ok((value, Some(tokens))) => {
                    debug!("generation prompt is {tokens} tokens");
                    // Serializing a `Value` into a `Vec` cannot fail: every
                    // key is already a string, and a non-finite number
                    // degrades to `null`.
                    let body = Bytes::from(
                        serde_json::to_vec(&value).expect("a serde_json::Value serializes"),
                    );
                    parts
                        .headers
                        .insert(CONTENT_LENGTH, HeaderValue::from(body.len() as u64));
                    body
                }
                Ok((_, None)) => bytes,
                Err(error) => {
                    warn!("history boundary measure failed, serving without it: {error}");
                    bytes
                }
            }
        }
    };
    next.run(Request::from_parts(parts, Body::from(body))).await
}

/// 413 belongs to the length-limit rejection alone; any other read failure is
/// a 400.
fn body_read_error(error: AxumError) -> Response {
    let message = format!("failed to read the request body: {error}");
    let status = if error.into_inner().is::<LengthLimitError>() {
        StatusCode::PAYLOAD_TOO_LARGE
    } else {
        StatusCode::BAD_REQUEST
    };
    error_response(status, &message)
}

/// The OpenAI-style JSON envelope the upstream routes answer with
/// (`vllm_server`'s `ApiError::to_error_response`, which is not exported).
fn error_response(status: StatusCode, message: &str) -> Response {
    (
        status,
        Json(json!({
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": null,
                "code": "invalid_request_error",
            }
        })),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use futures::StreamExt as _;
    use tower::ServiceExt as _;

    use super::*;

    const GREEDY_ONLY: Capabilities = Capabilities {
        logprobs: false,
        sampling: false,
    };

    /// Drive the layer over a route that echoes the body it received.
    async fn post(
        capabilities: Capabilities,
        boundary: Option<Boundary>,
        path: &str,
        body: Body,
    ) -> (StatusCode, Bytes) {
        let route = axum::routing::post(|body: Bytes| async move { body });
        let app = shape(
            Router::new()
                .route(COMPLETIONS_PATH, route.clone())
                .route(CHAT_COMPLETIONS_PATH, route),
            capabilities,
            boundary,
        );
        let request = Request::builder()
            .method(Method::POST)
            .uri(path)
            .body(body)
            .expect("request builds");
        let response = app
            .oneshot(request)
            .await
            .expect("the layer always answers");
        let status = response.status();
        let bytes = to_bytes(response.into_body(), BODY_LIMIT)
            .await
            .expect("response body reads");
        (status, bytes)
    }

    fn error_message(bytes: &Bytes) -> String {
        let body: Value = serde_json::from_slice(bytes).expect("the OpenAI error envelope");
        assert_eq!(
            body["error"]["type"],
            json!("invalid_request_error"),
            "{body}"
        );
        body["error"]["message"]
            .as_str()
            .expect("an error message")
            .to_string()
    }

    #[test]
    fn logprobs_are_refused_by_the_key_that_asks() {
        let refusal = |body: Value| GREEDY_ONLY.refusal(&body);
        for key in LOGPROBS {
            for asking in [json!(0), json!(true), json!(5)] {
                let message = refusal(json!({ key: asking })).expect("must refuse");
                assert!(message.starts_with(key), "{message}");
            }
            assert_eq!(refusal(json!({ key: null })), None);
            assert_eq!(refusal(json!({ key: false })), None);
        }
        assert_eq!(
            Capabilities::default().refusal(&json!({"logprobs": 5})),
            None
        );
    }

    #[test]
    fn only_a_real_draw_is_refused_without_sampling() {
        let refusal = |body: Value| GREEDY_ONLY.refusal(&body);
        assert!(refusal(json!({"temperature": 0.7})).is_some());
        assert_eq!(refusal(json!({"temperature": 0.0})), None);
        assert_eq!(refusal(json!({"temperature": 1e-6})), None);
        assert_eq!(refusal(json!({"temperature": 0.7, "top_k": 1})), None);
        assert_eq!(refusal(json!({})), None);
        assert_eq!(
            Capabilities::default().refusal(&json!({"temperature": 0.7})),
            None
        );
    }

    #[tokio::test]
    async fn a_refused_request_gets_a_400_naming_the_parameter() {
        for path in [COMPLETIONS_PATH, CHAT_COMPLETIONS_PATH] {
            let (status, bytes) = post(
                GREEDY_ONLY,
                None,
                path,
                Body::from(r#"{"prompt":"hi","logprobs":1}"#),
            )
            .await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{path}");
            assert!(error_message(&bytes).contains("logprobs"), "{path}");
        }
    }

    #[tokio::test]
    async fn a_servable_or_unreadable_body_passes_through_verbatim() {
        for body in [r#"{"prompt":"hi","temperature":0}"#, "not json at all"] {
            let (status, bytes) = post(GREEDY_ONLY, None, COMPLETIONS_PATH, Body::from(body)).await;
            assert_eq!(status, StatusCode::OK);
            assert_eq!(&bytes[..], body.as_bytes());
        }
    }

    #[tokio::test]
    async fn a_chat_body_carries_its_generation_prompt_to_the_route() {
        let body = r#"{"messages":[{"role":"user","content":"hi"}]}"#;
        let (status, bytes) = post(
            Capabilities::default(),
            Some(history::tests::boundary()),
            CHAT_COMPLETIONS_PATH,
            Body::from(body),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let value: Value = serde_json::from_slice(&bytes).expect("JSON body");
        assert_eq!(
            value["vllm_xargs"][history::GENERATION_PROMPT_TOKENS],
            json!(2)
        );

        // A completion has no generation prompt to measure.
        let body = r#"{"prompt":"hi"}"#;
        let (_, bytes) = post(
            Capabilities::default(),
            Some(history::tests::boundary()),
            COMPLETIONS_PATH,
            Body::from(body),
        )
        .await;
        assert_eq!(&bytes[..], body.as_bytes());
    }

    #[tokio::test]
    async fn an_oversized_body_is_rejected_as_payload_too_large() {
        let chunk = Bytes::from(vec![0u8; 1024 * 1024]);
        let chunks = BODY_LIMIT / chunk.len() + 2;
        let body = Body::from_stream(
            futures::stream::repeat_with(move || Ok::<Bytes, std::io::Error>(chunk.clone()))
                .take(chunks),
        );
        let (status, bytes) = post(GREEDY_ONLY, None, CHAT_COMPLETIONS_PATH, body).await;
        assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
        assert!(error_message(&bytes).contains("length limit exceeded"));
    }

    #[tokio::test]
    async fn a_truncated_body_keeps_the_bad_request_category_and_the_cause() {
        let frames = vec![
            Ok::<Bytes, std::io::Error>(Bytes::from_static(br#"{"messages":["#)),
            Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "connection closed before the body completed",
            )),
        ];
        let body = Body::from_stream(futures::stream::iter(frames));
        let (status, bytes) = post(GREEDY_ONLY, None, CHAT_COMPLETIONS_PATH, body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(error_message(&bytes).contains("connection closed before the body completed"));
    }
}
