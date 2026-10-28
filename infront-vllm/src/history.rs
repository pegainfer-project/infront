//! The prompt's history boundary for the OpenAI chat route.
//!
//! A chat prompt is the rendered conversation followed by the template's
//! generation prompt (`<|im_start|>assistant\n<think>\n` for Qwen3.8). The next
//! turn of the same conversation repeats the rendered conversation byte for
//! byte and then renders the reply the model gave *as history*, which is not
//! how the generation prompt reads. So the longest prefix two turns share ends
//! where the last message ends, and a scheduler whose state cannot roll back
//! (a recurrent line) serves the next turn only from a snapshot taken there.
//! Nothing behind the vendored chat route sees the request text, so this layer
//! measures the boundary in front of the route and the bridge carries it to
//! the scheduler as [`infront_engine::Request::history_tokens`].
//!
//! The measure is per request and local to the last message: the message is
//! rendered alone, with and without the generation prompt, through the same
//! renderer and tokenizer the route uses, and the generation prompt's token
//! count `T` is the tail the two tokenizations do not share. A token that
//! straddles the boundary counts towards `T`, so the boundary moves one token
//! earlier rather than into the generation prompt. `T` travels as
//! `vllm_xargs.generation_prompt_tokens`; the bridge, which sees the whole
//! prompt's token ids, sets `history_tokens = prompt_len - T`. Rendering only
//! the last message is exact for a template that appends the generation prompt
//! after the messages, which is every template the frontend serves, and costs
//! one message's rendering instead of the conversation's.
//!
//! `add_generation_prompt=false` and `continue_final_message=true` render no
//! generation prompt, so `T = 0` and the whole prompt is history. A body the
//! measure cannot read (not JSON, no messages, a last message the renderer
//! rejects) passes through untouched and the scheduler sees no boundary; the
//! route reports its own error for it.
//!
//! The conversation's earlier turns ended at boundaries of their own, each
//! the `history_tokens` of the request that carried it, and a scheduler that
//! keeps snapshots beyond the instance that made them (a pooled tier) can
//! serve the turn from any of them. With [`History::Marks`] those are
//! measured too, as `vllm_xargs.history_marks`, newest first: every message
//! followed by an assistant message ended some earlier request, and the
//! conversation up to it, rendered without a generation prompt and under the
//! body's tools, tokenizes to that request's history. The newest [`MARKS`]
//! are measured, each a rendering of the conversation up to it, so a long
//! agent trace pays for a few renderings rather than one per turn. A template
//! that tokenizes a boundary differently in a longer conversation yields a
//! mark no snapshot sits at, which costs a miss and nothing else.

use serde::de::DeserializeOwned;
use serde_json::Value;
use serde_json::json;
use vllm_chat::AssistantContentBlock;
use vllm_chat::AssistantToolCall;
use vllm_chat::ChatContent;
use vllm_chat::ChatMessage;
use vllm_chat::ChatOptions;
use vllm_chat::ChatRequest;
use vllm_chat::ChatRole;
use vllm_chat::ChatTool;
use vllm_chat::ChatToolChoice;
use vllm_chat::DynChatRenderer;
use vllm_chat::GenerationPromptMode;
use vllm_chat::LoadedModelBackends;
use vllm_chat::ResolvedToolContext;
use vllm_chat::SamplingParams;
use vllm_text::Prompt;
use vllm_text::TextDecodeOptions;
use vllm_text::tokenizer::DynTokenizer;

use crate::History;

/// The `vllm_xargs` key carrying the generation prompt's token count from
/// this layer to the bridge.
pub(crate) const GENERATION_PROMPT_TOKENS: &str = "generation_prompt_tokens";

/// The `vllm_xargs` key carrying the earlier turns' boundaries, newest first.
pub(crate) const HISTORY_MARKS: &str = "history_marks";

/// How many earlier turns' boundaries are measured, the newest first.
const MARKS: usize = 8;

/// The route's renderer and tokenizer, held to measure one message at a time.
pub(crate) struct Boundary {
    renderer: DynChatRenderer,
    tokenizer: DynTokenizer,
    marks: bool,
}

impl Boundary {
    /// `None` when `history` measures nothing.
    pub(crate) fn new(backends: &LoadedModelBackends, history: History) -> Option<Self> {
        (history != History::Unmeasured).then(|| Self {
            renderer: backends.chat_backend.chat_renderer(),
            tokenizer: backends.text_backend.tokenizer(),
            marks: history == History::Marks,
        })
    }

    /// The generation prompt's token count for one chat body, or `None` when
    /// the body does not render.
    fn generation_prompt_tokens(&self, body: &Value) -> Option<usize> {
        let options = chat_options(body);
        if !options.add_generation_prompt() {
            return Some(0);
        }
        let message = last_message(body)?;
        let with = self.tokens(&message, options.clone())?;
        let without = self.tokens(
            &message,
            ChatOptions {
                generation_prompt_mode: GenerationPromptMode::NoGenerationPrompt,
                ..options
            },
        )?;
        let shared = with
            .iter()
            .zip(&without)
            .take_while(|(a, b)| a == b)
            .count();
        Some(with.len() - shared)
    }

    /// Where the earlier turns of one chat body ended, newest first: the
    /// token count of the conversation up to each message an assistant
    /// message follows, rendered without a generation prompt.
    ///
    /// The tools and the tool choice render into the conversation, so each
    /// earlier turn is rendered under the body's own, as it was sent.
    fn history_marks(&self, body: &Value) -> Vec<usize> {
        let choice = match body.get("tool_choice") {
            None | Some(Value::Null) => Ok(None),
            Some(choice) => tool_choice(choice).map(Some).ok_or(()),
        };
        let (Some(messages), Some(tools), Ok(choice)) =
            (messages(body), tools(body.get("tools")), choice)
        else {
            return Vec::new();
        };
        let parallel = body
            .get("parallel_tool_calls")
            .and_then(Value::as_bool)
            .unwrap_or(true);
        // Bounded before rendering: a turn the template rejects still costs
        // its rendering.
        let ends = (0..messages.len().saturating_sub(1))
            .rev()
            .filter(|&i| messages[i + 1].role() == ChatRole::Assistant)
            .take(MARKS);
        let options = ChatOptions {
            generation_prompt_mode: GenerationPromptMode::NoGenerationPrompt,
            ..chat_options(body)
        };
        ends.filter_map(|i| {
            let turn = &messages[..=i];
            let context =
                ResolvedToolContext::new(turn, tools.clone(), choice.clone(), parallel).ok()?;
            Some(self.render(turn, options.clone(), context)?.len())
        })
        .filter(|&m| m > 0)
        .collect()
    }

    /// One message alone, tool-free: what it adds next to the generation
    /// prompt does not depend on the tools.
    fn tokens(&self, message: &ChatMessage, chat_options: ChatOptions) -> Option<Vec<u32>> {
        self.render(
            std::slice::from_ref(message),
            chat_options,
            ResolvedToolContext::default(),
        )
    }

    fn render(
        &self,
        messages: &[ChatMessage],
        chat_options: ChatOptions,
        tool_context: ResolvedToolContext,
    ) -> Option<Vec<u32>> {
        let rendered = self
            .renderer
            .render(&probe_request(
                messages.to_vec(),
                chat_options,
                tool_context,
            ))
            .ok()?;
        match rendered.prompt {
            Prompt::TokenIds(ids) => Some(ids),
            Prompt::Text(text) => self.tokenizer.encode(&text, false).ok(),
        }
    }
}

/// The template controls the route itself resolves from the body, so the
/// probe renders under the same mode, template and kwargs as the prompt.
fn chat_options(body: &Value) -> ChatOptions {
    let add_generation_prompt = body.get("add_generation_prompt").and_then(Value::as_bool);
    let continue_final_message = body
        .get("continue_final_message")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let generation_prompt_mode = match (add_generation_prompt, continue_final_message) {
        (_, true) => GenerationPromptMode::ContinueFinalAssistant,
        (Some(false), false) => GenerationPromptMode::NoGenerationPrompt,
        (None | Some(true), false) => GenerationPromptMode::StartNewAssistant,
    };
    ChatOptions {
        generation_prompt_mode,
        chat_template: field(body, "chat_template"),
        reasoning_effort: field(body, "reasoning_effort"),
        response_format: None,
        template_kwargs: field(body, "chat_template_kwargs").unwrap_or_default(),
    }
}

fn field<T: DeserializeOwned>(body: &Value, key: &str) -> Option<T> {
    serde_json::from_value(body.get(key)?.clone()).ok()
}

/// The last message in the route's own vocabulary; the route rejects what
/// this layer cannot read.
fn last_message(body: &Value) -> Option<ChatMessage> {
    message(body.get("messages")?.as_array()?.last()?)
}

/// Every message, as [`last_message`] reads one; `None` when any does not read.
fn messages(body: &Value) -> Option<Vec<ChatMessage>> {
    body.get("messages")?
        .as_array()?
        .iter()
        .map(message)
        .collect()
}

fn message(message: &Value) -> Option<ChatMessage> {
    let role = message.get("role")?.as_str()?;
    let content = match message.get("content") {
        None | Some(Value::Null) => ChatContent::Text(String::new()),
        Some(Value::String(text)) => ChatContent::Text(text.clone()),
        Some(parts) => serde_json::from_value(parts.clone()).ok()?,
    };
    Some(match role {
        "system" => ChatMessage::system(content),
        "developer" => {
            let tools = tools(message.get("tools"))?;
            ChatMessage::developer(content, (!tools.is_empty()).then_some(tools))
        }
        "user" => ChatMessage::user(content),
        "assistant" => assistant(message)?,
        "tool" => ChatMessage::tool_response(
            content,
            message
                .get("tool_call_id")
                .and_then(Value::as_str)
                .unwrap_or_default(),
        ),
        custom => ChatMessage::custom(custom, content),
    })
}

/// An assistant message as the route reads one: its reasoning, its text and
/// its tool calls, in that order, every one of which a template may render
/// into the history.
fn assistant(message: &Value) -> Option<ChatMessage> {
    let reasoning = ["reasoning", "reasoning_content", "thinking"]
        .into_iter()
        .find_map(|key| message.get(key)?.as_str())
        .filter(|text| !text.is_empty())
        .map(|text| AssistantContentBlock::Reasoning {
            text: text.to_string(),
        });
    let text = match message.get("content") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::String(text)) => vec![AssistantContentBlock::Text { text: text.clone() }],
        Some(Value::Array(parts)) => parts
            .iter()
            .map(|part| match part.get("type")?.as_str()? {
                "text" => Some(AssistantContentBlock::Text {
                    text: part.get("text")?.as_str()?.to_string(),
                }),
                _ => None,
            })
            .collect::<Option<_>>()?,
        Some(_) => return None,
    };
    let calls = match message.get("tool_calls") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(calls)) => calls.iter().map(tool_call).collect::<Option<_>>()?,
        Some(_) => return None,
    };
    let blocks: Vec<_> = reasoning.into_iter().chain(text).chain(calls).collect();
    (!blocks.is_empty()).then(|| ChatMessage::assistant_blocks(blocks))
}

fn tool_call(call: &Value) -> Option<AssistantContentBlock> {
    let function = call.get("function")?;
    (call.get("type")?.as_str()? == "function").then(|| {
        Some(AssistantContentBlock::ToolCall(AssistantToolCall {
            id: call.get("id")?.as_str()?.to_string(),
            name: function.get("name")?.as_str()?.to_string(),
            arguments: function
                .get("arguments")
                .and_then(Value::as_str)
                .unwrap_or("{}")
                .to_string(),
        }))
    })?
}

/// Request-level or message-scoped function tools; `None` when any does not
/// read.
fn tools(tools: Option<&Value>) -> Option<Vec<ChatTool>> {
    match tools {
        None | Some(Value::Null) => Some(Vec::new()),
        Some(Value::Array(tools)) => tools.iter().map(tool).collect(),
        Some(_) => None,
    }
}

fn tool(tool: &Value) -> Option<ChatTool> {
    let kind = tool.get("type").map_or(Some("function"), Value::as_str)?;
    let function = tool.get("function")?;
    let flag = |value: &Value, key| value.get(key).and_then(Value::as_bool);
    (kind == "function").then(|| {
        Some(ChatTool {
            name: function.get("name")?.as_str()?.to_string(),
            description: function
                .get("description")
                .and_then(Value::as_str)
                .map(str::to_string),
            parameters: function.get("parameters").cloned().unwrap_or_default(),
            strict: flag(function, "strict"),
            defer_loading: flag(function, "defer_loading").or(flag(tool, "defer_loading")),
        })
    })?
}

/// A tool choice the route accepts.
fn tool_choice(choice: &Value) -> Option<ChatToolChoice> {
    match choice {
        Value::String(choice) => match choice.as_str() {
            "auto" => Some(ChatToolChoice::Auto),
            "none" => Some(ChatToolChoice::None),
            "required" => Some(ChatToolChoice::Required),
            _ => None,
        },
        _ if choice.get("type")?.as_str()? == "function" => Some(ChatToolChoice::Function {
            name: choice.get("function")?.get("name")?.as_str()?.to_string(),
        }),
        _ => None,
    }
}

/// Messages under the request's template controls and tools, every other
/// field at the value the route resolves for a plain text chat.
fn probe_request(
    messages: Vec<ChatMessage>,
    chat_options: ChatOptions,
    tool_context: ResolvedToolContext,
) -> ChatRequest {
    ChatRequest {
        request_id: "history-boundary-probe".to_string(),
        messages,
        sampling_params: SamplingParams::default(),
        chat_options,
        tool_context,
        decode_options: TextDecodeOptions::default(),
        intermediate: false,
        prompt_truncation: None,
        priority: 0,
        documents: None,
        cache_salt: None,
        add_special_tokens: false,
        data_parallel_rank: None,
        session_id: None,
        lora_request: None,
    }
}

/// Write the measured count, and the earlier turns' boundaries when the
/// boundary measures them, into the body's `vllm_xargs`, returning the count
/// when the body changed. A caller's own value for either key is replaced:
/// they are this server's measure of its own template, not request
/// parameters.
pub(crate) fn mark(value: &mut Value, boundary: &Boundary) -> Option<usize> {
    let tokens = boundary.generation_prompt_tokens(value)?;
    let marks = boundary.marks.then(|| boundary.history_marks(value));
    let xargs = value
        .as_object_mut()?
        .entry("vllm_xargs")
        .or_insert_with(|| json!({}))
        .as_object_mut()?;
    xargs.insert(GENERATION_PROMPT_TOKENS.to_string(), json!(tokens));
    if let Some(marks) = marks {
        xargs.insert(HISTORY_MARKS.to_string(), json!(marks));
    }
    Some(tokens)
}

#[cfg(test)]
pub(crate) mod tests {
    use std::collections::HashMap;
    use std::fmt::Write as _;
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use vllm_chat::ChatRenderer;
    use vllm_chat::RenderedPrompt;
    use vllm_text::tokenizer::Tokenizer;

    use super::*;

    /// A template that wraps each message in its role and appends the
    /// generation prompt as separate words, except for the tail of a user
    /// message ending in `~`, which it glues to the generation prompt so one
    /// token straddles the boundary.
    struct Template;

    impl ChatRenderer for Template {
        fn render(&self, request: &ChatRequest) -> vllm_chat::Result<RenderedPrompt> {
            let mut text = String::new();
            let tools: Vec<_> = request
                .initial_tools()
                .iter()
                .map(|t| t.name.as_str())
                .collect();
            if !tools.is_empty() {
                write!(text, "<tools> {} <end>", tools.join(" ")).unwrap();
            }
            for message in &request.messages {
                let role = serde_json::to_value(message.role()).unwrap();
                let mut content = message.text_content()?;
                if let ChatMessage::Assistant { content: blocks } = message {
                    for block in blocks {
                        if let AssistantContentBlock::ToolCall(call) = block {
                            write!(content, " call:{}", call.name).unwrap();
                        }
                    }
                }
                write!(text, "<{}> {content} <end>", role.as_str().unwrap()).unwrap();
            }
            if request.chat_options.add_generation_prompt() {
                if text.ends_with("~ <end>") {
                    text.push_str("<assistant><think>");
                } else {
                    text.push_str(" <assistant> <think>");
                }
            }
            Ok(RenderedPrompt {
                prompt: Prompt::Text(text),
                media_order: None,
                effective_template_kwargs: HashMap::default(),
            })
        }
    }

    /// Whitespace words are tokens; the id is the word's length.
    struct Words;

    impl Tokenizer for Words {
        fn encode(&self, text: &str, _: bool) -> vllm_text::tokenizer::Result<Vec<u32>> {
            Ok(text.split_whitespace().map(|w| w.len() as u32).collect())
        }
        fn encode_ordinary(&self, text: &str) -> vllm_text::tokenizer::Result<Vec<u32>> {
            self.encode(text, false)
        }
        fn decode(&self, _: &[u32], _: bool) -> vllm_text::tokenizer::Result<String> {
            unimplemented!()
        }
        fn token_to_id(&self, _: &str) -> Option<u32> {
            None
        }
        fn id_to_token(&self, _: u32) -> Option<String> {
            None
        }
    }

    pub(crate) fn boundary() -> Boundary {
        Boundary {
            renderer: Arc::new(Template),
            tokenizer: Arc::new(Words),
            marks: true,
        }
    }

    fn mark_json(json: &str) -> (Option<usize>, Value) {
        let mut value = serde_json::from_str(json).expect("test body is JSON");
        let changed = mark(&mut value, &boundary());
        (changed, value)
    }

    #[test]
    fn the_generation_prompt_is_the_tail_the_history_rendering_lacks() {
        let (changed, value) = mark_json(r#"{"messages":[{"role":"user","content":"hi there"}]}"#);
        assert_eq!(changed, Some(2), "<assistant> <think>");
        assert_eq!(value["vllm_xargs"][GENERATION_PROMPT_TOKENS], json!(2));

        let (changed, _) = mark_json(
            r#"{"messages":[{"role":"system","content":"s"},{"role":"tool","content":"out","tool_call_id":"c1"}]}"#,
        );
        assert_eq!(changed, Some(2), "the last message alone decides");
    }

    #[test]
    fn a_token_straddling_the_boundary_is_counted_as_generation_prompt() {
        let (changed, _) = mark_json(r#"{"messages":[{"role":"user","content":"hi ~"}]}"#);
        assert_eq!(
            changed,
            Some(1),
            "`<end><assistant><think>` replaced `<end>`"
        );
    }

    #[test]
    fn a_request_without_a_generation_prompt_is_all_history() {
        let (changed, _) = mark_json(
            r#"{"messages":[{"role":"user","content":"hi"}],"add_generation_prompt":false}"#,
        );
        assert_eq!(changed, Some(0));

        let (changed, _) = mark_json(
            r#"{"messages":[{"role":"assistant","content":"so"}],"continue_final_message":true}"#,
        );
        assert_eq!(changed, Some(0));
    }

    #[test]
    fn the_mark_joins_the_callers_xargs_and_replaces_its_own_key() {
        let (changed, value) = mark_json(
            r#"{"messages":[{"role":"user","content":"hi"}],"vllm_xargs":{"kv_transfer_params":1,"generation_prompt_tokens":99}}"#,
        );
        assert_eq!(changed, Some(2));
        assert_eq!(
            value["vllm_xargs"],
            json!({"kv_transfer_params": 1, GENERATION_PROMPT_TOKENS: 2, HISTORY_MARKS: []})
        );
    }
    #[test]
    fn the_earlier_turns_end_where_an_assistant_message_follows() {
        // Three turns: each user message followed by an assistant one ended a
        // request, measured as the conversation up to it; the last did not.
        let (_, value) = mark_json(
            r#"{"messages":[{"role":"system","content":"s"},{"role":"user","content":"a b"},{"role":"assistant","content":"c"},{"role":"user","content":"d"},{"role":"assistant","content":"e f"},{"role":"user","content":"g"}]}"#,
        );
        // `<system> s <end><user> a b <end>` is 6 words (`<end>` glues to the
        // next role), `<assistant> c <end><user> d <end>` 4 more.
        assert_eq!(value["vllm_xargs"][HISTORY_MARKS], json!([10, 6]));
        let (_, value) = mark_json(r#"{"messages":[{"role":"user","content":"a"}]}"#);
        assert_eq!(value["vllm_xargs"][HISTORY_MARKS], json!([]));
    }

    #[test]
    fn an_earlier_turn_renders_with_the_tools_and_its_tool_calls() {
        // `<tools> grep <end><system> s <end><user> a <end>` is 7 words, the
        // request that ended there measured 9 with `<assistant> <think>` and a
        // 2-word generation prompt; the tool call and its result add
        // `<assistant> call:grep <end><tool> out <end>`, 4 more.
        let (_, value) = mark_json(
            r#"{"tools":[{"type":"function","function":{"name":"grep"}}],"messages":[{"role":"system","content":"s"},{"role":"user","content":"a"},{"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"grep","arguments":"{}"}}]},{"role":"tool","content":"out","tool_call_id":"c1"},{"role":"assistant","content":"done"},{"role":"user","content":"b"}]}"#,
        );
        assert_eq!(value["vllm_xargs"][HISTORY_MARKS], json!([11, 7]));
    }

    /// The test template, counting its renderings and rejecting every
    /// conversation that opens with `reject`.
    struct Rejecting(Arc<AtomicUsize>);

    impl ChatRenderer for Rejecting {
        fn render(&self, request: &ChatRequest) -> vllm_chat::Result<RenderedPrompt> {
            self.0.fetch_add(1, Ordering::Relaxed);
            match request.messages[0].text_content()?.as_str() {
                "reject" => Err(vllm_chat::Error::ChatTemplate("rejected".to_string())),
                _ => Template.render(request),
            }
        }
    }

    #[test]
    fn turns_the_template_rejects_count_towards_the_marks_measured() {
        // Every earlier turn of twenty fails to render; they cost the last
        // message's two renderings and the newest eight turns' only.
        let first = json!({"role":"user","content":"reject"});
        let turns = (0..20).flat_map(|_| {
            [
                json!({"role":"assistant","content":"a"}),
                json!({"role":"user","content":"u"}),
            ]
        });
        let mut value =
            json!({"messages": std::iter::once(first).chain(turns).collect::<Vec<_>>()});
        let renders = Arc::new(AtomicUsize::new(0));
        let boundary = Boundary {
            renderer: Arc::new(Rejecting(Arc::clone(&renders))),
            ..boundary()
        };
        assert_eq!(mark(&mut value, &boundary), Some(2));
        assert_eq!(
            (
                value["vllm_xargs"][HISTORY_MARKS].clone(),
                renders.load(Ordering::Relaxed)
            ),
            (json!([]), 2 + MARKS)
        );
    }

    #[test]
    fn a_boundary_without_marks_writes_only_the_count() {
        let mut value = json!({"messages":[{"role":"user","content":"a"},{"role":"assistant","content":"b"},{"role":"user","content":"c"}]});
        let boundary = Boundary {
            marks: false,
            ..boundary()
        };
        assert_eq!(mark(&mut value, &boundary), Some(2));
        assert_eq!(value["vllm_xargs"], json!({GENERATION_PROMPT_TOKENS: 2}));
    }

    #[test]
    fn a_body_the_measure_cannot_read_passes_through_verbatim() {
        for json in [
            r#"{"prompt":"not a chat"}"#,
            r#"{"messages":[]}"#,
            r#"{"messages":[{"content":"no role"}]}"#,
            r#"{"messages":[{"role":"user","content":"hi"}],"vllm_xargs":7}"#,
        ] {
            let (changed, value) = mark_json(json);
            assert_eq!(changed, None, "{json}");
            assert_eq!(
                value,
                serde_json::from_str::<Value>(json).unwrap(),
                "{json}"
            );
        }
    }
}
