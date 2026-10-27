use std::env;
use std::path::Path;
use std::process::Command;
use std::process::ExitCode;

use clap::Parser;
use clap::Subcommand;

#[derive(Parser)]
#[command(
    name = "cargo x",
    bin_name = "cargo x",
    about = "Repository workflows for infront"
)]
struct Cli {
    #[command(subcommand)]
    task: Task,
}

#[derive(Subcommand)]
enum Task {
    /// Check formatting, Clippy and dependency policy
    Lint {
        /// Apply Clippy fixes and formatting instead of checking
        #[arg(long)]
        fix: bool,
    },
    /// Run workspace tests and the echo-server smoke test
    Test,
}

const CLIPPY: [&str; 5] = [
    "clippy",
    "--workspace",
    "--all-targets",
    "--all-features",
    "--locked",
];

fn main() -> ExitCode {
    let steps = match Cli::parse().task {
        Task::Lint { fix: false } => vec![
            cargo(["fmt", "--all", "--", "--check"]),
            cargo(CLIPPY.into_iter().chain(["--", "-D", "warnings"])),
            cargo(["deny", "--all-features", "check"]),
        ],
        Task::Lint { fix: true } => vec![
            cargo(
                CLIPPY
                    .into_iter()
                    .chain(["--fix", "--allow-dirty", "--allow-staged"]),
            ),
            cargo(["fmt", "--all"]),
        ],
        Task::Test => vec![
            cargo(["test", "--workspace", "--locked"]),
            in_workspace(Command::new(workspace().join("scripts/echo_smoke.sh"))),
        ],
    };
    steps
        .into_iter()
        .try_for_each(run)
        .map_or_else(|code| code, |()| ExitCode::SUCCESS)
}

fn workspace() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask lives inside the workspace")
}

fn cargo<'a>(args: impl IntoIterator<Item = &'a str>) -> Command {
    let mut command = in_workspace(Command::new(
        env::var_os("CARGO").unwrap_or_else(|| "cargo".into()),
    ));
    command.args(args);
    command
}

fn in_workspace(mut command: Command) -> Command {
    command.current_dir(workspace());
    command
}

fn run(mut command: Command) -> Result<(), ExitCode> {
    let program = Path::new(command.get_program())
        .file_name()
        .unwrap_or_default();
    let args = command
        .get_args()
        .map(|arg| arg.to_string_lossy())
        .collect::<Vec<_>>();
    eprintln!("$ {} {}", program.to_string_lossy(), args.join(" "));
    match command.status() {
        Ok(status) if status.success() => Ok(()),
        Ok(_) => Err(ExitCode::FAILURE),
        Err(error) => {
            eprintln!("failed to run {command:?}: {error}");
            Err(ExitCode::FAILURE)
        }
    }
}
