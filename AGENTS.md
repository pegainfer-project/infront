# Working on infront

- `cargo x lint` and `cargo x test` are the checks CI runs; pass both before handing work back. `cargo x lint --fix` applies Clippy fixes and formatting.
- PR titles are `<type>: <description>` with no scope, where type is one of `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`. PRs are squash-merged with the title as the commit subject.
- Every commit carries a DCO sign-off (`git commit -s`).
- Match the surrounding code. Prefer expressions and iterator chains over mutable state; let names and types carry intent and keep comments for what the code cannot say.
- Keep the README and doc comments in step with the public API. Doc comments state observable contracts, not implementation history.
- Never commit internal hostnames, IP addresses, filesystem paths or user names.
