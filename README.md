# llm-chain

Build sequential LLM call chains in Rust, where each step's output feeds the
next step's input. `llm-chain` handles the bookkeeping — template rendering,
message construction, and (optional) conversation history accumulation — while
you remain in control of the actual LLM calls.

This is a small, dependency-light library (`serde_json` only) that produces the
`{"role": ..., "content": ...}` message objects most chat-completion APIs expect.
It is provider-agnostic: it never makes network calls itself.

## How it works

- A **`Step`** holds a `role` (`user` or `system`), a prompt `template`, and an
  optional `name`. Templates support a `{{input}}` placeholder that is replaced
  with the step's input when rendered.
- A **`Chain`** is an ordered list of steps. Calling `prepare(initial_input)`
  walks the chain, rendering each step and passing its rendered output forward
  as the input to the next step.
- With `with_history()`, messages accumulate across steps so each step sees the
  full prior conversation; without it, each step gets a fresh single message.
- Each step produces a **`StepResult`** carrying the step index, optional name,
  the input it received, and the message list to send to your LLM.

## Installation

Add the crate to your `Cargo.toml`:

```toml
[dependencies]
llm-chain = "0.1"
```

## Usage

```rust
use llm_chain::{Chain, Step};

let chain = Chain::new()
    .step(Step::user("Summarize this: {{input}}"))
    .step(Step::user("Now translate to Spanish: {{input}}"));

assert_eq!(chain.len(), 2);

// Build the message list for each step from an initial input.
for result in chain.prepare("The quick brown fox.") {
    // `result.messages` is a Vec<serde_json::Value> ready to send to your LLM.
    // In real use you would call your provider here and feed the reply forward.
    println!("step {}: {:?}", result.step_index, result.messages);
}
```

Named steps and a system prompt:

```rust
use llm_chain::{Chain, Step};

let chain = Chain::new()
    .with_history()
    .step(Step::system("You are a concise assistant."))
    .step(Step::user("Explain {{input}} in one sentence.").named("explain"));
```

### Key API

| Item | Purpose |
| --- | --- |
| `Step::user(t)` / `Step::system(t)` | Create a step with the given role and template. |
| `Step::named(n)` | Attach a name to a step. |
| `Step::render(input)` | Substitute `{{input}}` and return the rendered string. |
| `Step::to_message(input)` | Build the `{role, content}` JSON message. |
| `Chain::new()` | Create an empty chain. |
| `Chain::step(s)` | Append a step (builder style). |
| `Chain::with_history()` | Accumulate conversation history across steps. |
| `Chain::prepare(input)` | Produce a `StepResult` for every step. |
| `Chain::build_messages_for(i, input)` | Build messages for a single step. |
| `Chain::next_message(i, input)` | Build the message for step `i`, or `None` if out of bounds. |

## Tech stack

- **Language:** Rust (edition 2021)
- **Dependencies:** [`serde_json`](https://crates.io/crates/serde_json)

## Development

```sh
cargo build
cargo test
```

## License

Licensed under the [MIT License](https://opensource.org/licenses/MIT).
