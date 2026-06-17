//! A runnable example of a two-step chain.
//!
//! `llm-chain` never calls an LLM itself — it only prepares the message lists
//! you send to a provider. This example stands in a trivial fake "LLM" (it just
//! uppercases the text) so you can run the whole flow with no network or API key:
//!
//! ```sh
//! cargo run --example two_step_chain
//! ```

use llm_chain::{Chain, Step};
use serde_json::Value;

/// A stand-in for a real chat-completion call.
///
/// In a real program this would POST `messages` to your provider and return the
/// assistant's reply. Here we just echo the last message's content, uppercased.
fn fake_llm(messages: &[Value]) -> String {
    let last = messages
        .last()
        .and_then(|m| m["content"].as_str())
        .unwrap_or_default();
    last.to_uppercase()
}

fn main() {
    let chain = Chain::new()
        .with_history()
        .step(Step::system("You are a concise assistant."))
        .step(Step::user("Summarize: {{input}}").named("summarize"))
        .step(Step::user("Now shout the summary: {{input}}").named("shout"));

    let mut reply = String::new();

    for result in chain.prepare("The quick brown fox jumps over the lazy dog.") {
        println!(
            "--- step {} ({}) ---",
            result.step_index,
            result.step_name.as_deref().unwrap_or("unnamed")
        );
        println!(
            "messages sent: {}",
            serde_json::to_string(&result.messages).unwrap()
        );

        // Run the (fake) model on the prepared messages.
        reply = fake_llm(&result.messages);
        println!("model reply:    {reply}");
    }

    println!("\nfinal reply: {reply}");
}
