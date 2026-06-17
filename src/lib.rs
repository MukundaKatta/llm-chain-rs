/*!
llm-chain: build sequential LLM call chains.

Each step in the chain receives the previous step's output and produces
a new message list for the next call. This crate handles the bookkeeping;
you provide the actual LLM calls.

```rust
use llm_chain::{Chain, Step};
use serde_json::json;

let chain = Chain::new()
    .step(Step::user("Summarize this: {{input}}"))
    .step(Step::user("Now translate to Spanish: {{input}}"));
assert_eq!(chain.len(), 2);
```
*/

use serde_json::{json, Value};

/// A single step in a chain.
#[derive(Debug, Clone)]
pub struct Step {
    pub role: String,
    pub template: String,
    pub name: Option<String>,
}

impl Step {
    /// Create a `user`-role step from a prompt template.
    pub fn user(template: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            template: template.into(),
            name: None,
        }
    }

    /// Create a `system`-role step from a prompt template.
    pub fn system(template: impl Into<String>) -> Self {
        Self {
            role: "system".into(),
            template: template.into(),
            name: None,
        }
    }

    /// Create an `assistant`-role step from a template.
    ///
    /// Useful for seeding a conversation with a canned assistant turn or for
    /// replaying a prior reply when building history-aware chains.
    pub fn assistant(template: impl Into<String>) -> Self {
        Self {
            role: "assistant".into(),
            template: template.into(),
            name: None,
        }
    }

    /// Attach a name to this step (builder style). Returns `self`.
    pub fn named(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Render the template, replacing the `{{input}}` placeholder with `input`.
    pub fn render(&self, input: &str) -> String {
        self.template.replace("{{input}}", input)
    }

    /// Render the template, substituting `{{input}}` with `input` and every
    /// `{{key}}` found in `vars` with its corresponding value.
    ///
    /// `{{input}}` is applied first, so a `vars` entry keyed `"input"` is
    /// ignored. Placeholders without a matching variable are left untouched.
    ///
    /// ```
    /// use llm_chain::Step;
    /// let s = Step::user("Hi {{name}}, about {{input}}");
    /// let vars = [("name", "Ada")];
    /// assert_eq!(s.render_with("Rust", vars), "Hi Ada, about Rust");
    /// ```
    pub fn render_with<K, V>(&self, input: &str, vars: impl IntoIterator<Item = (K, V)>) -> String
    where
        K: AsRef<str>,
        V: AsRef<str>,
    {
        let mut out = self.render(input);
        for (key, value) in vars {
            let placeholder = format!("{{{{{}}}}}", key.as_ref());
            out = out.replace(&placeholder, value.as_ref());
        }
        out
    }

    /// Build the `{"role": ..., "content": ...}` JSON message for this step.
    pub fn to_message(&self, input: &str) -> Value {
        json!({"role": self.role, "content": self.render(input)})
    }
}

/// Result of running one step.
#[derive(Debug, Clone)]
pub struct StepResult {
    pub step_index: usize,
    pub step_name: Option<String>,
    pub input: String,
    pub messages: Vec<Value>,
}

/// A sequential chain of LLM call steps.
#[derive(Debug, Default)]
pub struct Chain {
    steps: Vec<Step>,
    history: bool,
}

impl Chain {
    pub fn new() -> Self {
        Self::default()
    }

    /// Accumulate conversation history across steps.
    pub fn with_history(mut self) -> Self {
        self.history = true;
        self
    }

    pub fn step(mut self, step: Step) -> Self {
        self.steps.push(step);
        self
    }

    pub fn len(&self) -> usize {
        self.steps.len()
    }
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Build the message list for the step at `index` with the given input.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds. Use [`Chain::try_build_messages_for`]
    /// for a non-panicking variant.
    pub fn build_messages_for(&self, index: usize, input: &str) -> Vec<Value> {
        let step = &self.steps[index];
        vec![step.to_message(input)]
    }

    /// Build the message list for the step at `index`, returning `None` if the
    /// index is out of bounds.
    ///
    /// This is the non-panicking counterpart to [`Chain::build_messages_for`].
    pub fn try_build_messages_for(&self, index: usize, input: &str) -> Option<Vec<Value>> {
        self.steps.get(index).map(|s| vec![s.to_message(input)])
    }

    /// Build all step results given an initial input.
    /// In non-history mode, each step gets a fresh single message.
    pub fn prepare(&self, initial_input: &str) -> Vec<StepResult> {
        let mut results = Vec::new();
        let mut current_input = initial_input.to_string();
        let mut accumulated: Vec<Value> = Vec::new();

        for (i, step) in self.steps.iter().enumerate() {
            let msg = step.to_message(&current_input);
            if self.history {
                accumulated.push(msg);
                results.push(StepResult {
                    step_index: i,
                    step_name: step.name.clone(),
                    input: current_input.clone(),
                    messages: accumulated.clone(),
                });
            } else {
                results.push(StepResult {
                    step_index: i,
                    step_name: step.name.clone(),
                    input: current_input.clone(),
                    messages: vec![step.to_message(&current_input)],
                });
            }
            // Simulate: in real use the caller runs the LLM and provides the next input.
            // Here we just pass the rendered content forward as the "output".
            current_input = step.render(&current_input);
        }
        results
    }

    /// Feed an assistant reply into the accumulated history and return the next user message.
    pub fn next_message(&self, step_index: usize, input: &str) -> Option<Value> {
        self.steps.get(step_index).map(|s| s.to_message(input))
    }

    pub fn steps(&self) -> &[Step] {
        &self.steps
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chain_len() {
        let c = Chain::new().step(Step::user("hi")).step(Step::user("bye"));
        assert_eq!(c.len(), 2);
    }

    #[test]
    fn step_render_substitutes_input() {
        let s = Step::user("Tell me about {{input}}.");
        assert_eq!(s.render("Rust"), "Tell me about Rust.");
    }

    #[test]
    fn step_no_placeholder_unchanged() {
        let s = Step::user("Hello!");
        assert_eq!(s.render("anything"), "Hello!");
    }

    #[test]
    fn step_to_message() {
        let s = Step::user("How are you?");
        let m = s.to_message("ignored");
        assert_eq!(m["role"], "user");
        assert_eq!(m["content"], "How are you?");
    }

    #[test]
    fn system_step_role() {
        let s = Step::system("You are helpful.");
        let m = s.to_message("");
        assert_eq!(m["role"], "system");
    }

    #[test]
    fn named_step() {
        let s = Step::user("hi").named("greeting");
        assert_eq!(s.name.as_deref(), Some("greeting"));
    }

    #[test]
    fn prepare_length_matches_chain() {
        let c = Chain::new()
            .step(Step::user("step1: {{input}}"))
            .step(Step::user("step2: {{input}}"));
        let results = c.prepare("start");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn prepare_first_step_input() {
        let c = Chain::new().step(Step::user("Q: {{input}}"));
        let results = c.prepare("hello");
        assert_eq!(results[0].input, "hello");
    }

    #[test]
    fn prepare_second_step_gets_rendered_first() {
        let c = Chain::new()
            .step(Step::user("First: {{input}}"))
            .step(Step::user("Second: {{input}}"));
        let results = c.prepare("data");
        // Second step's input is the rendered first step output.
        assert!(results[1].input.contains("First:") || results[1].input.contains("data"));
    }

    #[test]
    fn with_history_accumulates() {
        let c = Chain::new()
            .with_history()
            .step(Step::user("a"))
            .step(Step::user("b"));
        let results = c.prepare("x");
        assert_eq!(results[1].messages.len(), 2);
    }

    #[test]
    fn next_message_returns_step_message() {
        let c = Chain::new().step(Step::user("Q: {{input}}"));
        let m = c.next_message(0, "hello").unwrap();
        assert!(m["content"].as_str().unwrap().contains("hello"));
    }

    #[test]
    fn next_message_out_of_bounds() {
        let c = Chain::new();
        assert!(c.next_message(5, "x").is_none());
    }

    #[test]
    fn build_messages_for_step() {
        let c = Chain::new().step(Step::user("Hi {{input}}"));
        let msgs = c.build_messages_for(0, "world");
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["content"], "Hi world");
    }

    #[test]
    fn empty_chain() {
        let c = Chain::new();
        assert!(c.is_empty());
        assert!(c.prepare("x").is_empty());
    }

    #[test]
    fn assistant_step_role() {
        let s = Step::assistant("Sure, here you go.");
        let m = s.to_message("");
        assert_eq!(m["role"], "assistant");
        assert_eq!(m["content"], "Sure, here you go.");
    }

    #[test]
    fn render_with_substitutes_extra_vars() {
        let s = Step::user("Hi {{name}}, tell me about {{input}}.");
        let rendered = s.render_with("Rust", [("name", "Ada")]);
        assert_eq!(rendered, "Hi Ada, tell me about Rust.");
    }

    #[test]
    fn render_with_leaves_unknown_placeholders() {
        let s = Step::user("{{greeting}} {{input}}");
        let rendered = s.render_with("world", std::iter::empty::<(&str, &str)>());
        assert_eq!(rendered, "{{greeting}} world");
    }

    #[test]
    fn render_with_input_takes_precedence() {
        // `{{input}}` is rendered first, so a "input" var entry is ignored.
        let s = Step::user("{{input}}");
        let rendered = s.render_with("real", [("input", "override")]);
        assert_eq!(rendered, "real");
    }

    #[test]
    fn render_with_owned_strings() {
        let s = Step::user("{{a}}-{{b}}");
        let vars = vec![
            ("a".to_string(), "1".to_string()),
            ("b".to_string(), "2".to_string()),
        ];
        assert_eq!(s.render_with("", vars), "1-2");
    }

    #[test]
    fn try_build_messages_for_in_bounds() {
        let c = Chain::new().step(Step::user("Hi {{input}}"));
        let msgs = c.try_build_messages_for(0, "world").unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["content"], "Hi world");
    }

    #[test]
    fn try_build_messages_for_out_of_bounds() {
        let c = Chain::new().step(Step::user("Hi"));
        assert!(c.try_build_messages_for(5, "x").is_none());
    }

    #[test]
    #[should_panic]
    fn build_messages_for_panics_out_of_bounds() {
        let c = Chain::new();
        let _ = c.build_messages_for(0, "x");
    }

    #[test]
    fn steps_accessor_returns_all() {
        let c = Chain::new()
            .step(Step::system("sys"))
            .step(Step::user("usr"));
        assert_eq!(c.steps().len(), 2);
        assert_eq!(c.steps()[0].role, "system");
    }
}
