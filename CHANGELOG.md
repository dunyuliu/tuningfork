# Changelog

## 0.1.0 — the first public release

A step debugger and experiment lab for agent skills on local LLMs, pre-alpha. What it does:

- **Records** every model call of an agent harness, through a proxy that speaks OpenAI
  chat-completions and Anthropic Messages, with no change to the harness. Hermes Agent and
  Claude Code have adapters; any harness that takes a base URL can be recorded.
- **Finds where runs part**: a batch of runs of a task, judged by the task's check, and the
  first step where they stopped agreeing.
- **Tests a step, not a whole run**: resamples a step with its history held fixed, swaps an
  edited skill in at that step (an A/B with its target declared before sampling), and follows
  either to the end of the run, where the task's check decides. Every rate comes with its
  interval.
- **Stops a run where it matters**: live breakpoints before or after the model, at a call that
  matches, with the call held to read, edit, step or drop.
- **Restarts a run from any step**: the workspace as each call found it, kept in snapshots;
  forks replayed from the recording up to a step and live from there; every launch in a
  bubblewrap sandbox that looks the same each time.
- **Reads inside a step**, on llama.cpp: how sure the model was of each token of an answer,
  and which token decided the step.
- **Refines a skill at the step that fails**: a target declared first; edits proposed by a
  person, an agent or the built-in proposer; each screened against the original; a survivor
  validated with full runs, held-out tasks included; and written to the skill only when a
  person approves, typing a code.
- **All of it from a web UI**, or as tools for an agent over MCP.

Two guides walk through it on real runs: `examples/README.md`, a toy skill made to fail one
way, and `examples/hermes-design-md/`, Hermes Agent's own design-md skill fixed from its
failing step to an approved edit.
