# Hermes Agent adapter

Verified against **Hermes Agent v0.18.2** talking to `llama-server` (Gemma 4 31B) through
the tuningfork proxy. Everything below was read out of an actual recording, not from docs.

Implementation: `tuningfork/harness/hermes.py`. Template for other harnesses:
[`TEMPLATE.md`](TEMPLATE.md).

## Launching

| Need | Flag |
|---|---|
| One non-interactive prompt | `hermes chat -q "<prompt>"` (top-level `hermes -z` also works) |
| Machine-readable output | `-Q` — suppresses banner, spinner and tool previews |
| Bound the run | `--max-turns N` |
| Preload skills | `-s name1,name2` |
| No TTY to approve tools | `--yolo` — only acceptable in a disposable workspace |
| Skip injected memory and rules | `--ignore-rules`, `--ignore-user-config` |
| Git isolation | `--worktree` |
| Filesystem checkpoints | `--checkpoints` (on `chat`, not top level) |

There is **no `--profile` flag.** `hermes profile use` sets a *sticky global default*, so it
must never be used for experiment isolation — it would change the behaviour of every other
Hermes on the machine.

## Isolation

`HERMES_HOME` relocates the entire home directory — config, memory, skills, curator state,
sessions, cron and gateway state. That is the isolation primitive; the adapter passes
`HERMES_HOME` and nothing else, dropping every other `HERMES_*` variable so a stray setting
in the parent shell cannot change what is being measured.

A minimal isolated home needs only `config.yaml`:

```yaml
model:
  default: <model-id>
  provider: custom
  base_url: http://127.0.0.1:8787/v1
providers: {}
fallback_providers: []
```

Never set `HERMES_SHARED_AUTH_DIR`, and never start a gateway in an experiment home: both
reach back into a real installation's credentials.

**Hermes refuses any model reporting fewer than 64,000 context tokens**, aborting before the
first request. The experiment server must be configured accordingly.

A fresh home also pays first-run setup costs on every batch run. Hermes downloads its
`tirith` command scanner from GitHub into `<HERMES_HOME>/bin` whenever a home lacks it, in a
background thread — so every isolated run fetched 38 MB from outside the proxy, and whether
the scanner was installed before the model's first shell command varied from run to run
(it fails open while missing). That is run-to-run variance with nothing to do with the
skill, so the adapter's `config.yaml` turns it off (`security: tirith_enabled: false`), the
same in every run.

## Finding it, and what it needs of the server

tuningfork runs the `hermes` it finds on `PATH`, unless the recordings folder's
`config.json` names another executable (`tuningfork connections harness hermes
--executable /path/to/hermes`, or the Connections page). It never reads or writes the
harness's own configuration to do so. Its version is read with `hermes --version`, run in a
throwaway home — `HOME`, `HERMES_HOME`, `CLAUDE_CONFIG_DIR` and the XDG folders all
pointed there, no input, 15 s at most — so even detecting it leaves nothing behind in a
real installation. Verified: detecting both harnesses changed no file under the real
`~/.hermes`, `~/.claude` or `~/.config`.

The adapter declares what it needs of the inference server (`requirements`), and a batch
or a fork refuses to launch it against a server that lacks it — before a launch is
recorded, not after a failed run:

| Requirement | Why |
|---|---|
| `n_ctx` of at least 64,000 | Hermes refuses a model reporting less, before its first request (above) |
| Streaming tool calls | Hermes streams every call and acts on the tool calls in the stream |

What a server offers is measured, not assumed: `tuningfork connections check` (or *Check
the server* on the Connections page) makes one streaming tool call and one `/v1/messages`
call through the proxy, and keeps the answers with the server's identity. Until a server
is checked, a launch goes ahead with a note that the requirement was not measured.

## Payload anatomy

From a recorded two-step run (create a file), `write_file` called on step 1.

### What Hermes actually sends

```json
{"model": "...", "messages": [...], "tools": [...],
 "max_tokens": 65536, "stream": true,
 "stream_options": {"include_usage": true}}
```

That is the **complete** set of top-level keys. Two consequences matter:

1. **No `temperature`, no `top_p`, no `seed`.** Hermes sends *no sampling parameters at
   all*, inheriting whatever the server defaults to. Resampling cannot default to the
   sampling parameters the harness used — there are none — so reproducing a run means
   capturing the **server's** defaults as well as the request.
   llama.cpp exposes them at `/props` under `default_generation_settings`.
2. **`max_tokens` is set to the whole context window** (65536 here, matching `n_ctx`), not
   to a generation budget. Resampling therefore has no natural cost ceiling from the
   harness, and tuningfork must impose its own.

### Message layout

Two messages on the first step: `system`, then `user`. Hermes puts everything in the system
prompt rather than spreading it across turns.

The system prompt was 8,895 characters, with these sections:

- `# Finishing the job`
- `# Parallel tool calls`
- `## Mid-turn user steering`
- `# Tool-use enforcement`
- `# Google model operational directives` — **model-family-specific**, so the system prompt
  is not constant across models. A wording experiment must record which variant was in play.
- `Follow these operational rules strictly:`

Token budget for that first step: **13,150 prompt tokens** for 8,895 characters of system
prompt plus 18 tool schemas. Tool definitions are therefore roughly a third of the context
before the conversation starts.

### Tools

18 in the default toolset:

```
clarify, cronjob, delegate_task, execute_code, memory, patch, process,
read_file, search_files, session_search, skill_manage, skill_view,
skills_list, terminal, text_to_speech, todo, vision_analyze, write_file
```

`delegate_task` is the subagent mechanism — the thing to watch when working out how
subagent traffic appears on the wire.

### Skills

Two routes into the context, both verified from recordings:

**Loaded mid-run.** With no skills preloaded, no skill text appears in the system prompt.
Skills are reached through the `skills_list` / `skill_view` / `skill_manage` tools, so a
`SKILL.md` body enters the context as a **tool result** — progressive disclosure, as the
brief anticipates.

**Preloaded with `-s`.** The skill file is inlined **verbatim — frontmatter included — at
the end of the system prompt**, bracketed by two markers Hermes writes itself:

```
[IMPORTANT: The user launched this CLI session with the "<name>" skill preloaded. Treat
its instructions as active guidance for the duration of this session unless the user
overrides them.]

<the complete SKILL.md, byte for byte>

[Skill directory: <HERMES_HOME>/skills/<name>]
Resolve any relative paths in this skill ... against that directory.
```

Preloaded, the skill arrives as a literal copy of the file, which is what makes a wording
A/B tractable: `tuningfork ab` locates the skill by exact match and replaces it, with no
fuzzy matching to go wrong. A copy the model fetched again through `skill_view` (below) is
the same text JSON-escaped inside a tool result; the A/B finds and replaces that one too,
escaped the same way, so the variant arm never reads both wordings.

The system prompt also carries **volatile fields**: the session date
(`Conversation started: <weekday, date>`) and the per-run skill directory path. Two runs of
the same task therefore never share a system prompt, which matters for anything comparing
contexts across runs — and means a resample reproduces the original run's date, not
today's, because it resends the recorded request.

The system prompt also instructs the model to *patch skills it finds wanting*, via
`skill_manage`. That is §6.8's risk in the open: the agent can rewrite the thing under test
during the run. Hashing skill files before and after every run is not optional.

A skill reached through `skill_view` arrives as a JSON tool result —
`{"success": true, "name": ..., "content": "<the SKILL.md text>", ...}` — so the adapter
decodes it before the UI can mark it. It has no character offsets in the message, because
the text had to be unescaped out of a string first.

## Holding a call

Phase 3 breakpoints hold a call while a person decides. Whether the harness tolerates that
is the harness's business, so it is recorded here. From the Hermes 0.18.2 source:

| Setting | Default | For a local endpoint |
|---|---|---|
| `HERMES_API_TIMEOUT` — the whole request | 1800 s | 1800 s |
| `HERMES_STREAM_READ_TIMEOUT` — between bytes of a stream | 120 s | raised to the request timeout |
| `HERMES_STREAM_STALE_TIMEOUT` — stale-stream detector | 180 s | **disabled** |
| `HERMES_API_CALL_STALE_TIMEOUT` — non-streaming calls | 90 s | **disabled**, except for models with a reasoning floor |
| `HERMES_STREAM_RETRIES` | 2 | 2 |

Hermes counts loopback, RFC 1918 ranges, link-local and Tailscale addresses as local, and
tuningfork's proxy is on loopback, so a held call gets the lenient column — unless a value
was set explicitly, which switches the auto-adjustment off. The reasoning floor applies to
a fixed list of model names (`deepseek-r1`, `qwen3`, `o3`, …; Gemma is not on it), and only
to non-streaming calls; Hermes streams. In practice the ceiling is the 30-minute request
timeout — and, in a batch, the task's own `timeout`.

tuningfork does not rely on the lenient column. A held *streaming* call starts its response
at once and sends an SSE comment every `keepalive_interval` seconds (10 by default), so a
harness — or anything in between — with an idle-read timeout sees traffic throughout. The
cost is that such a call always answers HTTP 200; a failure while held arrives as an SSE
`error` event, which the OpenAI SDK raises as an `APIError`.

To raise the ceiling for long debugging sessions, Hermes reads per-provider and per-model
overrides from `config.yaml`: `providers.<id>.request_timeout_seconds`,
`providers.<id>.stale_timeout_seconds`, and `models.<model>.timeout_seconds`.

## Launches: every call, attributed as it arrives

A run tuningfork starts is a *launch* with a token of its own, and Hermes is given
`http://127.0.0.1:8787/_launch/<token>/v1` as its `base_url`. Verified: every call Hermes
0.18.2 makes keeps the prefix — its chat completions, the model-metadata probes it makes at
start-up (`/api/v1/models`, `/api/tags`, `/v1/props`, `/props`) and a subagent's calls.

## Subagents

`delegate_task` runs each child in the same process, as a fresh conversation with its own
system prompt, on the parent's base URL — so a child's calls reach the same launch, as a
thread of their own. The child's system prompt carries the `goal` and `context` the parent
passed, verbatim; that is how its thread is matched to the call that spawned it
(`Store.finalize_threads`). In 0.18.2 **every delegation runs in the background**: the
tool's own description says so, and its `background` parameter is ignored. `delegate_task`
returns `"status": "dispatched"` at once, the parent carries on, and the child's result
arrives later as a new message — so the child begins after the parent's next call, not
while it waits.

Two things found on the real stack:

- **On Python 3.14, `delegate_task` fails before the child makes a call**:
  `'DaemonThreadPoolExecutor' object has no attribute '_initializer'`, returned to the model
  as the tool's result. Hermes installed with `uv tool install` on Ubuntu 26.04 gets
  Python 3.14. The same Hermes on Python 3.12 (`uvx --python 3.12 --from hermes-agent==0.18.2
  hermes`) delegates as described.
- **In `-q` mode Hermes exits when the parent's turn ends**, and a child still running is
  cut off mid-call: its step is recorded as `cut_off`, and the run is not counted, since it
  measured the harness quitting rather than the skill. A batch run's delegation therefore
  completes only if the parent's turn outlasts the child. In the validation, both runs asked
  to delegate ended their turn first, even when the prompt said to wait for the answer, so
  there was no complete run to fork.

## Replaying a recorded run (forks)

A fork answers a relaunched Hermes from the recording until the step it goes live at. What
differs between two launches of one task, besides per-run paths, is declared in the
adapter's `volatile` list and masked when requests are compared:

- the session date in the system prompt (`Conversation started: …`);
- how long a tool took: `duration_seconds` in `execute_code`'s result and in
  `delegate_task`'s (`total_duration_seconds` too).

With those masked, a relaunched Hermes sent exactly the recorded requests at every replayed
step.

## Still to verify

- The context-compaction signature
- Whether `--pass-session-id` puts the session id in the system prompt
