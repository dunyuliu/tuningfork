# Tuning Fork

A step debugger and experiment lab for agent skills on local LLMs. What belongs to one
machine — its safety rules, its paths, its notes — goes in an untracked `CLAUDE.local.md`
beside this file (ignored by `.gitignore`), never here: this file is public.

Experiments run against llama.cpp's `llama-server`, by default on port 8081. For Gemma 4 (or
any model with sliding-window layers) start it with `--swa-full`. Without it, the same
request's behaviour depends on what the server processed before it, and a screen favours its
proposals. It costs memory: the 26B-A4B with two 64k slots fits a 32 GB card with it, and the
31B at 64k does not. Start it with `--no-cache-prompt` too: otherwise a Claude Code step's
samples share the first one's cached processing (its Messages endpoint ignores
`cache_prompt`), and so do forks with matching prompts — every fork, in the sandbox. The
server check measures it (`fresh_prompts`).

## Commands

    uv run pytest                # the test suite — no GPU or model required
    uv run tuningfork serve --upstream http://127.0.0.1:8081   # the recording proxy; prints the UI address
    uv run tuningfork probe <base-url> [--out profile.json]    # what a server can do

Inspecting and experimenting (all take `--home <recordings dir>`):

    tuningfork runs                       # recorded runs (--called T, --failed: pick one to study)
    tuningfork show <run>                 # a run's steps and what the model did
    tuningfork timeline <run>             # the run as one conversation, each step in its place
    tuningfork step <run> <k>             # one step: exact context in, behaviour out
    tuningfork diff <runA> <runB>         # two runs side by side
    tuningfork divergence --task <id>     # first step where runs stopped agreeing
    tuningfork batch <spec.json> -n 8     # run a task N times through the harness
    tuningfork batch ... --skill <file>   # the same, with <file> as SKILL.md: confirm an A/B
    tuningfork batch ... --concurrency 4  # runs at once; each is its own launch
    tuningfork batch ... --eval 2 --include-held-out   # a held-out eval: never the default
    tuningfork tasks                      # pass rates: one per harness, model, skill, server
    tuningfork resample <run> <k> -n 20   # re-run step k N times, history held fixed
    tuningfork resample ... --logprobs 8  # ... keeping each sample's tokens (chat-completions steps)
    tuningfork ab <run> <k> --variant F   # swap an edited skill in at step k, test the shift
    tuningfork resample/ab ... --to-end   # each sample a fork run to the end; the check decides
    tuningfork tokens <run> <k> --score   # the answer as tokens: how sure the model was of each
    tuningfork forking-tokens <run> <k>   # force alternatives at uncertain tokens: which flip it
    tuningfork fork <run> <k>             # replay a run up to k, live from there (breakpoints on)
    tuningfork files <run> <k|start|end>  # the workspace as a call found it (--diff, --out)
    tuningfork recheck <run> [--check …]  # try a check on a run's end state; records nothing
    tuningfork experiments / experiment   # list, or re-report without new samples
    tuningfork servers                    # which servers answered, and what each was
    tuningfork skills / skill <name>      # each version of a skill: text, pass rates (--text, --diff A B)
    tuningfork glossary [word]            # tuningfork's words, each defined once
    tuningfork trash run|experiment|job|scenario|refinement <id>   # set aside: out of every list and rate
    tuningfork trash list | restore <id> | purge <id> | empty   # deletion is from the trash, for good

Moving something to the trash changes a mark and nothing else: it leaves every list and every
statistic (`Store.runs`, `experiments`, `jobs` leave it out), what is tied to it stays where it
is, and no new work starts on a run there (`ops.trash.refuse_trashed`). Deleting from the
trash is permanent: the item goes with what is tied to it (`Store.purge_closure`: a run's
threads, the experiments at its steps and their samples, forks tuningfork launched from it,
token streams, launches and their folders), then what nothing left refers to — message nodes,
blobs, snapshot objects — with a two-minute grace for a recording in flight. A refinement goes
with what it made (`refine.made_by`): its screens, its validations' forks and fresh training
runs, its proposals' saved texts and its folder — never the run it studies, nor the changelog
and old versions an approval wrote beside the skill's folder. One in the trash is shown as it
was and takes no new work (`refine.refuse_closed`).

A shell check runs in the run's workspace with `$TUNINGFORK_REPLY` (a file holding the final
reply) and `$TUNINGFORK_SPEC_DIR` (its spec's folder, where a check script can live out of the
agent's sight). A batch interrupted — Ctrl-C, or a SIGTERM, which the runner turns into one —
kills the harness it was waiting on: each runs in a session of its own.

Resampling sends straight to the inference server (`TUNINGFORK_UPSTREAM`, default :8081) —
never a protected endpoint (`endpoints.py`: none until a person lists one). An A/B must declare its target before sampling
(`--target <letter>` from an earlier resample, or `--target-match <regex>`); without one
the result is exploratory and Bonferroni-corrected. Run to the end, the target is the task's
check. A pattern or described target also reads what a behaviour says (`Outcome.text`): a
script's code, which its key keeps only as a syntax-tree digest; one stored before that (no
`code` in its spec) matches as it was declared. Every sample's prompt is processed whole where llama.cpp takes `cache_prompt: false` —
chat completions, not its Messages endpoint, which drops the field — so no sample depends on
what the server ran before (`--prompt-cache` reuses it). Each arm draws a random seed base,
recorded with it; `--seed-base` draws an earlier experiment's samples again.

Refining a skill at a step — the target first, then proposed edits, each screened, a survivor
validated with full runs, and approved by a person:

    tuningfork refine start <run> <k> --target-rules '[…]'    # or --target <letter>, --target-match
    tuningfork refine show <id> [--packet]                    # proposals as diffs, verdicts; the evidence
    tuningfork refine propose <id> --builtin [-m 3] [--no-task]   # the proposer's edits ...
    tuningfork refine propose <id> --file SKILL.md --why '…'      # ... or a person's own wording
    tuningfork refine screen <id> -n 20 [--proposals a,b]     # each against a fresh original arm
    tuningfork refine validate <id> <proposal> [-n 5 --runs 5] [--dry-run] [--keep-private DIR]
    tuningfork refine approve <id> <proposal> [--validation v] [--spec S | --target F] [--confirm CODE]
    tuningfork refine show <id> --repair                      # finish an interrupted approval
    tuningfork refine list | close <id>
    tuningfork connections proposer show | set …              # which model proposes; --key-stdin

A refinement (`ops/refine.py`) is a folder, `<home>/refinements/<id>/`, written once wherever
it can be: the checkpoint, the run's copy of the skill, its target, and every proposal. A
proposal is at most 3 exact `{old, new}` edits (`ops/diff.py`) changing at most 10 lines —
a person's whole text only warns — never the frontmatter, and it must change what the model
saw at the step; it is saved as a variant, with who proposed it, why, and the packet they
were shown (`ops/packet.py`: built from the run's own task's records only, never sampled
for). A screen (`ops/screen.py`; `analysis/screen.py`) draws a fresh original arm first,
then each proposal, and tests each against it — Fisher, Holm over the proposals named
before sampling; an unfinished screen has rates and no verdict. The built-in proposer
(`ops/proposer.py`) asks the proxy's own server through the proxy (gated, never recorded)
unless `config.json` names another endpoint; one that is not this machine needs
`allow_remote`, and its key is `TUNINGFORK_PROPOSER_KEY` or `<home>/.proposer-key` (0600),
never anywhere else. A held-out task's run never shapes an edit.

A held-out eval (`tuningfork.held_out` in the spec) judges edits, so no agent may read its
runs: a batch never picks one, and runs one named only with `include_held_out` (warned; never
from MCP). A validation (`ops/validate.py`, job kind `validate`, EXCLUDED from MCP) runs one
screened proposal three ways, every arm drawn fresh against the original: forks from the
checkpoint to the end, full runs of the training task with the edit from the first step, and
full runs of each held-out eval through a private proxy (`ops/private.py`) that records into
a 0700 temp folder outside the home and is deleted after — its calls reach the server through
a relay launch the main proxy forwards and never records (`LaunchState.records`), and only
verdicts leave. It is refused when the skill's file, its other files or the training eval
changed since the checkpoint's run. Its record, `refinements/<id>/validations/<vid>.json`, is
what the gate reads (`analysis/validation.py`, G0–G8, each check with its counts and rule
name; timeouts count as failures, the not-worse checks are labelled low-powered, and a
held-out task the original fails must improve with the edit — G8, what caught a cheat). A job's
plan, result and progress carry no held-out verdict; `refine.view(held_out=True)` — the CLI,
the refinement's page — shows them, MCP's `show_refinement` never does.

Approving (`ops/approve.py`; the CLI, `POST …/approve` with the access token; never MCP) is
the only thing that writes a skill's own file. The target is worked out again from the run's
spec (a moved spec, or one leading elsewhere, needs `--spec`/`--target`), the gate must pass,
and the person types an 8-character code naming the proposal, the file and its bytes now.
Under a lock in `<skill>.history/`: `approval.json` written once with its evidence, the old
file kept in `<skill>.history/`, an atomic replace keeping the file's line endings and BOM,
an entry in `<skill>.changelog.md` — beside the skill folder, never inside it, which every
run copies — then `applied.json`. A crash between is reported by `refine show` and finished
by `--repair`, unless the file changed since. Deleting for good what a refinement cites says
so first (`refine.citations`); an approval keeps its evidence.

Token-level work needs llama.cpp's own endpoints (`/apply-template`, `/tokenize`,
`/completion`) and a model whose raw output tuningfork reads (`backends/formats/`: Gemma 4).
Scoring rebuilds a recorded answer as the model's tokens and forces it after the step's
exact prompt in one request (a grammar of token ids); the prompt must have the step's
recorded token count, or it is refused. The forking-token probe forces tokens from a stored
stream, and tests each alternative against its position's baseline, Holm-corrected over the
family planned before sampling. `tests/test_real_server.py` re-measures what all this stands
on against a real server: `TUNINGFORK_REAL_SERVER=http://127.0.0.1:8081 uv run pytest -m
real_server tests/test_real_server.py`; skipped otherwise.

A harness tuningfork starts is a *launch*: it talks to `/_launch/<token>/…` on the proxy, so
every call it makes — a subagent's included — is attributed to it as it arrives. A call the
harness adapter names a side call (`side_call`: Hermes's background review, which replays
the whole conversation after the answer) starts a side thread, never the task's next step,
and is never the run's final reply. A fork is a launch whose calls before step k are answered
from the recording (`tuningfork.replay`); the proxy's `--max-inflight` must be at least the
`--concurrency` of forks run through it. A launch is laid out as the harness installs itself
(an adapter's layout "install": Hermes in `~/.hermes`, a skill in the category folder it
declares; Claude Code in `~/.claude`), because the model reads those paths; each launch
records its layout, and a fork is laid out as its source was ("flat" before layouts were
recorded). Where bubblewrap runs, every launch runs in a sandbox (`sandbox.py`): it sees
its home as `/home/user` and works in `/home/user/project`, the same in every run, because
a few digits of a launch's own folder name were seen to change what the model did; the
rest of the machine read-only, `/tmp` and its processes its own, and the recordings folder,
the runs' folder and the spec's folder empty inside. A launch records its sandbox, a fork of
a sandboxed run is refused without one, and a batch's plan warns when there is none.
`TUNINGFORK_SANDBOX=off` turns it off — the suite's `conftest.py` does, and
`tests/test_sandbox.py` turns it back on where bubblewrap can run. The normalizer masks the
sandbox's paths as it masks a launch's own (`NORMALIZER_VERSION` 3).

Live breakpoints talk to a running proxy (`--proxy`, or `TUNINGFORK_PROXY`, default :8787):

    tuningfork break add --tool write_file --path '*.json'   # after the model: stop that call
    tuningfork break add --from-step 2                       # before the model sees step 2
    tuningfork break add --no-reasoning                      # after the model: a tool call, unreasoned
    tuningfork break list | rm | on | off <id>
    tuningfork hold wait | list | show <id> [--json]         # what is held, and why
    tuningfork hold continue <id> [--edited f.json]          # also: step, drop
    tuningfork pause / resume                                # refuse model calls meanwhile
    tuningfork ui --home <dir>                               # browse recordings, no proxy

Work started from the web UI runs as *jobs* inside `serve` (`jobs.py`), each on a worker
thread, its model calls sent through the proxy's own gate:

    tuningfork jobs list | show <id> | cancel <id> | wait <id>
    tuningfork serve --jobs N                                # background jobs at once (default 1)
    tuningfork token [--raw | --mcp | --header-json]         # the UI address with its token; the MCP token
    tuningfork connections show | switch <url> | harness <name> --executable P | check

The same operations, as tools for an agent over MCP (`tuningfork.mcp`, the SDK an optional
extra, `tuningfork[mcp]`; nothing else imports it):

    tuningfork mcp --home <dir> [--proxy URL] [--tools evidence|refine|all]   # stdio
    tuningfork serve ... --mcp-tools evidence|refine|all|none                 # also at /_tuningfork/mcp

Reads answer from the recordings in the MCP process; `start_*` tools are job kinds, sent to
the proxy recording into the same folder. Every job kind needs a decision in
`mcp/schema.py` — `EVIDENCE` (calls the model only), `HARNESS` (runs a harness or a check on
the machine: `--tools all`) or `EXCLUDED` with a reason — or `tests/test_mcp_schema.py`
fails. The default toolset is the evidence kinds but `screen`; `--tools refine` is the
refinement loop: start_refinement, show_refinement and propose_edit (which act in the MCP
process and call no model), resample, A/B and screen. No agent is offered the built-in
proposer — sending recordings to an endpoint is the person's decision — nor anything that
approves an edit. An answer is a compact view (`ops/compact.py`) whose
sentences lead both its text and its structured content (`summary`): Claude Code shows the
model only the structured content. Point a driving agent's own model at the inference
server, never at the proxy it studies.

A tool call made without reasoning is marked only where the same run reasons at another step
(`analysis/reasoning.py`), with whether a tool call had just failed — a reply in text without
reasoning is the usual end of a run, and is not; an experiment counts, per behaviour, the
samples that reasoned, said where some did not. The web UI's front page is `#/guide` — setup, where
you left off, a map of the loop, the ways in, and the glossary, whose words live once in
`ops/glossary.py`. The Skills page (`ops/skills.py`) names a skill by its folder, as variants
and refinements do; it is the person's view, never offered over MCP: the changelog an
approval writes carries held-out verdicts.

Every state change under `/_tuningfork/` needs the proxy's access token
(`x-tuningfork-token`, or `Authorization: Bearer`): minted each `serve`, written to
`<home>/.access-token` (0600), and checked in `api/guard.LocalOnly`, never in a route. Reads
stay open, except the MCP endpoint, which takes its own MCP token on every method
(`<home>/.mcp-token`, pinned by `TUNINGFORK_MCP_TOKEN`) and nothing else: the access token is
refused there, and the MCP token everywhere else, so an MCP client never holds a key to what
its tools leave out. The CLI finds both itself; tests send them
(`tests/helpers.control_headers`, `mcp_headers`). One proxy per home: two would overwrite
each other's tokens.

Every operation lives once, in `ops/`: a function with typed parameters, progress events,
cancellation, and a result that knows its JSON form. The CLI, the HTTP API behind the web UI
(`/_tuningfork/ui/`) and the MCP server (`mcp/`) are thin front ends over it, and every
function is meant to be reachable from each: logic a second front end would need does not
belong in `cli.py`, `api/` or `ui/`. The job kinds a front end can start are listed once,
in `ops/catalog.py`, with strict parameters (`GET /jobs/kinds` serves them). Only a call a
breakpoint could apply to leaves the proxy's ordinary streaming path, so nothing changes for
traffic while no breakpoint is armed.

The test suite must always pass with no GPU and no model loaded. That is the contribution
gate for an outside PR, so a test that needs real inference belongs behind a marker, not in
the default run. Check pytest's **exit code**, not just the pass count: warnings are errors,
and an unclosed SQLite connection fails the run at teardown after every test has passed.

## Layout

    src/tuningfork/ops/        every operation, once: experiments, forks, batch, holds (tools at
                               a held call), scenarios, variants, connections, inspect, timeline
                               (a run as one conversation; a step's detail), checks (a check
                               tried before it is trusted), tokens (an answer as tokens, scored),
                               forking (the forking-token probe), targets (what a target would
                               count among the behaviours seen at a step), trash (set aside,
                               restored, deleted for good), compact (results small enough for an
                               agent, intervals intact), refine (a skill refined at a step: its
                               target, proposals and screens), packet (the evidence a proposer
                               is shown), screen (several wordings against one baseline),
                               proposer (the built-in proposer: endpoint, consent, key), diff
                               (line diffs; exact edits), validate (a proposal's full runs:
                               forward, training, held out), private (held-out runs, verdicts
                               only, through a private proxy), approve (the gate met, the code
                               typed, the skill's file written back, history and changelog
                               beside it), skills (each skill's versions, texts and rates; its
                               folder now), glossary (the words, once); the job catalog;
                               `common`: events, cancellation, strict parameters
    src/tuningfork/jobs.py     the job runner inside `serve`: lanes, progress, cancel, sweep
    src/tuningfork/access.py   the access token and the MCP token; endpoints.py: protected endpoints
    src/tuningfork/paths.py    paths as a person types them (a Windows path under WSL, relative
                               to what), read the way tuningfork will open them
    src/tuningfork/proxy/      recording proxy (app, sse, recorder, held: the breakpoint path;
                               launches: calls attributed on arrival; replay: a fork's answers;
                               local: the proxy as in-process work sees it)
    src/tuningfork/launches.py launch tokens and URLs, and a launch's threads
    src/tuningfork/sandbox.py  every launch in bubblewrap, seeing /home/user and /home/user/project
    src/tuningfork/replay.py   what a fork replays, masked requests, path rewriting
    src/tuningfork/snapshots.py workspace snapshots in the home's own git object store
    src/tuningfork/breakpoints/ conditions, the hold engine, the live event bus
    src/tuningfork/api/        HTTP API for the UI and CLI: control, data, jobs, connections,
                               scenarios, trash, refinements (and the proposer's settings);
                               guard (Host, Origin, token)
    src/tuningfork/mcp/        the operations as MCP tools: schema (which kinds, how an agent
                               may call them), tools (the registry; reads), server (the SDK's
                               low-level Server: parsing, refusals, waits), backend (jobs in the
                               proxy, or over its API), stdio, http (the route on the proxy)
    src/tuningfork/control.py  Python client for a running proxy (what `break`/`hold` use)
    src/tuningfork/ui/static/  the web UI: plain HTML/CSS/JS modules, no build step; markdown.js
                               renders Markdown as DOM nodes, never as HTML
    src/tuningfork/wire/       wire formats: OpenAI chat-completions, Anthropic Messages; raw
                               completions (model load the proxy gates, never recorded)
    src/tuningfork/backends/   server capability profiles + runtime probe; raw.py, the raw path
                               (render, score, complete from token ids) and llamacpp/, its
                               llama.cpp implementation and the server's Messages conversion,
                               mirrored; formats/, model output formats (Gemma 4)
    src/tuningfork/harness/    harness adapters (hermes, claude-code, generic)
    src/tuningfork/store/      content-addressed blobs + SQLite message tree; token streams
    src/tuningfork/analysis/   normalizers, Wilson/Fisher/Holm, distributions, divergence;
                               tokens (entropy, margin, the step's sampler, redaction), forking;
                               screen (each proposal against one baseline, Holm over the family);
                               validation (the gate a validated edit meets, check by check);
                               reasoning (a step the model answered without it)
    src/tuningfork/experiment/ resampling, skill substitution, A/B orchestration; scoring an
                               answer; the forking-token probe
    src/tuningfork/runner.py   batch runs and forks (fresh workspace + home per launch)
    src/tuningfork/specs.py    task specs (evals.json convention)
    tests/mock_server/        scriptable mock: OpenAI chat-completions and Anthropic Messages;
                               toy.py, a toy llama.cpp (tokenizer, template, a model with forks)
    tests/fixtures/formats/   raw output beside the server's parse of it; capture.py remakes them
    tests/fake_agent.py       a real agent process for end-to-end tests of batches and forks
    tests/browser/            opt-in browser tests (Edge/Chromium over DevTools) and demo servers
    examples/                 a skill that mostly fails, and a step-by-step guide (README.md);
                               hermes-design-md/, Hermes's own design-md skill fixed from the
                               failing step to an approved edit, with its guide

A request is never stored whole: messages live in the content-addressed tree and are
rebuilt by `Store.rebuild_request`. Never report a rate without its interval — see
`analysis/stats.py`.

Server- and harness-specific knowledge lives behind the `backends/`, `harness/` and `wire/`
seams, never in the core. Supporting another harness or inference server must not require
editing anything under `proxy/`. A request is stored in its own wire format, exactly as
sent; responses are also kept in one common shape (an OpenAI `chat.completion`), which is
what outcomes, conditions and the UI read. Features degrade on a probed capability rather
than on a hardcoded assumption about which server is running.

## Conventions

- Python 3.12+, managed with uv. Under WSL, keep the venv and the recordings on the Linux
  file system, never under `/mnt/c`: DrvFs breaks SQLite's WAL locking.
- Nothing of one machine in a tracked file: real paths, hostnames, ports that serve something
  else, notes. Those go in untracked `*.local.md` files.
- The UI puts every string on the page as a text node — payloads are full of markup-looking
  text. `tests/test_ui.py` fails on any HTML-parsing sink (`innerHTML` and friends).
- tuningfork's own endpoints (`/_tuningfork/`) answer only to a loopback Host, refuse
  cross-site state changes by Origin, and accept no state change without the access token
  (`api/guard.py`): a loopback port is reachable from any page in the user's browser. Keep new endpoints under
  that prefix. The UI is served with a strict CSP — its own scripts only, nothing inline —
  and only `ui/static/js/api.js` talks to the server or touches the token.
