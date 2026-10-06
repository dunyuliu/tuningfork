# Tuning Fork, step by step

This folder holds one small skill that mostly doesn't work, and everything needed to watch
tuningfork find out why. Follow it top to bottom and you will have used every part of the
tool.

- **Part 1 — no GPU, about five minutes.** Install, browse a set of recordings, and stop a
  live model call at a breakpoint and change it. The "model" is a script, so what you see
  matches this page exactly, apart from ids and times.
- **Part 2 — the real experiment, about half an hour on one GPU.** Record eight runs of the
  skill on Gemma 4 26B-A4B, find the decision that fails, resample it, test a rewording at
  exactly that step, follow the rewording from there to the end of the run, confirm the fix
  with full runs, and restart a run at the failing step to catch it live — then start the
  same work from the browser.
- **Part 3 — inside a step, about ten minutes more.** Read an answer as the model's own
  tokens, and how sure it was of each; then, where runs split, look for the token that
  splits them.
- **Part 4 — let tuningfork propose the fix, about twenty minutes more.** Say what the
  failing step should do, have a model propose edits from the evidence — or write your own
  — screen each one at that step against the original wording, validate a survivor with
  full runs, and approve it into the skill.

Every output below was produced by running this page's commands, as written, on a fresh
copy of the repository. Home directories are shortened to `/home/you`, the user's name to
`you`, the WSL distro's to `Ubuntu` and the proxy's port to its default, 8787; long outputs
are trimmed where marked `...`.

The skill here is a toy, made to fail one way. For a real one — Hermes Agent's own
`design-md` skill on Gemma 4 26B, from the failing step to an approved fix and the runs after —
see [`hermes-design-md/`](hermes-design-md/README.md).

## What is in this folder

| File | What it is |
|---|---|
| [`tally-report/SKILL.md`](tally-report/SKILL.md) | The skill under test |
| [`tally-report.evals.json`](tally-report.evals.json) | The task: a prompt, an input file, and a pass/fail check |
| [`variants/tally-report/SKILL.v2.md`](variants/tally-report/SKILL.v2.md) | The same skill, reworded to fix it |

The skill describes its job correctly and completely — read a CSV, count the line items,
total the amount column, write `report.json` — but never says **what to call the keys**. The
check wants `row_count` and `amount_total`. The model writes `count` and `total`.

The model on this page also fails a second way, which the skill was not written to provoke;
Part 2 meets it.

The variant lives outside the skill's folder on purpose. A batch gives every run a copy of
the skill's whole folder, so a reworded copy kept beside `SKILL.md` would be an answer key
the model could read. (It sat there for a while; see the notes at the end.)

---

## Part 1 — tuningfork without a model

You need [uv](https://docs.astral.sh/uv/) and git; uv fetches Python 3.12 or later itself if
you don't have it. On Windows, work inside WSL, and keep the checkout in the Linux file
system (`~/...`), not under `/mnt/c`: SQLite's locking does not work there.

### 1. Install

```bash
git clone https://github.com/hhuang91/tuningfork.git
cd tuningfork
uv sync
source .venv/bin/activate
```

Optionally, run the test suite. It needs no GPU and no model:

```bash
pytest -q
```

### 2. Make some recordings to look at

```bash
python tests/browser/demo.py seed ~/tuningfork-demo
export TUNINGFORK_HOME=~/tuningfork-demo
```

```
seeded /home/you/tuningfork-demo: 10 runs, a resample and an A/B. Try: tuningfork runs --home /home/you/tuningfork-demo
```

This writes what nine runs of the example task would leave behind, as if recorded from Hermes
Agent, plus one experiment of each kind. They are made up: in this version of the story the
model gets the keys right two times in three, which gives more to look at. Part 2 measures
what really happens.

Every `tuningfork` command reads its recordings from `$TUNINGFORK_HOME` (by default
`~/.tuningfork`), or from `--home`.

### 3. Look around from the terminal

The recorded runs, newest first:

```bash
tuningfork runs
```

```
 run       label                  steps  model        started      check
 44b59982  -                      1      gemma-4-31b  09-25 23:12  -
 46591e6e  tally-report/1#demo.8  3      gemma-4-31b  09-25 23:12  pass
 4f32e400  tally-report/1#demo.7  3      gemma-4-31b  09-25 23:12  pass (not counted)
 8f0ca133  tally-report/1#demo.6  3      gemma-4-31b  09-25 23:12  pass
 3142caaa  tally-report/1#demo.5  3      gemma-4-31b  09-25 23:12  pass (not counted)
 61583323  tally-report/1#demo.4  3      gemma-4-31b  09-25 23:12  fail
 e55d8dc1  tally-report/1#demo.3  3      gemma-4-31b  09-25 23:12  pass
 8e136eac  tally-report/1#demo.2  3      gemma-4-31b  09-25 23:12  fail
 ab0a318d  tally-report/1#demo.1  3      gemma-4-31b  09-25 23:12  pass
 115c2386  tally-report/1#demo.0  3      gemma-4-31b  09-25 23:12  pass

10 run(s), 28 step(s), 9 unique message(s) in /home/you/tuningfork-demo
```

Your ids and times will differ; the labels won't. Two runs are *not counted*: one was edited
at a breakpoint, and in the other the skill changed while it ran. Either way the verdict
measures more than the skill, so it is kept out of every rate. The pass rate, always with its
interval:

```bash
tuningfork tasks
```

```
tally-report/1 · hermes · gemma-4-31b
 skill     runs  check passed
 a606f71e  9     5/7  71%  [36%-92%]  2 not counted

skill: a fingerprint of the skill files the runs were given. Intervals are Wilson 95%.
```

One run, step by step. A run can be named by its id, a prefix of it, or its label:

```bash
tuningfork show "tally-report/1#demo.2"
```

```
run 8e136eac  tally-report/1#demo.2
  task    tally-report/1
  model   gemma-4-31b
  started 09-25 23:12:12
  harness hermes
  skills  a606f71e (1 file(s))
 k  outcome                                                       finish      ms      tokens  think
 0  read_file(path='data.csv')                                    tool_calls  2210.5  4145    29
 1  write_file(content='{"count":4,"total":61.5}',                tool_calls  4080.0  4342    58
    path='report.json')
 2  Done — the totals are in report.json.                         stop        1150.0  4274    -
```

One step: exactly what the model saw, and what it did.

```bash
tuningfork step "tally-report/1#demo.2" 1
```

```
run 8e136eac step 1
  model   gemma-4-31b
  params  {}
  server  {"n_ctx": 65536, "temperature": 1.0, "top_k": 64, "top_p": 0.95}
  timings {"ttft_ms": 1780.0, "total_ms": 4080.0}
  usage   {"prompt_tokens": 4180, "completion_tokens": 162, "total_tokens": 4342}

context
  [0] system: You are an agent.  [IMPORTANT: The user launched this CLI session with the
"tally-report" skill prel…
  [1] user: Use the tally-report skill on data.csv.
  [2] assistant: tool_calls(read_file)
  [3] tool: id,amount 1,10.00 2,20.50 3,15.00 4,16.00

response
  reasoning_content: Four rows; 10 + 20.5 + 15 + 16 = 61.5. Keys: count, total.
  tool_call write_file({"path": "report.json", "content": "{\n  \"count\": 4,\n  \"total\":
61.5\n}"})
```

`--full` prints whole messages. Now: where do the runs stop agreeing?

```bash
tuningfork divergence --task tally-report/1
```

```
9 runs of tally-report/1
  agreed for 1 step(s), diverged at step 1

step 1
   count  rate  95% CI     behaviour                                           then passed
A  6/9    67%   [35%-88%]  write_file(content='{"amount_total":61.5,"row_cou…  5/5  100%  [57%-100%]
                           path='report… (+7 chars)
B  3/9    33%   [12%-65%]  write_file(content='{"count":4,"total":61.5}',      0/2  0%  [0%-66%]
                           path='report.json')

Intervals are Wilson 95%. With this many runs they are wide; separating 60% from 80% takes about 82
runs per arm.
```

Every run did the same thing at step 0, and they split at step 1 — the write. *Then passed*
says which behaviour went on to pass the check. Behaviours are compared after normalization,
so `{"count": 4}` and `{\n  "count": 4\n}` are one behaviour.

Last, the two experiments — a resample of step 1, and a wording A/B there. `experiments`
lists them; `-q -n 1` picks the newest, the A/B:

```bash
tuningfork experiments
tuningfork experiment $(tuningfork experiments -q -n 1)
```

```
ab at run 115c2386 step 1 — after the result of read_file
  swapped   skills/tally-report/SKILL.md → skills/tally-report/SKILL.v2.md

   behaviour                                      original              variant               p adj.
A  write_file(content='{"amount_total":61.5,"ro…  12/20  60% [39%-78%]  19/20  95% [76%-99%]  0.039
   … (+19 chars)
B  write_file(content='{"count":4,"total":61.5}…  8/20  40% [22%-61%]   1/20   5% [1%-24%]    0.039
   path='repor… (+8 chars)
p adj.: Fisher exact per behaviour, Bonferroni-corrected for 2 comparisons

target (declared before sampling): behaviour matching /row_count/
  original  12/20  60% [39%-78%]
  variant   19/20  95% [76%-99%]
  Fisher exact p = 0.020 → significant at α=0.05, variant higher
  a screening result: confirm a winner with full runs

The 1 model-written message(s) before this step were generated under the original wording, and both
arms inherit them. A checkpoint A/B is a fast screen; confirm a winner with full runs.
```

### 4. The same, in a browser

```bash
tuningfork ui
```

```
tuningfork 0.0.1 — browsing /home/you/tuningfork-demo
  open http://127.0.0.1:8788/_tuningfork/ui/
```

Open <http://127.0.0.1:8788/_tuningfork/ui/>. **Runs** groups runs by task, with the pass rate
and its interval. Open run `…#demo.2`: its conversation, top to bottom, with each step's
answer where it happened and the skill marked in the system prompt, mapped to its file line
by line. Click step 1 for its details; *Exact context* shows everything that step was sent.
*Where do they diverge?* on the task, and the **Experiments** page, show the last two
commands' results. Stop it with Ctrl-C.

### 5. Stop a live call, and change it

Now a real tuningfork proxy, in front of the scripted model. It answers every call by
writing `report.json` with the wrong keys. In a **first terminal**:

```bash
cd tuningfork && source .venv/bin/activate
python tests/browser/demo.py proxy ~/tuningfork-live 8791
```

```
tuningfork proxy on http://127.0.0.1:8791/v1, in front of a mock model (no GPU)
  recording to /home/you/tuningfork-live
  web UI       http://127.0.0.1:8791/_tuningfork/ui/?token=…
  control it with TUNINGFORK_PROXY=http://127.0.0.1:8791 — Ctrl-C to stop
```

The web UI's address carries the proxy's *access token*: a new one each time a proxy
starts, and what lets a page change anything. Reading needs no token. The commands below
find it by themselves, in the recordings folder the proxy names; `tuningfork token` prints
the address again.

In a **second terminal**, point tuningfork's commands at that proxy, and set a breakpoint:
stop any `write_file` of a `.json` file after the model has decided on it, before the caller
acts on it.

```bash
cd tuningfork && source .venv/bin/activate
export TUNINGFORK_PROXY=http://127.0.0.1:8791 TUNINGFORK_HOME=~/tuningfork-live
tuningfork break add --tool write_file --path '*.json'
```

```
breakpoint e12afb3d — after the model: write_file on *.json
```

Make a model call the way a harness would, in the background, then wait for it to be held:

```bash
curl -s http://127.0.0.1:8791/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model": "demo", "messages": [{"role": "user", "content": "Tally data.csv"}]}' \
  > answer.json &
tuningfork hold wait
```

```
hold 58c57cb5 — after the model, a new run step 0, held 0.1s
  reason  after the model: write_file on *.json
  user: Tally data.csv

the model answered
  reasoning_content: The CSV has four rows; I will write the totals.
  tool_call write_file({"path": "report.json", "content": "{\n  \"count\": 4,\n  \"total\":
61.5\n}"})

tuningfork hold continue 58c57cb5 · step · drop — to edit the response: hold show 58c57cb5 --json >
f.json, edit, hold continue 58c57cb5 --edited f.json
```

`curl` is still waiting: to the caller, a held call is just a slow response. Fix the keys and
let it go on. `hold show --json` prints the held response; a tool call's arguments are JSON
text inside it, so this edits them with a few lines of Python:

```bash
HOLD=$(tuningfork hold list -q)
tuningfork hold show $HOLD --json > held.json
python - <<'EOF'
import json
held = json.load(open("held.json"))
call = held["choices"][0]["message"]["tool_calls"][0]["function"]
arguments = json.loads(call["arguments"])
arguments["content"] = '{"row_count": 4, "amount_total": 61.5}'
call["arguments"] = json.dumps(arguments)
json.dump(held, open("held.json", "w"))
EOF
tuningfork hold continue $HOLD --edited held.json
wait
python -m json.tool answer.json | grep arguments
```

```
58c57cb5: continue with your edit
                            "arguments": "{\"path\": \"report.json\", \"content\": \"{\\\"row_count\\\": 4, \\\"amount_total\\\": 61.5}\"}"
```

The caller received your version. The recording keeps both — what the model produced is the
step's behaviour, and the edit is noted beside it:

```bash
tuningfork show $(tuningfork runs -q -n 1)
```

```
run c3c5a23a
  task    -
  model   demo
  started 09-25 23:12:23
 k  outcome                                        finish      ms    tokens  think  note
 0  write_file(content='{"count":4,"total":61.5}…  tool_calls  0.45  18      47     response edited
    path='report.json')

An edited step's outcome is what the model produced; `tuningfork step <run> <k> --harness-view`
shows what the harness saw instead.
```

The browser makes this easier. Open the web UI address the proxy printed — the one with
`?token=` — and go to **Live**. Run the `curl` line again, and edit the held call's
`content` field right there — the page shows a file's content raw, not as escaped JSON —
then click *Continue with edits*. Opened without the token, the page shows everything and
changes nothing, and says how to paste the token in. Stop the proxy with Ctrl-C in the
first terminal.

---

## Part 2 — the real experiment

### What you need

- **An NVIDIA GPU** with about 27 GiB free: a 17 GB model and its context, with the whole
  sliding-window cache. We used one RTX 5090 (32 GB); the times below come from it.
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)**'s `llama-server`. We used build
  b11108; any recent build with Gemma 4 support should do.
- **A harness.** The commands below drive **Claude Code** — we used 2.1.281, which needs
  Node 18 or later: `npm install -g @anthropic-ai/claude-code@2.1.281`. To use **Hermes
  Agent** instead (`uv tool install hermes-agent==0.18.2`), see
  [the differences](#using-hermes-agent-instead) at the end. Either way, tuningfork runs the
  harness in a throwaway home directory against your local model only: it never sees your
  own settings or memory files, and needs no API key.
- **[bubblewrap](https://github.com/containers/bubblewrap)**, on Linux:
  `sudo apt install bubblewrap`. With it, every run tuningfork launches sees the same home and
  folder, `/home/user` and `/home/user/project`, as an installation would; without it, each
  run sees its own folders, whose names differ from run to run, and the model reads them. We
  used 0.11.1.

Everything stays on your machine: the model server, the proxy and the harness all listen on
127.0.0.1.

### 1. Start the model server

In a **first terminal**:

```bash
llama-server -hf lmstudio-community/gemma-4-26B-A4B-it-GGUF:Q4_K_M --no-mmproj \
  --alias gemma-4-26b-a4b --host 127.0.0.1 --port 8081 \
  -ngl 999 -c 131072 -np 2 -ctk q4_0 -ctv q4_0 --jinja --swa-full --no-cache-prompt
```

The first start downloads the model, about 17 GB. If you already have the file, use
`-m /path/to/gemma-4-26B-A4B-it-Q4_K_M.gguf` in place of `-hf ... --no-mmproj`. It is ready
when this answers `{"status":"ok"}`:

```bash
curl -s http://127.0.0.1:8081/health
```

That is two slots of 65,536 tokens each: Hermes, in Part 3, needs 64,000.

**Keep `--swa-full` and `--no-cache-prompt`.** Without either, the same request can behave
differently depending on what the server ran before it (see the
[README](../README.md#start-llama-server-with---swa-full)):

- Gemma 4 has sliding-window attention layers, and without `--swa-full` llama-server keeps
  only part of their cache. The flag costs memory: this command uses about 27 GiB with it, 22
  without. Gemma 4 31B at the same context does not fit a 32 GB card with it, which is why this
  page runs the 26B.
- Without `--no-cache-prompt`, llama-server answers a prompt from what it cached of an earlier
  one. Claude Code's calls cannot ask it not to, so every sample of a step would share the first
  one's processing — and in the sandbox, every fork of an arm too. The flag costs time: every
  step processes its whole prompt, about 1.5 s for this task's 14,000 tokens.

### 2. Start tuningfork

In a **second terminal**, the recording proxy — it sits between the harness and the model:

```bash
cd tuningfork && source .venv/bin/activate
export TUNINGFORK_HOME=~/tuningfork-guide
tuningfork serve --upstream http://127.0.0.1:8081
```

```
tuningfork 0.0.1
  listening on  http://127.0.0.1:8787/v1
  forwarding to http://127.0.0.1:8081
  recording to  /home/you/tuningfork-guide
  max in flight 1
  web UI        http://127.0.0.1:8787/_tuningfork/ui/?token=…
  MCP           http://127.0.0.1:8787/_tuningfork/mcp  (evidence tools; its own token, `tuningfork token --mcp`)
  the token lets the page change things; it is new each start, and `tuningfork token` prints this
address again
```

Leave it running. Open a **third terminal** for everything that follows, set up the same way:

```bash
cd tuningfork && source .venv/bin/activate
export TUNINGFORK_HOME=~/tuningfork-guide
```

Before anything runs, check that the server can do what the harness needs. Claude Code
speaks only Anthropic's Messages API, and Hermes streams every call and acts on the tool
calls in the stream; tuningfork measures both with two small calls, then whether the server
processes every prompt whole — and, with three more, what Part 3's token-level work needs:

```bash
tuningfork connections check
```

```
  streaming_tool_calls   yes — structured delta.tool_calls observed
  anthropic_messages     yes — a Messages response
  fresh_prompts          yes — a prompt sent twice was processed whole both times
  known_output_format    yes — gemma4
  prompt_equivalence     yes — rendered 68 tokens, the chat endpoint's prompt has 68
  raw_completion         yes — 2 tokens forced by a grammar of ids, with their top alternatives
```

A batch or a fork refuses to start a harness against a server that lacks what it needs —
Hermes also needs a context of at least 64,000 tokens — rather than finding out from a
failed run, and warns when the server answers prompts from its cache. `tuningfork
connections show` lists the server, what was measured of it, the harnesses found with their
versions, and whether launches run in the sandbox.

### 3. Record eight runs

```bash
tuningfork batch examples/tally-report.evals.json -n 8 --harness claude-code --model gemma-4-26b-a4b
```

Each run gets a fresh workspace holding `data.csv`, a fresh harness home with the skill
installed — which it sees as `/home/user/project` and `/home/user`, the same in every run —
and the task's check afterwards. Five to forty-five seconds a run:

```
tally-report/1 — 8 runs
  prompt: Use the tally-report skill to summarize data.csv into report.json.
  check:  python3 -c "import json,sys; d=json.load(open('report.json')); sys.exit(0 if
d.get('row_count')==4 and abs(float(d.get('amount_total',0))-61.5)<0.01 else 1)"
  skill:  examples/tally-report
  sandbox: every run works in /home/user/project, its home /home/user

  run 0  fail  6.7s
  run 1  fail  6.5s
  run 2  fail  40.4s
  run 3  fail  10.9s
  run 4  fail  29.5s
  run 5  fail  38.7s
  run 6  fail  43.5s
  run 7  fail  29.6s

0/8  0%  [0%-32%] passed

8 runs of tally-report/1
  agreed for 0 step(s), diverged at step 0

step 0
   count  rate  95% CI     behaviour                    then passed
A  6/8    75%   [41%-93%]  Bash(command='ls data.csv')  0/6  0%  [0%-39%]
B  2/8    25%   [7%-59%]   Bash(command='ls')           0/2  0%  [0%-66%]

Intervals are Wilson 95%. With this many runs they are wide; separating 60% from 80% takes about 82
runs per arm.
```

Every run failed, in two ways. All eight began by checking the file was there — six with
`ls data.csv`, two with `ls`; *then passed* says that made no difference — and then loaded the
skill. There they parted:

- **Three took loading the skill for doing the job.** Two answered straight away that the
  report was written; one read the file and looked for the report first. None wrote anything.
- **One kept looking for the report** the skill was supposed to have written, loading the skill
  twice more, until it reached the task's limit of eight model calls (`max_turns` in the spec).
- **Four went looking for it too, found none, and wrote it themselves** — one with `Write`,
  three with a Python script. Every report had the keys `count` and `total`, or
  `total_amount`.

The keys are the failure the skill was made for. Taking the loaded skill for the job done is
this model's own, which the skill was not written to provoke.

### 4. Find the decision that fails

Take the most recent run in which the model called `Write` — here, the only one — and list
what it did:

```bash
RUN=$(tuningfork runs -q -n 1 --called Write)
tuningfork show $RUN
```

```
run eca15f11  tally-report/1#001cec9a.2
  task    tally-report/1
  model   gemma-4-26b-a4b
  started 10-05 23:59:15
  harness claude-code
  sandbox /home/user/project, its home /home/user
  skills  c417ecd4 (1 file(s))
 k  outcome                                                     finish      ms        tokens  think
 0  Bash(command='ls data.csv')                                 tool_calls  2191.78   13786   177
 1  Skill(args='input_file: data.csv, output_file:              tool_calls  2187.32   13897   95
    report.json', skill='ta… (+12 chars)
 2  Bash(command='ls report.json')                              tool_calls  1765.85   13985   none
 3  Read(file_path='data.csv')                                  tool_calls  4450.5    14353   1105
 4  Skill(args='input_file: data.csv, output_file:              tool_calls  6271.24   14975   1634
    report.json', skill='ta… (+12 chars)
 5  Bash(command='ls -l')                                       tool_calls  12291.59  15446   4267
 6  Write(content='{"count":4,"total":61.5}',                   tool_calls  8230.39   16231   2134
    file_path='report.json')
 7  Read(file_path='report.json')                               tool_calls  2157.35   16331   61
```

This run is the batch in small. It checked the file was there and loaded the skill, then
looked for a `report.json` at once, as if loading the skill had written one. It read the
file, loaded the skill again — Claude Code answered that it was already loaded — and listed
the folder. Only then, at step 6, did it write the report itself, with `count` and `total`,
and read it back: its eighth call, the limit. The write is where the keys were chosen. Its
number depends on the path a run took — a run that writes straight after reading the file
does it at step 3 — so name the step by the tool the model called there instead: `Write` is
the first step where it called Write.

*think* is the length of the model's reasoning before each answer, in characters. `none`
marks a tool call made with no reasoning at all, in a run that reasons at other steps: here,
the look for a report straight after loading the skill. The four calls after it reasoned
over 9,000 characters between them.

```bash
tuningfork step $RUN Write
```

```
context
  [0] system: x-anthropic-billing-header: cc_version=2.1.281.606; cc_entrypoint=sdk-cli; You are a
Claude agent, b…
  [1] user: <system-reminder> Attribution for git commits and pull requests you create from here on
(this replac…
  [2] system: # Environment You have been invoked in the following environment:   - Primary working
directory: /ho…
  [3] assistant: tool_calls(Bash)
  [4] tool: data.csv
  [5] system: <total_tokens>14986214 tokens left</total_tokens>
  [6] assistant: tool_calls(Skill)
  [7] tool: Launching skill: tally-report
  [8] user: Base directory for this skill: /home/user/.claude/skills/tally-report  # Tally report
Use this when…
  [9] system: <total_tokens>14986103 tokens left</total_tokens>
  [10] assistant: tool_calls(Bash)
  [11] tool: Exit code 2 ls: cannot access 'report.json': No such file or directory
  [12] system: <total_tokens>14986015 tokens left</total_tokens>
  [13] assistant: tool_calls(Read)
  [14] tool: 1  id,item,amount 2        1,widget,10.00 3        2,gadget,25.50 4
3,widget,6.00 5 4,doohickey,20.00 6
  [15] system: <total_tokens>14985647 tokens left</total_tokens>
  [16] assistant: tool_calls(Skill)
  [17] tool: Launching skill: tally-report
  [18] user: Skill /tally-report is already loaded above; instructions unchanged. Arguments:
input_file: data.csv…
  [19] system: <total_tokens>14985025 tokens left</total_tokens>
  [20] assistant: tool_calls(Bash)
  [21] tool: total 4 -rw-r--r-- 1 you you 77 Oct  5 23:59 data.csv
  [22] system: <total_tokens>14984554 tokens left</total_tokens>

response
  reasoning_content: The `ls -l` command shows that `report.json` definitely does not exist.
The `tally-report` skill call returned the skill's instructions.
This might mean that when I call `Skill`, it doesn't "run" the skill as a program, but rather it's a
way to *invoke* a pre-defined procedure.
If the procedure is "… (+1834 chars)
  tool_call Write({"content": "{\n  \"count\": 4,\n  \"total\": 61.5\n}", "file_path":
"report.json"})
```

(The header lines above the context are left out here.) Message 8 is the skill as the model
read it: after `Base directory for this skill:`, the body of `SKILL.md`, frontmatter stripped,
from where Claude Code installs a skill — in the sandbox, every run's home is `/home/user`.
The reasoning says what the model had made of it: it expected the `Skill` call to run the
skill and write the report, and only now, finding none, read it as instructions to follow.
Then it wrote `count` and `total`.

What was on disk when the model made that call? Every call of a run tuningfork launched
finds the workspace snapshotted:

```bash
tuningfork files $RUN Write
```

```
run eca15f11 before step 6  (snapshot 894dd123)
   file      bytes
   data.csv  77
```

Only the input: the report does not exist yet. `tuningfork files $RUN end` lists what the
run left behind, `--diff` shows what changed since the step before, and `--out <dir>`
writes the files out.

### 5. Bad luck, or the wording? Resample that step

Send that step's exact context to the model twenty more times, with everything before it
held fixed:

```bash
tuningfork resample $RUN Write -n 20
```

```
run eca15f11 step 6 — 20 samples from http://127.0.0.1:8081
  context   23 messages, after the result of Bash
  sampling  the recorded conditions (the request's own parameters, else server defaults)
  budget    max_tokens 4096, concurrency 1
  estimate  the recorded step took 8.2s; at most ~2.7 min

    1/20  Write(content='{"count":4,"total":61.5}', file_path='report.json… (+2 chars)  14.4s
    2/20  Write(content='{"count":4,"total":61.5}', file_path='report.json… (+2 chars)  6.5s
    3/20  Write(content='{"count":4,"total":61.5}', file_path='report.json… (+2 chars)  9.9s
    4/20  Write(content='{"count":4,"total":41.5}', file_path='report.json… (+2 chars)  8.3s
    ...
   20/20  ListAgents()  4.7s

step 6, after the result of Bash — 20 samples
   count  rate  95% CI     behaviour
A  14/20  70%   [48%-85%]  Write(content='{"count":4,"total":61.5}', file_path='report.json')
B  3/20   15%   [5%-36%]   Write(content='{"count":4,"total":41.5}', file_path='report.json')
C  2/20   10%   [3%-30%]   ListAgents()
D  1/20   5%    [1%-24%]   Bash(command='python3 -c "import csv, json; data =
                           list(csv.DictReader(open(\'data.csv\'))… (+7 chars)

experiment 98e65286 · 2.8 min · 8.3s per sample · use `tuningfork ab eca15f11 6 --target <letter>
--variant <file>` to test a wording change
```

None of the twenty used `row_count`. Seventeen wrote the report with `Write`, all with
`count` and `total` — three of them with a total of 41.5, the first three amounts' sum — and
one wrote the same keys from a Python script. Two called `ListAgents`, a Claude Code tool
that lists the other agents it could message, and has nothing to do with the task. Not bad
luck: given this skill, the model never picks the keys the check wants.

Every sample reasoned before it answered, and the server processed the whole prompt every
time, 15,500 tokens: about eight seconds a sample. When some samples answer without
reasoning, a *reasoned* column says which.

### 6. Test a rewording at that step

The variant pins the output down, with exact key names and an example:

```bash
diff examples/tally-report/SKILL.md examples/variants/tally-report/SKILL.v2.md
```

Swap it in at the write and sample both wordings, twenty each. The target — *calls `Write`,
and mentions `row_count`* — is declared before sampling, so it gets one exact test rather
than a search through whatever happens to differ:

```bash
tuningfork ab $RUN Write --variant examples/variants/tally-report/SKILL.v2.md \
  --target-rules '[{"if": "calls", "value": "Write"}, {"if": "contains", "value": "row_count"}]' -n 20
```

```
run eca15f11 step 6 — 20 samples per arm from http://127.0.0.1:8081
  context   23 messages, after the result of Bash
  sampling  the recorded conditions (the request's own parameters, else server defaults)
  budget    max_tokens 4096, concurrency 1
  estimate  the recorded step took 8.2s; at most ~5.5 min
  swapping
/home/you/tuningfork-guide/batches/001cec9a/skill-snapshot/tally-report/SKILL.md →
examples/variants/tally-report/SKILL.v2.md
            found: body, lines 6-16, in message 6 (user), part 1 at chars 71-411
  target    a behaviour that calls Write, and mentions "row_count" (declared before sampling)

  original   1/20  Write(content='{"count":4,"total":41.5}', file_path='report.json… (+2 chars)
11.0s
  original   2/20  Write(content='{"count":4,"total":61.5}', file_path='report.json… (+2 chars)
10.0s
  ...
  variant    1/20  ListAgents()  2.8s
  variant    2/20  Write(content='{"amount_total":61.5,"row_count":4}', file_path='… (+13 chars)
6.4s
  ...

   behaviour                                       original             variant               p adj.
A  Write(content='{"amount_total":61.5,"row_coun…  0/20   0% [0%-16%]   10/20  50% [30%-70%]  0.006
   file_… (+19 chars)
B  ListAgents()                                    5/20  25% [11%-47%]  4/20  20% [8%-42%]    1.000
C  Write(content='{"count":4,"total":61.5}',       9/20  45% [26%-66%]  0/20   0% [0%-16%]    0.017
   file_path='repor… (+8 chars)
D  Write(content='{"count":4,"total_amount":61.5…  2/20  10% [3%-30%]   0/20   0% [0%-16%]    1.000
   file_path… (+15 chars)
...
p adj.: Fisher exact per behaviour, Bonferroni-corrected for 14 comparisons

target (declared before sampling): a behaviour that calls Write, and mentions "row_count"
  original  0/20   0% [0%-16%]
  variant   10/20  50% [30%-70%]
  Fisher exact p = 4.4e-04 → significant at α=0.05, variant higher
  a screening result: confirm a winner with full runs

The 6 model-written message(s) before this step were generated under the original wording, and both
arms inherit them. A checkpoint A/B is a fast screen; confirm a winner with full runs.
experiments 5e2c01e5 (original) and 53434921 (variant) · 5.5 min
```

tuningfork found the skill in the recorded request — the body, lines 6–16 of the file, where
Claude Code put it — replaced exactly that text, and sent both versions; nothing else
differed. It needed no `--original`: every batch run records the copy of the skill it was
given, and that copy is what gets swapped out. With the original, the same mix as the
resample. With the variant, half the samples wrote `row_count` and `amount_total` with
`Write`, four more wrote them from a script — `awk` or Python — and four called `ListAgents`,
as often as the original did.

A target reads a call's arguments as written, a script's code included, so
`--target-match row_count` would have counted the scripts too: 14 in 20. The rules ask for
what the step should do, a `Write` with the right key, and get 10 — still one exact test,
p = 4.4 × 10⁻⁴. The per-behaviour table tests every behaviour seen, fourteen here, and
corrects for them all.

### 7. Follow the rewording to the end of the run

That A/B tested one decision: everything before the write was held fixed, and nothing after
it was looked at. `--to-end` follows every sample to the finish. Each sample becomes a
*fork*: Claude Code, relaunched on the same task in a fresh workspace, is answered from the
recording up to the write — it runs the recorded `ls`, `Skill` and `Read` calls again
itself, but the model is not asked — then runs live from the write on, with the variant
swapped in, until the task's check decides:

```bash
tuningfork ab $RUN Write --variant examples/variants/tally-report/SKILL.v2.md --to-end -n 20
```

```
run eca15f11 from step 6 — 20 forks per arm through http://127.0.0.1:8787
  context   after the result of Bash
  replays   6 recorded call(s), answered from the recording; live from step 6
  harness   claude-code, model gemma-4-26b-a4b
  sandbox   every fork works in /home/user/project, its home /home/user, as its source did
  swapping
/home/you/tuningfork-guide/batches/001cec9a/skill-snapshot/tally-report/SKILL.md →
examples/variants/tally-report/SKILL.v2.md, from step 6 on
            found: body, lines 6-16, in message 6 (user), part 1 at chars 71-411
  target    the task's check passes (declared by the task, before any fork ran)

  original   1/20 8544f00f  fail  8.7s  Bash(command='echo \'{"count": 4, "total": 61.5}\' >
report.… (+6 chars)
  original   2/20 38ef1828  fail  10.5s  Write(content='{"count":4,"total_amount":61.5}',
file_path='… (+13 chars)
  ...
  variant    1/20 bdb4e605  pass  14.7s  Write(content='{"amount_total":61.5,"row_count":4}',
file_pa… (+17 chars)
  ...
  variant   11/20 c246b576  fail  15.7s  Bash(command='python3 -c "import csv, json; rows = 0; total
… (+37 chars)
  ...

target: the task's check passes, end to end from step 6
  original  0/20   0% [0%-16%]
  variant   19/20  95% [76%-99%]
  Fisher exact p = 3.0e-10 → significant at α=0.05, variant higher

original: what the model did at step 6
   count  rate  95% CI     behaviour
A  9/20   45%   [26%-66%]  Write(content='{"count":4,"total":61.5}', file_path='report.json')
B  3/20   15%   [5%-36%]   Write(content='{"count":4,"total_amount":61.5}', file_path='report.json')
C  1/20   5%    [1%-24%]   Bash(command='awk -F, \'NR > 1 {count++; sum += $3} END {printf
                           "{\\"count\\": %d, \\"tota… (+7 chars)
...

variant: what the model did at step 6
   count  rate  95% CI     behaviour
A  15/20  75%   [53%-89%]  Write(content='{"amount_total":61.5,"row_count":4}',
                           file_path='report.json')
B  2/20   10%   [3%-30%]   ListAgents()
C  1/20   5%    [1%-24%]   Agent(prompt="Read data.csv, calculate the number of line items
                           (excluding header) and the… (+8 chars)
D  1/20   5%    [1%-24%]   Bash(command='python3 -c "import csv, json; data =
                           list(csv.DictReader(open(\'data.csv\'))… (+7 chars)
E  1/20   5%    [1%-24%]   Bash(command='python3 -c "import csv, json; rows = 0; total = 0.0; \nwith
                           open(\'data.csv\\… (+7 chars)

Steps before 6 were answered from the recording in every fork — the same history, generated under
the original wording; from step 6 to the end each fork ran live, with each arm's wording, and the
task's check decided. An edit that would also change earlier steps still needs full runs to confirm.
experiments 437ea38d (original) and 59f099f5 (variant) · 8.3 min
```

Every original fork wrote keys of its own and failed: thirteen with `Write`, six from a
script or with `echo`, and one called `ListAgents`, then did the same. Nineteen variant forks
passed. Fifteen wrote `row_count` and `amount_total` with `Write` at once, and one from a
Python script; two called `ListAgents` first, and one handed the job to a subagent
(`Agent`), each writing the right keys at its next call. The one that failed wrote a script
that read the CSV as tab-separated, which failed, and spent its last call trying to clean the
file.

The fork is where it pays to look at the whole run: at the step, §6 counted only the variant's
`Write`s, and the forks show that the scripts and detours got there too. Each fork saw the
folders its source saw — the sandbox gives every launch the same — so its request at the
step was the recorded one, word for word but for the date: the forks ran after midnight.
Without bubblewrap a fork runs in folders of its own, whose names are in every request it
sends; see [Will I get the same numbers?](#will-i-get-the-same-numbers).

A fork whose harness asks for something the recording never answered — its history has
changed — is left out of the rates, and says where it left the recording; none did here.
Each fork is a run of its own, kept out of `tuningfork runs` and of the task's pass rates:
`tuningfork show bdb4e605` lists its steps, the first six marked as answered from the
recording.

### 8. Confirm with full runs

A fork still starts from history the original wording produced — here, six calls that
checked for the file, loaded the skill twice, looked for a report and read the data. Full
runs put the variant in from the very first step. `--skill` installs a file as the skill's
`SKILL.md` in every run:

```bash
tuningfork batch examples/tally-report.evals.json -n 8 --harness claude-code --model gemma-4-26b-a4b \
  --skill examples/variants/tally-report/SKILL.v2.md
```

```
tally-report/1 — 8 runs
  prompt: Use the tally-report skill to summarize data.csv into report.json.
  check:  python3 -c "import json,sys; d=json.load(open('report.json')); sys.exit(0 if
d.get('row_count')==4 and abs(float(d.get('amount_total',0))-61.5)<0.01 else 1)"
  skill:  examples/tally-report  SKILL.md replaced by examples/variants/tally-report/SKILL.v2.md
  sandbox: every run works in /home/user/project, its home /home/user

  run 0  pass  36.2s
  run 1  fail  8.1s
  run 2  pass  10.3s
  run 3  pass  11.4s
  run 4  fail  7.1s
  run 5  fail  30.8s
  run 6  fail  38.1s
  run 7  pass  27.8s

4/8  50%  [22%-78%] passed
...
```

Four out of eight, against none out of eight before. Every report these runs wrote used
`row_count` and `amount_total`: the variant fixed the keys. Three of the four failures are
the other way to fail, which starts before the write. Two answered that the report was done
right after loading the skill — one after looking for it — and wrote nothing. One looked for
the report three times, read the data, then went looking in the skill's own folder, until it
reached the limit of eight calls. The fourth, having looked for the report, wrote a total of
41.5: the first three amounts' sum, and the variant's own example. Nothing in the skill says
that loading it only shows its instructions.

### 9. Catch the failure live

A breakpoint stops a call as it happens. Rather than start a new run and wait for it to reach
the write, restart this one from it: `tuningfork fork` replays a run up to a step, which takes
a second or two since the model is not asked, then hands it to the model with breakpoints
armed. With the original skill, stop Claude Code's `Write` of the report before it acts on it:

```bash
tuningfork break add --tool Write --path '*.json'
tuningfork fork $RUN Write
```

```
breakpoint b22b927c — after the model: Write on *.json
run eca15f11 from step 6 — 1 fork through http://127.0.0.1:8787
  context   after the result of Bash
  replays   6 recorded call(s), answered from the recording; live from step 6
  harness   claude-code, model gemma-4-26b-a4b
  sandbox   every fork works in /home/user/project, its home /home/user, as its source did
```

The fork now waits for the model, and is held when it writes the report. In the web UI —
opened from the address `tuningfork serve` printed, token and all — go to **Live**, change
the keys in the held call's `content` to `row_count` and `amount_total`, and click *Continue
with edits*. Or, from a **fourth terminal** set up like the third, do what Part 1 did:

```bash
tuningfork hold wait
```

```
hold c933e402 — after the model, run 92ceed23 step 6, held 0.5s
  reason  after the model: Write on *.json
  … 20 earlier message(s)
  assistant: tool_calls(Bash)
  tool: total 4 -rw-r--r-- 1 you you 77 Oct  5 23:59 data.csv
  system: <total_tokens>14984554 tokens left</total_tokens>

the model answered
  reasoning_content: The `ls -l` output shows only `data.csv`. `report.json` is definitely not
there.
The `tally-report` skill was called, but `report.json` was not created.
The output of the `Skill` tool was the skill definition itself.
This is very strange. If a tool is called, it should return its result.
If the resu… (+1277 chars)
  tool_call Write({"content": "{\n  \"count\": 4,\n  \"total_amount\": 61.50\n}\n", "file_path":
"report.json"})

tuningfork hold continue c933e402 · step · drop — to edit the response: hold show c933e402 --json >
f.json, edit, hold continue c933e402 --edited f.json
```

```bash
HOLD=$(tuningfork hold list -q)
tuningfork hold show $HOLD --json > held.json
python - <<'EOF'
import json
held = json.load(open("held.json"))
call = held["choices"][0]["message"]["tool_calls"][0]["function"]
arguments = json.loads(call["arguments"])
arguments["content"] = '{"row_count": 4, "amount_total": 61.5}'
call["arguments"] = json.dumps(arguments)
json.dump(held, open("held.json", "w"))
EOF
tuningfork hold continue $HOLD --edited held.json
```

```
c933e402: continue with your edit
```

Claude Code writes your version, and the fork passes — reported as not counted, because the
edit, not the skill, made it pass. Back in the third terminal:

```
  92ceed23  pass  11.2s  Write(content='{"count":4,"total_amount":61.5}', file_path='… (+13 chars)
edited at a breakpoint (step 6) — not counted

`tuningfork show 92ceed23` for the fork's steps; its first 6 were answered from the recording.
```

The line shows what the model did at the write, before your edit. `tuningfork show 92ceed23`
marks steps 0 to 5 as replayed and step 6 as edited, and `tuningfork files 92ceed23 end`
finds the report you wrote. Remove the breakpoint with `tuningfork break list` and
`tuningfork break rm <id>`. Then compare the two wordings, each with its own rate — never
one pooled across them:

```bash
tuningfork tasks
```

```
tally-report/1 · claude-code · gemma-4-26b-a4b
 skill     runs  check passed
 c417ecd4  8     0/8  0%  [0%-32%]
 6bae9525  8     4/8  50%  [22%-78%]  SKILL.md from SKILL.v2.md

skill: a fingerprint of the skill files the runs were given. Intervals are Wilson 95%.
```

Forks are not in it: each is a sample of an experiment, conditional on the history it
replayed, not a run of the task.

### 10. Look at all of it in the browser

The proxy serves the web UI at the address it printed, with its token:

- **Runs** — the task split into its two conditions, one per wording, each with its pass
  rate; *Where do they diverge?* on each compares runs of that wording only.
- **A run** — its conversation top to bottom, each step's answer where it happened,
  reasoning included, and the skill marked inside it and mapped to `SKILL.md` line by
  line. Click the `Write` step for its details and the workspace as the call found it;
  *Exact context* shows everything it was sent.
- **Experiments** — the resample, the A/B, and the A/B run to the end, each fork linked.
  A fork's steps say which were answered from the recording and where it went live; the
  step you edited is flagged, and can be shown as the harness saw it.
- **Live** — breakpoints, and any call being held.
- **Jobs**, **Scenarios** and **Connections** — the next section.

### 11. The same from the browser

Everything this part did from the terminal can be started from the web UI, as jobs the
proxy runs. Open the address `tuningfork serve` printed — the one with `?token=`:

1. **Scenarios** → *Import a spec*: the absolute path of `examples/tally-report.evals.json`,
   named `tally-report`. The editor shows the spec exactly as it will be saved, keeps the
   skill folder where it is, and flags the check for what it is: a shell command that runs,
   as you, in every run's workspace.
2. *Save and run…* opens **Run a task** with the scenario chosen. Choose the harness and the
   number of runs, and read what will run — the check shown whole, the model the server
   lists — before *Start the runs*. The job's page follows each run and links it. (A
   check can judge the agent's final reply instead: *The reply contains…* in the editor,
   with no shell command. Either kind can be tried first with *Test this check…*: on a
   recorded run of the task, beside the verdict that run got, or on a reply and files you
   write. Nothing is recorded.)
3. Open a run, and on its conversation click the `Write` step. **From this step** →
   *Resample…* opens the form right there — so does **Jobs** → *Start…*, from a list of
   runs. The plan says what will be sent, through the proxy, and how long it may take. The
   finished job links the experiment.
4. *A/B a wording…* opens an editor holding the skill as that run was given it. Make the
   change `SKILL.v2.md` makes — or paste the file — name it, and say what the A/B is about:
   *What the model does matches a pattern*, `row_count`. Nothing is chosen for you. The
   plan shows where the skill sits in the context, and any part of your edit the model
   never sees at this step. The variant is saved in the recordings folder when the A/B
   starts, never beside the skill.
5. *Reach: each to the end* turns either form into forks followed to the task's check.
   *Fork from here…* stops a fork at the step for you; on the **Live** page, *Evidence at
   this call* samples the held call where it stands, and *Hypotheses, not evidence* asks
   the model about it — its answer is its own account, never a result.

**Jobs** lists all of it, as `tuningfork jobs list` does, and **Connections** shows the
server, what was measured of it, and the harnesses. Opened without the token, every page
shows everything and changes nothing.

---

## Part 3 — inside a step

Part 2 found the step that fails and the wording that fixes it. This part looks inside a
step, at the model's own tokens: how sure it was of each one, and — where runs split —
which token splits them. It uses Part 2's proxy and recordings, and needs what Part 2 does:
llama.cpp's server, and a model whose raw output tuningfork reads — Gemma 4. §2 adds eight
runs with Hermes Agent.

### 1. How sure was the model of each token?

Take the run from Part 2 §4 again: `$RUN` still holds it in the third terminal. (In a new
terminal, set it to that run's id, as `tuningfork show` printed it there — `tuningfork runs
--called Write` would now find one of §8's runs, which were given the reworded skill.) Score
the answer its `Write` step gave:

```bash
tuningfork tokens $RUN Write --score
```

````
scoring step 6's answer (714 tokens) on http://127.0.0.1:8081…
run eca15f11 step 6  stream f9ba7314
  scored · grammar · top 8 · 714 tokens · rebuilt exactly

reasoning
  The `ls -l` command shows that `report.json` definitely does not exist.
  The `tally-report` skill call returned the skill's instructions.
  This might mean that when I call `Skill`, it doesn't "run" the skill as a program, but rather it's
a way to *invoke* a pre-defined procedure.
  ...
  Now I need to "Write the results to `report.json`".
  The instructions say: "Keep the report small — just the count and the total."
  So `report.json` should look something like:
  ```json
  {
    "count": 4,
    "total": 61.5
  }
  ```

  I will use the `Write` tool to create `report.json`.

tool call
  call:Write{content:<|"|>{
    "count": 4,
    "total": 61.5
  }<|"|>,file_path:<|"|>report.json<|"|>}

where the step's sampler could most easily have drawn another token
  [136] reasoning ' perform' 5% | ' *' 93%  H 0.48 bits  runner-up drawn 95%
  [585] reasoning ' "' 6% | ' write' 93%  H 0.41 bits  runner-up drawn 94%
  [167] reasoning "'" 5% | ' am' 87% | ' expect' 8%  H 0.71 bits  runner-up drawn 87%
  ...

Entropy counts the probability beyond the top 8 as one outcome: a lower bound. The runner-up's
chance is under the step's own sampler settings.
````

Scoring sends the step's exact context once more with the answer forced, token by token, and
reads the model's own distribution at each one: one call, about as long as the answer took.
*Rebuilt exactly* says these are the run's own tokens — the answer, written back as the raw
text the model generated, reasoning and tool-call syntax and all, has as many tokens as the
run reported. (That raw text is how Gemma 4 writes a call: `<|"|>` fences each string.) In a
terminal, the text is shaded by entropy.

The model reasoned for most of its 714 tokens. The places it could most easily have gone
another way are all in the reasoning, and none touches the keys: those were settled where it
quoted the skill — "Keep the report small — just the count and the total" — and sketched the
report from it. `tuningfork tokens $RUN Write --json` puts the sketch's `count` and `total` at
100% each, and `row` is not among the eight tokens the model weighed for the first key; the
call copies them. No amount of resampling this step would find `row_count` — Part 2 §5 drew
twenty and got none — so the wording has to change, as Part 2 §6 found.

### 2. Which token decided it?

Where runs of a step do split, `forking-tokens` finds the token that splits them (after
[Bigelow et al., ICLR 2025](https://arxiv.org/abs/2412.07961)). At the tokens where the
step's own sampler could most easily have drawn another, it forces each likely alternative,
lets the model go on, and tests which change what it does.

Part 2's runs split at their first step, between `ls data.csv` and `ls`, which led to the same
place. Hermes Agent's runs of this task split on which tool they use first, so this section
records eight of those (installing Hermes is under
[Using Hermes Agent instead](#using-hermes-agent-instead)):

```bash
tuningfork batch examples/tally-report.evals.json -n 8 --harness hermes --model gemma-4-26b-a4b
```

```
tally-report/1 — 8 runs
  ...
0/8  0%  [0%-32%] passed

8 runs of tally-report/1
  agreed for 0 step(s), diverged at step 0

step 0
   count  rate  95% CI     behaviour                                         then passed
A  7/8    88%   [53%-98%]  terminal(command='ls data.csv')                   0/7  0%  [0%-35%]
B  1/8    12%   [2%-47%]   search_files(pattern='data.csv', target='files')  0/1  0%  [0%-79%]
```

Seven runs listed `data.csv` with `ls`, and one searched for it. All eight then wrote the
report from a Python script, with `count` and `total_amount` or `total`. Take the latest run
that searched *at step 0* — `--at-step` matters: a run can do something else first and search
later — and probe that step:

```bash
HRUN=$(tuningfork runs -q -n 1 --called search_files --at-step 0)
tuningfork forking-tokens $HRUN 0 --score
```

```
scoring step 0's answer first…
no token in the answer had a runner-up the step's sampler would draw at least 5% of the time: the
model had decided before it — while reasoning, most likely — probe the reasoning instead (`--in
reasoning`), or pick the positions yourself (`--at`)
the best in the reasoning: [128] ' directly' or '.' (~56 tokens to regenerate); [148] ' search' or '
check' / ' list' (~36 tokens to regenerate); [125] ' fulfill' or ' proceed' / ' perform' (~59 tokens
to regenerate); [143] 'First' or 'Wait' / 'Let' / 'Since' (~41 tokens to regenerate); [140] ' task'
or ' logic' / ' steps' (~44 tokens to regenerate)
```

`--score` scored the step's answer first, as `tokens --score` does. The probe then looked in
the answer — here a tool call and nothing else — for tokens the step's own sampler could
easily have drawn otherwise, and found none: by the time the model wrote the call, the choice
was made. So it says where to look instead, the likeliest places in the reasoning, each with
the length of reasoning a forced token there would regenerate. `--in reasoning` probes them:

```bash
tuningfork forking-tokens $HRUN 0 --in reasoning
```

```
forking tokens at run 1d46136f step 0 — after user: Use the tally-report skill to summarize data.csv
into report.json.
  base      stream 489f70d1 (scored, 184 tokens)
  target    the base's own behaviour: search_files(pattern='data.csv', target='files') (declared
before sampling)
  arms      15 forced tokens × 10 = 150 continuations
  power     a forced token is found to change the outcome when about 7 of its 10 continuations
change; smaller shifts need more (-n)
  estimate  ~6,930 tokens to generate, ~4.7 min at the step's own speed
            counting the whole prompt, 13,729 tokens, processed first for each continuation (~1.4s)
  reasoning positions regenerate the rest of the reasoning: each continuation is long
  [125] reasoning ' fulfill' 9% → ' proceed' 65% | ' perform' 22%
  [128] reasoning ' directly' 10% → '.' 85%
  [140] reasoning ' task' 21% → ' logic' 55% | ' steps' 13%
  [143] reasoning 'First' 6% → 'Wait' 56% | 'Let' 21% | 'Since' 5%
  [148] reasoning ' search' 12% → ' check' 82% | ' list' 5%

  125:' proceed'  1/10  search_files(pattern='data.csv', target='files')  3.3s
  ...
  148:' search' 10/10  search_files(pattern='data.csv', target='files')  17.7s

target (declared before sampling): the base's own behaviour: search_files(pattern='data.csv',
target='files')
  the baseline arms reached it 22/50  44% [31%-58%]
  at [140] and [143] the step was still undecided: from there the run's own token led elsewhere more
often than to its behaviour, so a later token decided it (`tuningfork tokens` shows the tokens
after, and --at probes them)
...
[140] reasoning  ….
If it does exist, I should execute the⟦ task⟧
  as run  ' task' 21%              0/10   0% [0%-28%]
  forced  ' logic' 55%             5/10  50% [24%-76%]   p=0.033 → Holm 0.228
  forced  ' steps' 13%             6/10  60% [31%-83%]   p=0.011 → Holm 0.087
  the chance of the target from here, weighted by each token's chance: 40%
...
[148] reasoning  … I should execute the task.

First, I'll⟦ search⟧
  as run  ' search' 12%            10/10 100% [72%-100%]
  forced  ' check' 82%             3/10  30% [11%-60%]   p=0.003 → Holm 0.028  flips away from the
target
  forced  ' list' 5%               2/10  20% [6%-51%]   p=7.1e-04 → Holm 0.007  flips away from the
target
  the chance of the target from here, weighted by each token's chance: 38%

behaviours, across the probe
   count   rate  95% CI     behaviour
A  85/150  57%   [49%-64%]  terminal(command='ls data.csv')
B  62/150  41%   [34%-49%]  search_files(pattern='data.csv', target='files')
C  3/150   2%    [1%-6%]    terminal(command='ls')

2 of 10 forced tokens changed what the model did (Holm-corrected over the 10 planned before
sampling; α=0.05).

experiment 6f31093c · 6.2 min · `tuningfork experiment 6f31093c` shows it again
```

One word decided it. At [148] — "First, I'll ⟦search⟧" — the run's own ` search`, a token the
sampler draws 12% of the time, led to `search_files` ten times in ten, forced again as the
baseline every alternative is compared with. Forced to ` check`, the word the model was
likeliest to write there, it listed the file with `ls` seven times in ten; to ` list`, eight.
Fisher's exact test puts each at p < 0.01, and both hold after the Holm correction over all
ten tests planned before sampling. The target is fixed before sampling too: by default, the
base's own behaviour.

Before [148] nothing was settled. At [125] and [128] the alternatives searched about as often
as the run's own word, and at [140] and [143] the run's own word led to `ls` more often than
not — the probe says so, and that a later token decided. At [140], ` task` searched none of
ten times against five and six for ` logic` and ` steps`: p = 0.03 and 0.01 alone, neither
after the correction. Ten continuations an arm can detect only large shifts, as the power
line said.

The estimate counted each continuation's prompt, 13,729 tokens at about 1.4 s apiece — most
of its 4.7 minutes; the probe took 6.2, since a continuation of reasoning can run longer than
the base's. `tuningfork tokens $HRUN 0` shows the whole answer:

```
run 1d46136f step 0  stream 489f70d1
  scored · grammar · top 8 · 184 tokens · rebuilt exactly

reasoning
  The user wants to use the `tally-report` skill to summarize `data.csv` into `report.json`.
  The `tally-report` skill is already preloaded in this session.
  The skill documentation says:
  1. Read the CSV file named in the request.
  2. Work out how many line items it contains, and the total of the amount column.
  3. Write the results to `report.json` in the working directory.

  I need to check if `data.csv` exists first. If it doesn't, I can't fulfill the request directly.
  If it does exist, I should execute the task.

  First, I'll search for `data.csv` in the current directory.
tool call
  call:search_files{pattern:<|"|>data.csv<|"|>,target:<|"|>files<|"|>}

where the step's sampler could most easily have drawn another token
  [128] reasoning ' directly' 10% | '.' 85%  H 0.82 bits  runner-up drawn 89%
  [148] reasoning ' search' 12% | ' check' 82% | ' list' 5%  H 0.86 bits  runner-up drawn 82%
  [125] reasoning ' fulfill' 9% | ' proceed' 65% | ' perform' 22%  H 1.46 bits  runner-up drawn 68%
  [143] reasoning 'First' 6% | 'Wait' 56% | 'Let' 21% | 'Since' 5%  H 2.01 bits  runner-up drawn 61%
  [140] reasoning ' task' 21% | ' logic' 55% | ' steps' 13% | ' skill' 4%  H 1.88 bits  runner-up
drawn 56%
  [142] reasoning '\n\n' 17% | ' Since' 52%  H 1.71 bits  runner-up drawn 55%
  ...
```

The reasoning decides early *that* the file should be checked — "I need to check if
`data.csv` exists first" — and *how* only in its last sentence, by one word: where the model
would most likely have written "check", this run wrote "search", and the call followed.

A probe takes the five likeliest places by default; `--at` probes any position by hand. Take
[142], sixth on the list — the line break before "First":

```bash
tuningfork forking-tokens $HRUN 0 --at 142
```

```
forking tokens at run 1d46136f step 0 — after user: Use the tally-report skill to summarize data.csv
into report.json.
  base      stream 489f70d1 (scored, 184 tokens)
  target    the base's own behaviour: search_files(pattern='data.csv', target='files') (declared
before sampling)
  arms      4 forced tokens × 10 = 40 continuations
  power     a forced token is found to change the outcome when about 6 of its 10 continuations
change; smaller shifts need more (-n)
  estimate  ~1,680 tokens to generate, ~1.2 min at the step's own speed
            counting the whole prompt, 13,729 tokens, processed first for each continuation (~1.4s)
  reasoning positions regenerate the rest of the reasoning: each continuation is long
  [142] reasoning '\n\n' 17% → ' Since' 52% | ' I' 1% | ' The' 1%

  142:' Since'  1/10  search_files(pattern='data.csv', target='files')  2.5s
  ...
  142:'\n\n' 10/10  terminal(command='ls data.csv')  27.0s

target (declared before sampling): the base's own behaviour: search_files(pattern='data.csv',
target='files')
  the baseline arms reached it 2/10  20% [6%-51%]
  at [142] the step was still undecided: from there the run's own token led elsewhere more often
than to its behaviour, so a later token decided it (`tuningfork tokens` shows the tokens after, and
--at probes them)

[142] reasoning  …t does exist, I should execute the task.⟦

⟧
  as run  '\n\n' 17%               2/10  20% [6%-51%]
  forced  ' Since' 52%             5/10  50% [24%-76%]   p=0.350 → Holm 1.000
  forced  ' I' 1%                  3/10  30% [11%-60%]   p=1.000 → Holm 1.000
  forced  ' The' 1%                4/10  40% [17%-69%]   p=0.628 → Holm 1.000
  the chance of the target from here, weighted by each token's chance: 42%

behaviours, across the probe
   count  rate  95% CI     behaviour
A  26/40  65%   [50%-78%]  terminal(command='ls data.csv')
B  14/40  35%   [22%-50%]  search_files(pattern='data.csv', target='files')

0 of 3 forced tokens changed what the model did (Holm-corrected over the 3 planned before sampling;
α=0.05).

experiment 45761986 · 1.9 min · `tuningfork experiment 45761986` shows it again
```

Six tokens before the decision, the step was still open: from the run's own line break, two
continuations in ten searched, and no alternative did significantly better. (At a position
you pick, the likeliest alternatives are forced even when the sampler would seldom draw one:
` I` and ` The` had 1% each.) The probe finds the token that decides a step when there is
one, and says where the step was still undecided. Here the choice made no difference to the
outcome — every Hermes run failed at the write — but it shows how far a run's path can turn on
one unlikely word.

### 3. The same in the browser

On a run's conversation, each step's answer has **Answer | Tokens**. *Tokens* shows the
heatmap — by entropy, or by how close the top two were — with the reasoning folded, and how
many of the likeliest forks are in it. An answer not scored yet offers *Score this answer*.
Click a token for what else the model weighed there; *Select to probe*, then *Find forking
tokens at these…*, opens the probe's form in the step, those positions filled in. The form
shows its plan before anything runs: the tokens it will force, what counts as the same
behaviour, what it can detect with that many continuations, and how long it may take.

What counts as the same behaviour can be the answer's own, a regular expression, or a
behaviour you describe: *Calls the tool* `search_files`, *Mentions* `data.csv`, every
condition holding. Whichever you choose, the form sets it against every behaviour already
seen at the step — the run's own, other runs of the task there, the samples drawn there —
and marks which would count, before anything is sampled.

The result is on the experiment's page: the answer it forked from, with the probed tokens
ringed — solid where a forced token changed what the model did — then each position, the
run's own token and each forced one, with where it flips. *In the answer* opens a position's
token there; *On the step* opens it on the step's own Tokens view, where every token a probe
forced is ringed and says what each probe found.

---

## Part 4 — let tuningfork propose the fix

Part 2 tested a fix written in advance. Here tuningfork gathers fixes itself — from a model
shown the evidence, from you, or from an agent — and screens each at the failing step
against the wording the run was given, validates a survivor with full runs, and writes it
into the skill once you approve it. It uses Part 2's model server and recordings: §1–4 run
no harness, §5 runs thirty. About twenty minutes.

### 1. Say what the step should do

A *refinement* fixes one step and what the model should do there, before anything is
proposed — as an A/B declares its target before it samples. Take the run from Part 2 §4
again — `$RUN`, as in Part 3 §1:

```bash
tuningfork refine start $RUN Write --target-rules '[{"if": "calls", "value": "Write"},
  {"if": "contains", "value": "row_count"}, {"if": "contains", "value": "amount_total"}]'
```

```
refinement 68a27e5f — open
  checkpoint  run eca15f11 step 6 (task tally-report/1)
  skill       tally-report, as the run was given it:
              /home/you/tuningfork-guide/batches/001cec9a/skill-snapshot/tally-report/SKILL.md
  its file    examples/tally-report/SKILL.md
  target      a behaviour that calls Write, and mentions "row_count", and mentions "amount_total"
              declared before anything was proposed
  started by  a person

next: ask for proposals — or give your own wording — then screen them:
  tuningfork refine propose 68a27e5f --builtin
  tuningfork refine propose 68a27e5f --file SKILL.md --why '…'
  tuningfork refine screen 68a27e5f
```

It keeps its own copy of the skill as the run was given it, and every proposal edits that
copy: nothing here writes to `examples/tally-report`.

### 2. What a proposer is shown

```bash
REF=$(tuningfork refine list -q -n 1)
tuningfork refine show $REF --packet
```

```
{
  "sha": "3c256c1a770ad343",
  "target": {
    "description": "a behaviour that calls Write, and mentions \"row_count\", and mentions \"amount_total\""
  },
  "skill": {
    "name": "tally-report",
...
      "14: 3. Write the results to `report.json` in the working directory.",
      "15: ",
      "16: Keep the report small — just the count and the total."
    ],
    "in_context": [
      {
        "lines": [
          6,
          16
        ],
        "form": "body",
        "role": "user"
      }
    ]
  },
...
  "answers_here": {
    "summary": "none of the 21 answers seen at this checkpoint reached the target",
...
  "runs_here": {
    "summary": "8 runs of this task with this harness, model and skill version",
...
  "task": {
    "prompt": "Use the tally-report skill to summarize data.csv into report.json.",
    "expected_output": "A report.json containing the line-item count and the amount total.",
    "expectations": [
      "report.json exists in the working directory",
      "it records 4 line items and a total of 61.50 under the keys row_count and amount_total"
    ],
...
```

The packet is built from what is recorded, and nothing is sampled for it:

- the step as the model saw it, and where the skill sits in it: lines 6 to 16, the body
  Claude Code loaded;
- the answers seen at this step that did, and did not, do what the target says: here the
  run's own and the twenty from Part 2 §5's resample, none of which did;
- how this task's runs with the same harness, model and skill version went there;
- the tokens the model was least sure of, once the answer has been scored (Part 3 §1);
- the task's own criteria;
- what was tried before.

Only this task's records are read.

### 3. Ask for proposals

The model asked is the one experiments sample from — llama-server. It reasons before it
answers, and on a bigger skill than this one it can need more than the 8192 tokens it is
given by default; give it room once, and the setting stays in the recordings folder:

```bash
tuningfork connections proposer set --max-tokens 32768
tuningfork refine propose $REF --builtin
```

```
proposer  the proxy's own server — through the proxy for a job, gated and never recorded
  wire         openai-chat
  model        the server default
  max tokens   32768
  temperature  0.8
  key          not set
the built-in proposer — 3 proposals
  sends     the packet — 13 KB of recorded prompts, answers and the skill — to 127.0.0.1:8081
  model     the server default (openai-chat, max_tokens 32768)
  packet    3c256c1a770ad343
  asking…
  answered in 37s, 5019 tokens · kept in proposer/20261006T043545-a0cffb
  proposal c50a9794: 1 line changed
    why  The agent uses the phrasing 'count and the total' from this line to name its JSON keys, but
the task requires 'row_count' and 'amount_total'. Explicitly stating the required keys will guide
the agent to use the correct names.
  proposal 729ec26f: 1 line changed
    why  Specifying the required JSON keys in the writing step directly addresses the failure where
the agent used 'count' and 'total' instead of the required 'row_count' and 'amount_total'.
  proposal 3 not kept: say why: a rationale that cites the evidence

hypotheses until screened: `tuningfork refine screen 68a27e5f --proposals c50a9794,729ec26f`
```

The proposer is sent the skill and the packet as data, with rules: at most three edits, each
`old` copied exactly from the skill, at most ten lines changed, the frontmatter left alone,
nothing copied from the task's own data, and a reason that cites the evidence. A proposal
that breaks a rule is not kept, and the reason is printed. Here the model misspelled the
field that holds the third proposal's reason — `rationals` — so as far as tuningfork could
tell, it gave none. Each one kept is saved with the model, the packet and the prompt it came
from; the prompt and the answer are in the refinement's folder, under `refinements/`.

`tuningfork connections proposer set` points the proposer at another endpoint — a cloud
API only with `--allow-remote`, its key read from `--key-stdin` and kept in a file only you
can read. To add your own wording instead, edit a copy of the skill and give it with a
reason: `tuningfork refine propose $REF --file my-SKILL.md --why "..."`.

### 4. Screen them

```bash
tuningfork refine screen $REF -n 20
```

```
run eca15f11 step 6 — 20 samples per arm from http://127.0.0.1:8081
  context   23 messages, after the result of Bash
  sampling  the recorded conditions (the request's own parameters, else server defaults)
  budget    max_tokens 4096, concurrency 1
  estimate  the recorded step took 8.2s; at most ~8.2 min
  target    a behaviour that calls Write, and mentions "row_count", and mentions "amount_total"
            declared when the refinement began
  arm       c50a9794: 1 line changed
  arm       729ec26f: 1 line changed
  survives  with at least 6 of 20 — the step did it 0 times in 20 in resample 98e65286
            Holm over 2 proposals, α=0.05
  samples   60 in all: the original first, then each arm

  original   1/20  Bash(command='python3 -c "\nimport csv\nimport json\n\ninput_fil… (+33 chars)
13.4s
...
target (declared before sampling): a behaviour that calls Write, and mentions "row_count", and
mentions "amount_total"
          target                p        Holm
original  0/20   0% [0%-16%]
c50a9794  16/20  80% [58%-92%]  1.5e-07  3.1e-07  survives
729ec26f  14/20  70% [48%-85%]  3.3e-06  3.3e-06  survives
p: Fisher's exact test of each proposal against the original arm; Holm: corrected over the 2 planned
before sampling (α=0.05).
A screen at one step is evidence, not proof: confirm a survivor with full runs.
screen 61992505 · 8.7 min · `tuningfork refine show 68a27e5f` shows every proposal with its verdicts
```

A screen samples the step with the original wording first, fresh, then with each proposal
swapped in, and compares each with the original — Fisher's exact test, corrected (Holm)
for the number of proposals named before the first sample. Before it samples, it says what
an arm would need to survive — here using Part 2 §5's resample, which saw the target 0 times
in 20 — and it refuses a screen in which nothing could. Its samples are drawn as Part 2's
were, each prompt processed whole.

```bash
tuningfork refine show $REF
```

```
refinement 68a27e5f — open
  checkpoint  run eca15f11 step 6 (task tally-report/1)
  skill       tally-report, as the run was given it:
              /home/you/tuningfork-guide/batches/001cec9a/skill-snapshot/tally-report/SKILL.md
  its file    examples/tally-report/SKILL.md
  target      a behaviour that calls Write, and mentions "row_count", and mentions "amount_total"
              declared before anything was proposed
  started by  a person

proposals (2) — hypotheses until a screen tests them
  c50a9794  1 line changed · the built-in proposer, gemma-4-26b-a4b at 127.0.0.1:8081 · packet 3c256c1a770ad343
    why  The agent uses the phrasing 'count and the total' from this line to name its JSON keys, but
the task requires 'row_count' and 'amount_total'. Explicitly stating the required keys will guide
the agent to use the correct names.
    @@ line 14 @@
      3. Write the results to `report.json` in the working directory.
      
    - Keep the report small — just the count and the total.
    + Keep the report small — use 'row_count' for the count and 'amount_total' for the total.
    screened  survives — 16/20 against the original's 0/20, Holm p 3.1e-07 (screen 61992505)
  729ec26f  1 line changed · the built-in proposer, gemma-4-26b-a4b at 127.0.0.1:8081 · packet 3c256c1a770ad343
    why  Specifying the required JSON keys in the writing step directly addresses the failure where
the agent used 'count' and 'total' instead of the required 'row_count' and 'amount_total'.
    @@ line 12 @@
      1. Read the CSV file named in the request.
      2. Work out how many line items it contains, and the total of the amount column.
    - 3. Write the results to `report.json` in the working directory.
    + 3. Write the results to `report.json` in the working directory using 'row_count' and 'amount_total' keys.
      
      Keep the report small — just the count and the total.
    screened  survives — 14/20 against the original's 0/20, Holm p 3.3e-06 (screen 61992505)

screens (1)
  61992505  done  2 proposals, n=20 an arm — 729ec26f, c50a9794 survived
```

Both survive. `c50a9794` names the keys in the skill's last line, in place of "the count and
the total" — the very words of the model's keys. `729ec26f` names them in the step that
writes the file. Of the ten samples that missed, four wrote the right keys from a script,
which a target asking for `Write` does not count; four called `ListAgents`; one loaded the
skill again, and one printed the numbers first.

### 5. Validate a survivor with full runs

A screen at one step inherits the history the original wording wrote, and a wording can reach
the target without fixing the task. Validation checks a survivor with full runs, each arm
against a fresh arm of the original wording:

- **forward:** forks of the run from the step to the end — does reaching the target carry the
  task?
- **training:** full runs of the task, the edit in place from the first step — does it upset
  an earlier step?
- **held out:** full runs of each task the spec holds out. `examples/tally-report.evals.json`
  has one, eval 2: the same skill on `sales.csv`, marked `"held_out": true`. A batch never
  runs it unless asked by name, and validation records its runs in a temporary folder outside
  `~/tuningfork-guide`, deleted when they end: only whether each passed is kept, and no agent
  reading the recordings is shown even that.

Its runs go through the proxy from Part 2 §2, so leave that running. Take a survivor's id
from `refine show` and look at the plan first:

```bash
PROPOSAL=c50a9794
tuningfork refine validate $REF $PROPOSAL --dry-run
```

```
validate proposal c50a9794 of refinement 68a27e5f — tally-report/1, at step 6
  screened  survived screen 61992505: 16/20 against the original's 0/20, Holm p 3.1e-07
  forward   5 forks an arm from step 6 to the end, the original first
  training  5 full runs an arm of tally-report/1, the edit from the first step
  held out  5 full runs an arm of each of: 2 — only their verdicts are kept,
            recorded in a temporary folder under /tmp, outside the recordings folder,
            and deleted when they end
  runs      30 in all, through http://127.0.0.1:8787
  at a few runs an arm only a large difference shows: the not-worse checks are low-powered
```

Five forks or runs an arm, thirty in all — about eleven minutes:

```bash
tuningfork refine validate $REF $PROPOSAL
```

```
...
forward — forks from step 6 to the end
  original   1/5 b65a6bdc  fail  10.4s  Bash(command='awk -F\',\' \'NR > 1 {count++; sum += $3} END
… (+37 chars)
...
  edit       5/5 b1e5de7f  pass  10.4s  Write(content='{"amount_total":61.5,"row_count":4}',
file_pa… (+17 chars)

training — full runs of tally-report/1
  original   1/5 3b665b6a  fail  24.8s
...
  edit       4/5 392c7311  pass  22.7s
  edit       5/5 76d84f17  fail  6.5s

held out 2 — verdicts only
  original   1/5  ended
...
  edit       5/5  ended

  dcd4934d  done  proposal c50a9794 — meets the gate
    forward   edit 5/5 (100%, 95% CI 57–100%); original 0/5 (0%, 95% CI 0–43%)
    training  edit 3/5 (60%, 95% CI 23–88%); original 0/5 (0%, 95% CI 0–43%)
    held out 2: edit 2/5 (40%, 95% CI 12–77%); original 0/5 (0%, 95% CI 0–43%)
    G0 met     survived a screen: it survived screen 61992505
    G1 met     a held-out task was validated: 1 held-out task(s) validated
    G2 met     finished, with enough runs: every arm ran its runs on one server
    G3 met     its own task succeeds from the target forward: with the edit 5/5 forks passed, without it 0/5
    G4 met     the training task's full runs are not worse (low-powered): not significantly worse (Fisher p = 0.167); at 5 runs an arm only a large drop would show
    G5 met     held-out task 2 is not worse (low-powered): not significantly worse (Fisher p = 0.444); at 5 runs an arm only a large drop would show
    G6 met     the skill's file is still the one refined: it holds the text the refinement started from (line endings aside)
    G7 met     the refinement is open, and nothing in it is approved: open
    G8 met     held-out task 2 improves where the original fails: with the edit 2/5, without it 0/5

it meets the gate: `tuningfork refine approve 68a27e5f c50a9794` shows what approving writes
```

Every check keeps its counts and the name of the rule that judged it. From the step on, the
edit passed every fork, as the screen promised — one after an `ls -a` and a script. From the
first step, three full runs in five passed, every one with the new keys, and the two that
failed show this model's other way to fail: one answered that the report was done right
after loading the skill, and the other kept looking for the report the skill had not written,
loading it twice more, until it ran out of calls. Neither wrote anything. Three in five
against none in five is not significant at this size, and G4 asks only that the edit be no
worse, saying that at five runs an arm only a large drop would show. Held out, two in five
against none is not significant either; G5 asks the same of it.

G8 asks more of a held-out task the original fails: the edit must pass it more often, as it
did, two to none. An edit that writes in the training task's answers would leave it failing
— and be no worse. Notes
name what no check decides, such as an original that already passes from the step: then the
step may not decide the task. `tuningfork refine show $REF` shows every validation again,
held-out verdicts included; no MCP tool shows those.

### 6. Approve it

Approving is the one thing in tuningfork that writes a skill's own file, and only you do it.
It shows the file it will write — under WSL, also as Windows names it — what it keeps, where
the old file and the changelog go, the gate and the edit, and writes nothing until you type
the code it shows:

```bash
tuningfork refine approve $REF $PROPOSAL
```

```
approve proposal c50a9794 of refinement 68a27e5f, on validation dcd4934d
  writes     /home/you/tuningfork/examples/tally-report/SKILL.md
             \\wsl.localhost\Ubuntu\home\you\tuningfork\examples\tally-report\SKILL.md
  keeps      LF line endings
  file before → /home/you/tuningfork/examples/tally-report.history/
  changelog   → /home/you/tuningfork/examples/tally-report.changelog.md
  the gate   every check met:
    G0 met     survived a screen: it survived screen 61992505
    G1 met     a held-out task was validated: 1 held-out task(s) validated
    G2 met     finished, with enough runs: every arm ran its runs on one server
    G3 met     its own task succeeds from the target forward: with the edit 5/5 forks passed, without it 0/5
    G4 met     the training task's full runs are not worse (low-powered): not significantly worse (Fisher p = 0.167); at 5 runs an arm only a large drop would show
    G5 met     held-out task 2 is not worse (low-powered): not significantly worse (Fisher p = 0.444); at 5 runs an arm only a large drop would show
    G6 met     the skill's file is still the one refined: it holds the text the refinement started from (line endings aside)
    G7 met     the refinement is open, and nothing in it is approved: open
    G8 met     held-out task 2 improves where the original fails: with the edit 2/5, without it 0/5
  the edit
    @@ line 14 @@
      3. Write the results to `report.json` in the working directory.
      
    - Keep the report small — just the count and the total.
    + Keep the report small — use 'row_count' for the count and 'amount_total' for the total.

  code  8cfda510
Type the code to write it (anything else writes nothing): 8cfda510
written: /home/you/tuningfork/examples/tally-report/SKILL.md
  the file before  /home/you/tuningfork/examples/tally-report.history/SKILL.20261006T045621Z-c3dd28e9.md
  changelog        /home/you/tuningfork/examples/tally-report.changelog.md
  to undo, copy the file before back over it
```

`examples/tally-report/SKILL.md` now has the edit. The old text is in
`examples/tally-report.history/`, and `examples/tally-report.changelog.md` says what changed,
why, on what evidence and who approved it — beside the skill folder, never in it, since every
run is given a copy of that folder. The refinement keeps the approval with its evidence, so
deleting its runs later leaves it standing. git sees the change:

```bash
git status --short examples/
```

```
 M examples/tally-report/SKILL.md
?? examples/tally-report.changelog.md
?? examples/tally-report.history/
```

Your code will differ: it names the proposal, the file and the file's bytes as they are now.
Run Part 2 §3's batch again to see the new wording's pass rate beside the old one in
`tuningfork tasks`. To put the example back as it was:

```bash
git checkout -- examples/tally-report/SKILL.md
rm -r examples/tally-report.history examples/tally-report.changelog.md
```

### 7. The same in the browser, or from an agent

The **Refine** page lists refinements; a step's **Refine the skill here…** starts one. A
refinement's page shows the evidence and each proposal as a line diff, with who proposed it,
why and its verdicts; asks the proposer, or takes your own wording; screens the proposals you
tick; validates a survivor; and shows each validation's parts and gate, with an **Approve…**
that shows the file, the edit and the code to type. An agent gets the loop up to the screen
over MCP with `tuningfork mcp --tools refine`: it starts a refinement, reads the evidence,
proposes its own edits and screens them. It is never offered the built-in proposer, a
validation or an approval, and never shown a held-out task's verdicts.

## Clean up

Stop the proxy and `llama-server` with Ctrl-C, then:

```bash
rm -rf ~/tuningfork-guide ~/tuningfork-demo ~/tuningfork-live
rm -f held.json answer.json
```

If you approved an edit in Part 4, put the example back with the two commands at the end of
Part 4 §6.

---

## Using Hermes Agent instead

Everything above works with Hermes Agent too. `--harness hermes` is the only change to a
batch command; the rest differs as follows.

| | Claude Code | Hermes Agent |
|---|---|---|
| Install | `npm install -g @anthropic-ai/claude-code@2.1.281` | `uv tool install hermes-agent==0.18.2` |
| `--harness` | `claude-code` | `hermes` |
| Speaks | Anthropic Messages | OpenAI chat-completions |
| The skill arrives | through the `Skill` tool, body only | preloaded into the system prompt, whole file |
| The write step | `Write`, or a script | a script (`execute_code`), whose code a target reads |
| Breakpoint | `--tool Write --path '*.json'` | `--tool execute_code` |

Our run with Hermes, through the same commands on the same stack:

- **Record eight runs:** 0/8, CI [0–32%] — Part 3 §2's batch. Every run wrote the report from
  a Python script (`execute_code`), with `count` and `total_amount` or `total`. Hermes puts the
  skill in the system prompt, so no run took loading it for the job done.
- **Find the decision:** no run called `write_file`, so name the step by the tool that wrote
  the report: `RUN=$(tuningfork runs -q -n 1 --called execute_code)` picks the latest run,
  whose script came at step 2, after `ls` and a read, with no reasoning. `tuningfork files
  $RUN execute_code` found only `data.csv`.
- **Resample** `$RUN execute_code`: no `row_count` in twenty. Nineteen scripts — sixteen with
  `count` and `total_amount`, three with `count` and `total` — and one `write_file` with
  `count` and `total`, in 1.4 minutes. The table lists twenty behaviours: a script's key is a
  digest of its syntax tree, so two scripts are the same behaviour only when their code is.
  Which keys they wrote is for a target to read.
- **A/B** at the write, with a target that reads the scripts — `--target-rules '[{"if":
  "contains", "value": "row_count"}, {"if": "contains", "value": "amount_total"}]'`: 0/20
  against 20/20, p = 1.5 × 10⁻¹¹. tuningfork found the whole file, frontmatter and all, where
  Hermes preloads it: in the system prompt.
- **Follow it to the end:** 0/20 → 20/20, p = 1.5 × 10⁻¹¹, in 6.6 minutes. Each fork replayed
  two calls, the `ls` and the read, and went live at the write.
- **Confirm:** 8/8 with `SKILL.v2.md`, CI [68–100%].
- **Catch it live:** `tuningfork break add --tool execute_code`, then `tuningfork fork $RUN
  execute_code`, held the fork's script. With its `code` replaced by one that writes the right
  keys — Part 2 §9's recipe, editing `code` where that one edits `content` — the fork went on
  to pass, not counted.
- **Propose and screen** at `execute_code`, with the A/B's target: the proposer's answer came in
  25 s, with three edits, each naming the keys in a different line — the step that works the
  values out, the step that writes the file, the closing line. All three survived, 20/20 each
  against the original's 0/20, Holm p = 4.4 × 10⁻¹¹. Hermes preloads the whole file, so the
  packet showed all 16 of its lines in the system prompt.

## Will I get the same numbers?

Not digit for digit. The model samples at temperature 1.0 from a fresh random seed each time
— a harness sends none, and tuningfork draws new ones for every experiment — so counts
wobble between tries and between machines, and so does the run `--called Write` picks: here
one run in eight wrote its report with `Write`, at step 6. What reproduces is the shape: the
original wording fails every time, by writing `count` and `total` or by taking the loaded
skill for the job done; resampling the write never gives `row_count`; the A/B separates the
wordings by a wide, significant margin at the write; to the end of the run the variant
passes nearly every fork; and full runs with it pass more often, held back by the other
failure, which no wording here addresses. Our batches of the original wording on this model
have all come out at zero: 0/8, 0/8, 0/5 and 0/8 with Claude Code, the last in the sandbox,
and 0/8 twice with Hermes. The variant's full runs passed 7/8 before the sandbox and 4/8 in
it. The step-0 split varies the most, and matters least.

Without bubblewrap, each run works in folders of its own, whose names are in its prompt, so
on your machine every run is asked something slightly different — and to this model that
can matter. Before the sandbox, forks of one run that differed only in a folder's number
were resampled at the write with the variant swapped in, every prompt processed whole: three
wrote the file 20 times in 20, and one wrote it 5 times, answering the other 15 in text.

Part 1's numbers are exact, because its "model" is a script.

## Notes

**This skill was hard to make fail on purpose.** An earlier version left a *judgement call*
ambiguous instead of a key name — the data had a refund, and the skill said only "ignore
anything that isn't a genuine sale". The expectation was a roughly even split. The model
resolved it the same way six times out of six. A skill's pass rate is a property of the model
as much as of the wording, and a genuine coin-flip is harder to engineer than it sounds.

**Two corrections.** This page once reported the shipped skill at 1/6. The one "pass" came
from a run recorded under a different `SKILL.md` — the file was edited while its batch was
still running — and the skill hashes recorded with every run exposed it; a batch now freezes
the skill before its first run. Separately, `SKILL.v2.md` itself sat inside the skill's folder
from the start, so every run was given a copy of it beside the skill under test. No recorded
call ever mentioned it, so no result here changes, but it has moved out, and `batch` now
warns about any other `SKILL*.md` in a skill's folder.

**Why Gemma 4 26B.** This page first ran on Gemma 4 31B, without `--swa-full`. With the flag
the 31B no longer fits a 32 GB card at this context, so the page moved to the 26B-A4B, a
mixture-of-experts model that is also several times faster. The story kept its shape — the
original wording fails, the variant fixes it — but the 26B fails a second way, which the 31B
did not in our runs: it takes loading the skill for doing the job, then looks for a report
nothing wrote, or says it wrote one. Once the keys are fixed, that is the failure left.

**What it costs.** On one RTX 5090 with Claude Code: five to forty-five seconds per full run
of this task; 2.8 minutes to resample the write twenty times — about eight seconds a sample,
since the model reasons there and every prompt is processed whole; 5.5 minutes for an A/B of
twenty per arm; 8.3 minutes to follow that A/B to the end, forty forks at about 12 seconds
each. In Part 4, the proposer took 37 seconds, screening its two proposals, 60 samples, 8.7
minutes, and validating one, thirty runs, 11 minutes. `--no-cache-prompt` adds a whole
prompt's processing to every step, about 1.5 seconds for this task's 14,000 tokens. Telling a
60% skill from an 80% one needs about 82 samples per arm — some eleven minutes an arm at this
checkpoint. The commands can run several samples at once (`--concurrency`), but when this
page ran the 31B that saved no time: four streams sharing one GPU each generated at a
fraction of the speed, and forks took 18 seconds each four at a time against 14 one at a
time. Measure it on your own server
before relying on it.
