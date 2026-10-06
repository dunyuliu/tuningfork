// A step's details, as both the run's timeline and a step's exact-context page show them:
// what it cost, what it was sent with, what tuningfork changed about it, the skills in its
// context, and the workspace its call found.

import { api, enc } from "./api.js";
import { chip, count, h, kv, millis, panel, plural, pretty, replace, short } from "./dom.js";
import { fileChip, skillAnchor } from "./payload.js";

export function summaryStrip(step) {
  const usage = step.usage || {};
  const item = (label, value) =>
    value === null || value === undefined || value === "" ? null : h("span", { class: "item" }, h("span", null, label), h("span", null, value));
  return h(
    "div",
    { class: "summary-strip" },
    item("model", step.model ? h("span", { class: "mono" }, step.model) : null),
    item("wire", step.wire && step.wire !== "openai-chat" ? h("span", { class: "mono" }, step.wire) : null),
    item("finish", step.finish_reason),
    item("tokens", usage.prompt_tokens !== undefined ? `${count(usage.prompt_tokens)} in → ${count(usage.completion_tokens)} out` : null),
    item("first token", step.timings?.ttft_ms != null ? millis(step.timings.ttft_ms) : null),
    item("total", step.timings?.total_ms != null ? millis(step.timings.total_ms) : null),
    step.edits && Object.keys(step.edits).length ? chip("edited at a breakpoint", "response") : null,
    step.substitution?.spans ? chip("variant swapped in", "skill") : null,
  );
}

/** What happened at a step that is not the model answering: a replay, a variant, an edit, a
 *  failure. `original` makes the way to what the harness sent instead: by default the
 *  exact-context page's own toggle, named; elsewhere, a link to that page. */
export function banners(step, { original = null } = {}) {
  const out = [];
  const replay = step.replay || {};
  const compare = (text) => (original ? original() : text);
  if (replay.served_from) {
    // A fork's step before the one it went live at: the model was not asked.
    out.push(h("div", { class: "banner" }, h("strong", null, "Answered from the recording. "),
      `This fork replayed step ${replay.source_idx} of run ${(replay.source_run || "").slice(0, 8)} here; the harness re-ran the tool calls in it`,
      replay.workspace ? `, and found its workspace ${replay.workspace === "same" ? "as the recording's was" : "different from the recording's"}.` : "."));
  }
  if (replay.live_from) {
    out.push(h("div", { class: "banner" }, h("strong", null, "The fork went live here. "),
      "Everything before this step was answered from the recording",
      replay.workspace ? `; the workspace was ${replay.workspace === "same" ? "the recording's" : "not the recording's"} at this point.` : "."));
  }
  if (replay.mismatch) {
    out.push(h("div", { class: "banner banner-bad" }, h("strong", null, "The replay left the recording here. "),
      "The harness asked for something the recording never answered, so this fork's history is not the recorded one, and it is not counted."));
  }
  if (step.substitution?.spans) {
    out.push(h("div", { class: "banner banner-edit" }, h("strong", null, "A variant wording was swapped in. "),
      `The model saw the variant in ${plural(step.substitution.spans, "place")}; the harness sent the original — `,
      compare("see “The original wording” for it."),
      step.substitution.leaked ? " The original wording still reached the model in another form." : ""));
  }
  const request = step.edits?.request;
  if (request) {
    const positions = step.edited_messages ?? request.messages ?? [];
    const what = [
      positions.length ? `message${positions.length === 1 ? "" : "s"} ${positions.map((i) => `#${i}`).join(", ")}` : null,
      ...(request.fields || []).filter((field) => field !== "messages"),
    ].filter(Boolean).join("; ");
    out.push(h("div", { class: "banner banner-edit" }, h("strong", null, "The context was edited before the model saw it"),
      ` (${what}). This step records what the model saw — `, compare("switch to “What the harness sent” to compare.")));
  }
  const response = step.edits?.response;
  const undelivered = step.error && step.error.startsWith("tuningfork could not deliver");
  if (step.error && step.response && !response) {
    // A stream that broke off keeps what arrived: shown, but it is not a behaviour.
    out.push(h("div", { class: "banner banner-bad" }, h("strong", null, "This call failed. "),
      `${step.error}. What arrived before it did is shown; it counts as an error, not as behaviour.`));
  }
  if (response?.dropped) {
    out.push(h("div", { class: "banner banner-bad" }, h("strong", null, "Dropped at a breakpoint. "),
      "The model answered, but the harness received an error instead."));
  } else if (response && undelivered) {
    out.push(h("div", { class: "banner banner-bad" }, h("strong", null, "The response was edited, but the edit could not be delivered. "),
      `The harness received an error instead (${step.error}). The step's behaviour is what the model produced.`));
  } else if (response) {
    const fields = (response.message_fields || response.fields || []).join(", ");
    out.push(h("div", { class: "banner banner-edit" }, h("strong", null, "The response was edited before the harness acted on it"),
      ` (${fields}). The step's behaviour is still what the model produced; the harness acted on the edited version.`));
  }
  return out.length ? h("div", { class: "banners" }, out) : null;
}

/** The skills in a context: each named, placed, and mapped to its file. `anchor` turns a
 *  span into the id of where it is shown on this page. */
export function skillsSummary(spans, recorded, { anchor = skillAnchor } = {}) {
  return h(
    "div",
    { class: "skills-list" },
    spans.map((span) =>
      h("div", { class: "skill-ref" },
        chip("skill", "skill"),
        h("a", { href: `#${anchor(span)}`, onclick: (event) => { event.preventDefault(); jumpTo(anchor(span)); } }, span.name),
        h("span", { class: "muted small" }, span.source === "preloaded" ? `preloaded in message #${span.message_index}` : `loaded by a tool, message #${span.message_index}`),
        fileChip(span, recorded),
        h("span", { class: "faint small" }, plural(span.lines, "line")))),
  );
}

export function jumpTo(id) {
  const target = document.getElementById(id);
  if (!target) return;
  const message = target.closest("details");
  if (message) message.open = true;
  target.scrollIntoView({ behavior: "smooth", block: "start" });
}

export function toolsOffered(tools) {
  if (!tools?.length) return null;
  const names = tools.map((tool) => tool.function?.name || tool.name || "?");
  return h(
    "details",
    { class: "fold tools-offered" },
    h("summary", null, h("span", null, `${plural(names.length, "tool")} offered: `), h("span", { class: "fold-preview" }, names.join(", "))),
    h("pre", { class: "code" }, pretty(tools)),
  );
}

/** The workspace as this call found it: every tool effect before it, on disk. Only a run
 *  tuningfork launched has one. */
export function workspacePanel(step) {
  if (!step.workspace_commit) return null;
  const body = h("p", { class: "muted" }, "Loading…");
  const holder = h("div", null, body);
  api.get(`/runs/${enc(step.run.id)}/files?at=${step.idx}`).then((view) => {
    if (view.error) {
      replace(holder, h("p", { class: "muted" }, view.error));
      return;
    }
    const status = new Map((view.changes || []).map((change) => [change.path, change.status]));
    const deleted = (view.changes || []).filter((change) => change.status === "D").map((change) => change.path);
    replace(
      holder,
      view.files.length
        ? h("table", { class: "table" }, h("tbody", null, view.files.map((file) =>
            h("tr", null,
              h("td", { class: "mono" }, file.path),
              h("td", { class: "num faint" }, file.size != null ? `${file.size} bytes` : ""),
              h("td", null, status.has(file.path)
                ? chip(status.get(file.path) === "A" ? "new since the step before" : "changed since the step before", "accent")
                : null)))))
        : h("p", { class: "muted" }, "Empty."),
      deleted.length ? h("p", { class: "muted" }, `Deleted since the step before: ${deleted.join(", ")}`) : null,
      h("p", { class: "faint" }, `tuningfork files ${short(step.run.id)} ${step.idx} --diff shows the changes; --out <dir> writes the files out.`),
    );
  }).catch((error) => replace(holder, h("p", { class: "muted" }, String(error?.message || error))));
  return panel("Workspace", holder, {
    meta: h("span", { class: "muted" }, "as this call found it · snapshot ", h("span", { class: "mono" }, short(step.workspace_commit))),
  });
}

export function detailsPanel(step) {
  const params = step.params || {};
  const defaults = step.server_defaults || {};
  return panel("Sampling and timing", [
    kv([
      ...Object.entries(params).map(([key, value]) => [key, h("span", { class: "mono" }, typeof value === "object" ? JSON.stringify(value) : String(value))]),
      ["time to first token", step.timings?.ttft_ms != null ? millis(step.timings.ttft_ms) : null],
      ["total time", step.timings?.total_ms != null ? millis(step.timings.total_ms) : null],
    ]),
    Object.keys(defaults).length
      ? h("details", { class: "fold" }, h("summary", null, "Server defaults at the time"), h("pre", { class: "code" }, pretty(defaults)))
      : null,
  ]);
}
