// The page: starts a run, shows its progress every second, and lets the user
// review, edit, approve and export the letters.
//
// Vacancy texts come from strangers, so nothing from the server is ever put
// into innerHTML: every value goes through textContent (see el()).

const POLL_MS = 1000;
const CITATION = /\s*\[((?:p|e|ed|lang)\d+(?:\.f\d+)?|sk|av)((?:,\s*(?:(?:p|e|ed|lang)\d+(?:\.f\d+)?|sk|av))*)\]/g;

let current = null; // {runId, token}
let openDraft = null; // the draft shown in the letter view
let demo = null; // the saved demo run while it is being shown
let lastRun = null; // the last run this browser started, kept in localStorage
let parsedCv = null; // the visitor's CV as read by the LLM
let confirmedCv = null; // ... once the visitor confirmed it

const $ = (id) => document.getElementById(id);

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else node.setAttribute(key, value);
  }
  for (const child of children) {
    if (child != null) node.append(child);
  }
  return node;
}

async function api(path, options = {}) {
  const headers = { "Content-Type": "application/json" };
  if (current) headers["X-Run-Token"] = current.token;
  const response = await fetch(path, { ...options, headers });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = body.detail || `Request failed (${response.status})`;
    // A string, our {reason, message}, or FastAPI's list of input errors.
    const message = typeof detail === "string" ? detail : detail.message || "Please check your input.";
    const error = new Error(message);
    error.reason = detail.reason; // busy, ip_limit, daily_limit
    throw error;
  }
  return body;
}

// Starting and following a run

$("run-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const ownCv = usingOwnCv();
  if (ownCv && !confirmedCv) {
    showNotice("Read your CV and confirm it first.");
    return;
  }
  $("run-button").disabled = true;
  try {
    current = null;
    const run = await api("/api/runs", {
      method: "POST",
      body: JSON.stringify({ goal: $("goal").value, cv: ownCv ? confirmedCv : null }),
    });
    current = { runId: run.run_id, token: run.token };
    lastRun = current;
    showLastRunLink(true); // the new run is on screen: the link can hide it
    try { localStorage.setItem("lastRun", JSON.stringify(current)); } catch {}
    leaveDemo();
    closeLetter();
    $("output").hidden = false;
    poll();
  } catch (error) {
    // A limit, or anything else: say why and offer the saved run.
    showNotice(error.message);
    $("run-button").disabled = false;
  }
});

function showNotice(text) {
  $("notice-text").textContent = text;
  $("notice").hidden = false;
}

$("watch-demo").addEventListener("click", watchDemo);

async function poll() {
  if (!current) return;
  let run;
  try {
    run = await api(`/api/runs/${current.runId}`);
  } catch (error) {
    if (error.message === "Run not found") {
      current = null;
      lastRun = null;
      $("last-run").hidden = true;
      try { localStorage.removeItem("lastRun"); } catch {}
      $("output").hidden = true;
    }
    $("run-button").disabled = false;
    return;
  }
  showRun(run);
  if (run.status === "running") setTimeout(poll, POLL_MS);
  else $("run-button").disabled = false;
  if (run.status === "failed" && !run.drafts.length) {
    showNotice(`The agent could not finish: ${run.error}`);
  }
}

// The saved demo run, replayed step by step

async function watchDemo() {
  let saved;
  try {
    saved = await api("/api/demo");
  } catch {
    $("notice-text").textContent += " No saved run is available yet.";
    return;
  }
  current = null;
  demo = saved;
  showLastRunLink(false); // the demo replaces the last run on screen
  $("notice").hidden = true;
  closeLetter();
  const date = new Date(saved.started_at).toLocaleString();
  $("demo-text").textContent =
    `This is a replay of a real run from ${date} with the example CV. Letters are read-only here.`;
  $("demo-banner").hidden = false;
  $("output").hidden = false;
  for (let shown = 1; shown <= saved.trace.length; shown++) {
    if (demo !== saved) return; // a new run started meanwhile
    showRun(replayState(saved, shown));
    await new Promise((resolve) => setTimeout(resolve, 450));
  }
}

// The demo run as it looked after `shown` steps.
function replayState(run, shown) {
  const trace = run.trace.slice(0, shown);
  const done = shown === run.trace.length;
  const scored = new Set(trace.filter((s) => s.tool === "score_match" && s.ok).map((s) => s.result.job_id));
  const drafted = new Set(trace.filter((s) => s.tool === "draft_cover_letter" && s.ok).map((s) => s.result.draft_id));
  return {
    ...run,
    trace,
    status: done ? run.status : "running",
    answer: done ? run.answer : null,
    scores: run.scores.filter((s) => scored.has(s.job_id)),
    drafts: run.drafts.filter((d) => drafted.has(d.draft_id)),
    llm_calls: trace.filter((s) => s.kind === "llm").length,
    tokens: trace.filter((s) => s.kind === "llm").reduce((n, s) => n + s.prompt_tokens + s.completion_tokens, 0),
  };
}

function leaveDemo() {
  demo = null;
  $("demo-banner").hidden = true;
  $("notice").hidden = true;
}

function showRun(run) {
  showTrace(run);
  showScores(run.scores);
  showDrafts(run.drafts);
  $("answer-card").hidden = !run.answer;
  $("answer").textContent = run.answer || "";
}

// The trace panel: one live line saying what the agent does now, five stages,
// and all steps (one per agent decision) behind "Show all steps".

const STAGES = [
  ["CV", ["read_cv"]],
  ["Search", ["search_jobs"]],
  ["Score", ["score_match"]],
  ["Letters", ["draft_cover_letter", "request_approval"]],
  ["Summary", []],
];

// One group per agent decision: the LLM step and the tool calls it asked for.
function groupSteps(trace) {
  const groups = [];
  for (const step of trace) {
    if (step.kind === "llm") groups.push({ llm: step, calls: [] });
    else if (groups.length) groups[groups.length - 1].calls.push(step);
  }
  for (const group of groups) group.done = group.calls.length >= group.llm.tool_calls.length;
  return groups;
}

const plural = (n, one, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;

// "Scored 8 vacancies · best 86 good" when done, "Scoring 8 vacancies…" while running.
function groupTitle(group) {
  const names = group.llm.tool_calls;
  if (!names.length) return "Wrote the summary";
  const parts = [];
  for (const tool of [...new Set(names)]) {
    const asked = names.filter((n) => n === tool).length;
    const calls = group.calls.filter((c) => c.tool === tool && c.ok);
    const results = calls.map((c) => c.result);
    if (tool === "read_cv") {
      parts.push(group.done && results[0] ? `Read the CV · ${results[0].name}, ${results[0].location}` : "Reading the CV…");
    } else if (tool === "search_jobs") {
      const found = results.reduce((n, r) => n + r.count, 0);
      parts.push(group.done ? `Searched ${plural(asked, "query", "queries")} · ${found} found` : `Searching vacancies…`);
    } else if (tool === "score_match") {
      const best = results.reduce((b, r) => (b && b.score >= r.score ? b : r), null);
      parts.push(group.done
        ? `Scored ${plural(results.length, "vacancy", "vacancies")}${best ? ` · best ${best.score} ${best.label}` : ""}`
        : `Scoring ${plural(asked, "vacancy", "vacancies")}…`);
    } else if (tool === "draft_cover_letter") {
      const shaky = results.filter((r) => !r.grounded).length;
      parts.push(group.done
        ? `Wrote ${plural(results.length, "letter")} · ${shaky ? `${shaky} need${shaky === 1 ? "s" : ""} a look` : "all backed by the CV"}`
        : `Writing ${plural(asked, "cover letter")}…`);
    } else {
      parts.push(group.done ? `Asked for approval` : `Asking for approval…`);
    }
  }
  const errors = group.calls.filter((c) => !c.ok).length;
  return parts.join(" + ") + (errors ? ` · ${plural(errors, "error")}` : "");
}

function icon(state) {
  // state: "spin", "ok" or "failed"
  return el("span", { class: `icon ${state}`, text: state === "ok" ? "✓" : state === "failed" ? "✕" : "" });
}

function showTrace(run) {
  const groups = groupSteps(run.trace);
  const running = run.status === "running";
  const last = groups[groups.length - 1];

  // The live line
  let text;
  if (run.status === "failed") text = `Stopped: ${run.error}`;
  else if (!running) text = `Done · ${plural(run.drafts.length, "letter")} waiting for your approval`;
  else if (!last) text = "Starting the agent…";
  else if (!last.llm.tool_calls.length) text = "Writing the summary…";
  else if (!last.done) text = groupTitle(last);
  else text = "Deciding the next step…";
  $("now-icon").replaceWith(Object.assign(icon(running ? "spin" : run.status === "done" ? "ok" : "failed"), { id: "now-icon" }));
  $("now-text").textContent = text;
  // A live run counts from its start, also while it waits for the LLM; the demo
  // replay has no real clock, so it adds up the steps.
  const ms = demo
    ? run.trace.reduce((n, s) => n + (s.duration_ms || 0), 0)
    : (run.finished_at ? Date.parse(run.finished_at) : Date.now()) - Date.parse(run.started_at);
  $("run-time").textContent = ms > 0 ? `${Math.round(ms / 1000)} s` : "";

  // The stages: done, active (the latest one reached while running) or not yet
  const reached = STAGES.map(([, tools], i) =>
    i === STAGES.length - 1
      ? groups.some((g) => !g.llm.tool_calls.length)
      : groups.some((g) => g.llm.tool_calls.some((t) => tools.includes(t))),
  );
  const latest = reached.lastIndexOf(true);
  $("stages").replaceChildren(
    ...STAGES.map(([label], i) => {
      let state = "";
      if (reached[i] && (i < latest || !running)) state = "done";
      else if (i === latest && running) state = "active";
      return el("li", { class: state, text: label });
    }),
  );

  // All steps; keep open the ones the user opened, the list is rebuilt every second
  const list = $("trace");
  const open = new Set([...list.querySelectorAll("details")].map((d, i) => (d.open ? i : -1)));
  list.replaceChildren(
    ...groups.map((group, i) => {
      const failed = group.calls.some((c) => !c.ok);
      const state = !group.done && running ? "spin" : failed ? "failed" : "ok";
      const tokens = group.llm.prompt_tokens + group.llm.completion_tokens;
      const item = el("li", {},
        icon(state),
        el("div", { class: "step-title", text: groupTitle(group) }),
        el("div", { class: "step-meta", text: `${(group.llm.duration_ms / 1000).toFixed(1)} s to decide · ${tokens.toLocaleString()} tokens` }),
      );
      for (const call of group.calls.filter((c) => !c.ok)) item.append(el("div", { class: "step-error", text: call.error }));
      if (group.calls.length) {
        const details = el("details", {},
          el("summary", { text: "Details" }),
          el("ul", { class: "calls" }, ...group.calls.map((c) => el("li", { text: toolSummary(c) || c.tool }))),
          el("pre", { text: JSON.stringify(group.calls.map((c) => ({ tool: c.tool, arguments: c.arguments, result: c.result ?? c.error })), null, 2) }),
        );
        details.open = open.has(i);
        item.append(details);
      }
      return item;
    }),
  );
  const stats = [];
  if (run.llm_calls) stats.push(`${plural(run.llm_calls, "LLM call")}, ${run.tokens.toLocaleString()} tokens`);
  $("run-stats").textContent = stats.join(" · ");
}

function toolSummary(step) {
  const args = step.arguments || {};
  if (!step.ok) return `${step.tool}: failed`;
  const result = step.result || {};
  switch (step.tool) {
    case "search_jobs":
      return `"${args.query}"${args.location ? " in " + args.location : ""}${args.level ? ", " + args.level : ""}: ${result.count} found`;
    case "score_match":
      return `${result.title} (${result.company}): ${result.score}, ${result.label}`;
    case "draft_cover_letter":
      return `${result.title} (${result.company}): ${result.grounded ? "all sentences backed by the CV" : "some sentences need a look"}`;
    case "read_cv":
      return `${result.name}, ${result.location}`;
    case "request_approval":
      return `${args.action} for draft ${args.draft_id}: ${result.status}`;
    default:
      return null;
  }
}

function showScores(scores) {
  $("no-scores").hidden = scores.length > 0;
  $("scores").hidden = scores.length === 0;
  const body = $("scores").querySelector("tbody");
  body.replaceChildren();
  for (const s of scores) {
    const title = s.url
      ? el("a", { href: s.url, target: "_blank", rel: "noopener noreferrer", text: s.title })
      : el("span", { text: s.title });
    body.append(
      el("tr", {},
        el("td", {}, title, el("div", { class: "company", text: s.company })),
        el("td", { class: "score" }, el("span", { class: `badge ${s.label}`, text: `${s.score} ${s.label}` })),
        el("td", { class: "gaps", text: s.gaps.length ? s.gaps.slice(0, 4).join(", ") : "none" }),
      ),
    );
  }
}

function showDrafts(drafts) {
  $("no-drafts").hidden = drafts.length > 0;
  const box = $("drafts");
  box.replaceChildren();
  for (const d of drafts) {
    const button = el("button", { type: "button", class: "secondary", text: "Review" });
    button.addEventListener("click", () => openLetter(d.draft_id));
    const state = d.status === "draft" ? "waiting for you" : d.status;
    box.append(
      el("div", { class: "draft" },
        el("div", {},
          el("div", { class: "step-title", text: `${d.title} · ${d.company}` }),
          el("div", { class: "step-meta", text: d.grounded ? "All sentences backed by the CV" : "Some sentences need a look" }),
        ),
        el("div", { class: "buttons" }, el("span", { class: `badge ${d.status}`, text: state }), button),
      ),
    );
  }
}

// The letter view

async function openLetter(draftId) {
  const draft = demo ? demo.letters[draftId] : await api(`/api/runs/${current.runId}/drafts/${draftId}`);
  openDraft = draft;
  showLetter(draft);
  $("letter").hidden = false;
  $("letter").scrollIntoView({ behavior: "smooth", block: "start" });
}

function closeLetter() {
  openDraft = null;
  $("letter").hidden = true;
}
$("letter-close").addEventListener("click", closeLetter);

function showLetter(draft) {
  $("letter-title").textContent = `${draft.vacancy.title} · ${draft.vacancy.company}`;
  $("letter-meta").textContent = draft.vacancy.location || "";
  $("letter-text").replaceChildren(...renderLetter(draft));
  $("letter-edit").value = draft.edited_text || draft.plain_text;
  $("letter-edit").disabled = demo !== null || draft.status !== "draft";

  const status = $("letter-status");
  const labels = { draft: "waiting for your approval", approved: "approved", rejected: "rejected" };
  status.textContent = labels[draft.status];
  status.className = `badge ${draft.status}`;

  const canAct = demo === null; // the demo is read-only
  $("approve").hidden = !canAct || draft.status !== "draft";
  $("reject").hidden = !canAct || draft.status !== "draft";
  $("export").hidden = !canAct || draft.status !== "approved";
  if (demo) {
    status.textContent = "saved run, read-only";
    status.className = "badge";
  }
  $("letter-error").hidden = true;
}

// Splits the letter into sentences; each claim keeps the facts it cites.
function renderLetter(draft) {
  return draft.text.split(/\n\n+/).map((paragraph) => {
    const p = el("p");
    let last = 0;
    for (const match of paragraph.matchAll(CITATION)) {
      const ids = match[0].replace(/[\s\[\]]/g, "").split(",");
      addSentences(p, paragraph.slice(last, match.index), ids, draft);
      last = match.index + match[0].length;
    }
    addSentences(p, paragraph.slice(last), [], draft);
    return p;
  });
}

function addSentences(p, text, ids, draft) {
  // Only the last sentence before a citation is the claim; earlier ones cite nothing.
  const sentences = text.trim().split(/(?<=[.!?])\s+(?=[A-Z])/).filter(Boolean);
  sentences.forEach((sentence, i) => {
    const cites = i === sentences.length - 1 ? ids : [];
    const span = el("span", { class: "sentence", text: sentence });
    const facts = cites.map((id) => [id, draft.cited_facts[id]]).filter(([, fact]) => fact);
    const issues = draft.issues.filter((issue) => issue.sentence.includes(sentence) || sentence.includes(issue.sentence));
    if (facts.length) span.classList.add("cited");
    if (issues.length) span.classList.add("issue");
    if (facts.length || issues.length) {
      span.tabIndex = 0;
      span.addEventListener("mouseenter", () => showTooltip(span, facts, issues));
      span.addEventListener("focus", () => showTooltip(span, facts, issues));
      span.addEventListener("mouseleave", hideTooltip);
      span.addEventListener("blur", hideTooltip);
    }
    p.append(span, " ");
  });
}

function showTooltip(target, facts, issues) {
  const tip = $("tooltip");
  tip.replaceChildren();
  for (const [id, fact] of facts) tip.append(el("b", { text: `CV fact ${id}` }), el("span", { text: fact }));
  for (const issue of issues) tip.append(el("b", { text: "Check failed" }), el("span", { text: issue.problem }));
  tip.hidden = false;
  const box = target.getBoundingClientRect();
  const top = box.bottom + 8 + tip.offsetHeight > window.innerHeight ? box.top - tip.offsetHeight - 8 : box.bottom + 8;
  tip.style.top = `${Math.max(8, top)}px`;
  tip.style.left = `${Math.min(Math.max(8, box.left), window.innerWidth - tip.offsetWidth - 8)}px`;
}

function hideTooltip() {
  $("tooltip").hidden = true;
}

async function sendDecision(decision) {
  const edited = $("letter-edit").value.trim();
  const body = { decision };
  if (decision === "approved" && edited !== openDraft.plain_text.trim()) body.edited_text = edited;
  await withButtons(async () => {
    const draft = await api(`/api/runs/${current.runId}/drafts/${openDraft.draft_id}/decision`, {
      method: "POST",
      body: JSON.stringify(body),
    });
    openDraft = draft;
    showLetter(draft);
    poll(); // refresh the drafts list
  });
}

$("approve").addEventListener("click", () => sendDecision("approved"));
$("reject").addEventListener("click", () => sendDecision("rejected"));

$("export").addEventListener("click", () =>
  withButtons(async () => {
    const { text } = await api(`/api/runs/${current.runId}/drafts/${openDraft.draft_id}/export`, { method: "POST" });
    const file = new Blob([text], { type: "text/plain" });
    const link = el("a", { href: URL.createObjectURL(file), download: `cover-letter-${openDraft.vacancy.id}.txt` });
    link.click();
    URL.revokeObjectURL(link.href);
    try { await navigator.clipboard.writeText(text); } catch {}
  }),
);

async function withButtons(action) {
  const buttons = ["approve", "reject", "export"].map($);
  buttons.forEach((b) => (b.disabled = true));
  $("letter-error").hidden = true;
  try {
    await action();
  } catch (error) {
    $("letter-error").textContent = error.message;
    $("letter-error").hidden = false;
  } finally {
    buttons.forEach((b) => (b.disabled = false));
  }
}

// The visitor's own CV: paste, read with the LLM, check, confirm.

function usingOwnCv() {
  return document.querySelector('input[name="cv"]:checked').value === "own";
}

function showCvLabel() {
  if (!usingOwnCv()) $("cv-label").textContent = "CV: the example";
  else if (confirmedCv) $("cv-label").textContent = `CV: yours (${confirmedCv.name || "no name"}), confirmed`;
  else $("cv-label").textContent = "CV: yours, not confirmed yet";
}

for (const radio of document.querySelectorAll('input[name="cv"]')) {
  radio.addEventListener("change", () => {
    $("own-cv").hidden = !usingOwnCv();
    $("cv-review").hidden = !usingOwnCv() || !parsedCv;
    showCvLabel();
  });
}

$("cv-text").addEventListener("input", () => {
  // A changed text must be read and confirmed again.
  if (confirmedCv) {
    confirmedCv = null;
    showCvConfirmed();
    showCvLabel();
  }
});

$("read-cv").addEventListener("click", async () => {
  $("read-cv").disabled = true;
  $("cv-reading").hidden = false;
  $("notice").hidden = true;
  try {
    const result = await api("/api/cv/parse", {
      method: "POST",
      body: JSON.stringify({ text: $("cv-text").value }),
    });
    parsedCv = result.cv;
    confirmedCv = null;
    showCvReview(result);
  } catch (error) {
    showNotice(error.message);
  } finally {
    $("read-cv").disabled = false;
    $("cv-reading").hidden = true;
    showCvLabel();
  }
});

function showCvReview(result) {
  const cv = result.cv;
  const parts = [cv.name, cv.headline, cv.location, cv.availability].filter(Boolean);
  $("cv-profile").textContent = `${parts.join(" · ")} — looking for: ${cv.looking_for.join(", ") || "not stated"}`;
  $("cv-verified").textContent = result.verified_skills.join(", ") || "none";
  $("cv-unverified").textContent = result.unverified_skills.join(", ") || "none";

  $("cv-warnings").hidden = result.warnings.length === 0;
  $("cv-warnings").replaceChildren(...result.warnings.map((w) => el("li", { text: w })));
  $("cv-facts").replaceChildren(
    ...Object.entries(result.facts).map(([id, fact]) =>
      el("li", {}, el("span", { class: "fact-id", text: id }), el("span", { text: fact })),
    ),
  );
  showCvConfirmed();
  $("cv-review").hidden = false;
  $("cv-review").scrollIntoView({ behavior: "smooth", block: "start" });
}

function showCvConfirmed() {
  $("cv-confirmed").textContent = confirmedCv ? "confirmed" : "";
  $("cv-confirmed").className = confirmedCv ? "badge approved" : "badge";
  $("confirm-cv").hidden = confirmedCv !== null;
}

$("confirm-cv").addEventListener("click", () => {
  confirmedCv = parsedCv;
  showCvConfirmed();
  showCvLabel();
  $("start").scrollIntoView({ behavior: "smooth", block: "start" });
});

// The page opens empty. If this browser started a run before, a link shows it,
// and the same link hides it again.
try {
  lastRun = JSON.parse(localStorage.getItem("lastRun") || "null");
} catch {}
$("last-run").hidden = !lastRun;

function showLastRunLink(runIsShown) {
  $("last-run").hidden = !lastRun;
  $("last-run").textContent = runIsShown ? "Hide my last run" : "Show my last run";
}

$("last-run").addEventListener("click", () => {
  const runIsShown = current !== null && !$("output").hidden;
  if (runIsShown) {
    current = null; // also stops the polling
    closeLetter();
    $("output").hidden = true;
    $("run-button").disabled = false;
    showLastRunLink(false);
  } else {
    leaveDemo();
    current = lastRun;
    $("output").hidden = false;
    $("run-button").disabled = true;
    showLastRunLink(true);
    poll();
  }
});
