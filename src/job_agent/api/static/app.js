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
  $("run-button").disabled = true;
  try {
    current = null;
    const run = await api("/api/runs", {
      method: "POST",
      body: JSON.stringify({ goal: $("goal").value }),
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
  const status = $("run-status");
  status.textContent = run.status;
  status.className = `badge ${run.status}`;
  showTrace(run.trace, run.status);
  showScores(run.scores);
  showDrafts(run.drafts);

  const stats = [];
  if (run.llm_calls) stats.push(`${run.llm_calls} LLM calls, ${run.tokens.toLocaleString()} tokens`);
  if (run.error) stats.push(run.error);
  $("run-stats").textContent = stats.join(" · ");

  $("answer-card").hidden = !run.answer;
  $("answer").textContent = run.answer || "";
}

const TOOL_LABELS = {
  read_cv: "Read the CV",
  search_jobs: "Searched vacancies",
  score_match: "Scored a vacancy",
  draft_cover_letter: "Wrote a cover letter",
  request_approval: "Asked for approval",
};

function toolSummary(step) {
  const args = step.arguments || {};
  if (!step.ok) return null;
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

function showTrace(trace, status) {
  const list = $("trace");
  list.replaceChildren();
  for (const step of trace) {
    const seconds = (step.duration_ms / 1000).toFixed(1);
    if (step.kind === "llm") {
      const decided = step.tool_calls.length
        ? `Decided to call ${countNames(step.tool_calls)}`
        : "Wrote the final answer";
      list.append(
        el("li", { class: "llm" },
          el("div", { class: "step-title", text: decided }),
          el("div", { class: "step-meta", text: `${seconds} s · ${step.prompt_tokens + step.completion_tokens} tokens` }),
        ),
      );
    } else {
      const item = el("li", { class: step.ok ? "ok" : "failed" },
        el("div", { class: "step-title", text: TOOL_LABELS[step.tool] || step.tool }),
      );
      const summary = toolSummary(step);
      if (summary) item.append(el("div", { class: "step-meta", text: summary }));
      if (!step.ok) item.append(el("div", { class: "step-error", text: step.error }));
      item.append(
        el("details", {},
          el("summary", { text: "Details" }),
          el("pre", { text: JSON.stringify({ arguments: step.arguments, result: step.result ?? step.error }, null, 2) }),
        ),
      );
      list.append(item);
    }
  }
  if (status === "running") {
    list.append(el("li", { class: "waiting", text: trace.length ? "Working…" : "Starting the agent…" }));
  }
}

function countNames(names) {
  const counts = {};
  for (const name of names) counts[name] = (counts[name] || 0) + 1;
  return Object.entries(counts)
    .map(([name, n]) => (n > 1 ? `${name} ×${n}` : name))
    .join(", ");
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
