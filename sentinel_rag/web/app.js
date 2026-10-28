"use strict";
const $ = (s, el = document) => el.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (v, d = 2) => (v === null || v === undefined ? "n/a" : typeof v === "number" ? v.toFixed(d) : esc(v));
let STATUS = null;
let LAST = null;

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || `HTTP ${r.status}`);
  return body;
}

const EXAMPLES = [
  ["What is the minimum attendance required to sit the end-semester examination, and are there any exceptions?", "requirement + exception"],
  ["What is the late submission penalty per day?", "conflicting versions"],
  ["How much attendance do students need to write exams?", "low-quality forum source"],
  ["What late submission penalty did the 2023 handbook specify?", "historical"],
  ["What is the annual hostel fee and does it include laundry?", "partly answerable"],
  ["What are the hostel fee and the re-examination fee?", "multi-document"],
  ["What is the parking fee for students?", "unanswerable"],
];

// ------------------------------------------------------------------ tabs
document.querySelectorAll("nav button").forEach((b) =>
  b.addEventListener("click", () => {
    document.querySelectorAll("nav button").forEach((x) => x.classList.toggle("active", x === b));
    document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.id === "tab-" + b.dataset.tab));
    if (b.dataset.tab === "corpus") loadDocs();
    if (b.dataset.tab === "eval") loadEval();
  })
);

// ------------------------------------------------------------------ status
async function loadStatus() {
  try {
    STATUS = await api("/api/status");
  } catch (e) {
    $("#status").innerHTML = `<span class="err">Server unreachable: ${esc(e.message)}</span>`;
    return;
  }
  const g = STATUS.generators.find((x) => x.name === STATUS.default_generator);
  $("#status").innerHTML =
    `${STATUS.models_ready ? "Models ready" : '<span class="spinner"></span>Loading embedder + verifier…'} · ` +
    `<b>${STATUS.corpus.documents}</b> docs, <b>${STATUS.corpus.chunks}</b> chunks · generator <b>${esc(g ? g.name + " / " + g.model : "?")}</b>` +
    (STATUS.model_error ? `<br><span class="err">${esc(STATUS.model_error)}</span>` : "");
  if (!STATUS.models_ready && !STATUS.model_error) setTimeout(loadStatus, 1500);
}

function fillSelectors() {
  const cond = $("#condition");
  cond.innerHTML = Object.entries(STATUS.conditions).map(([k, v]) => `<option value="${k}" ${k === "sentinel" ? "selected" : ""}>${esc(v)}</option>`).join("");
  const gens = STATUS.generators.map((g) => `<option value="${g.name}" ${g.available ? "" : "disabled"} ${g.name === STATUS.default_generator ? "selected" : ""}>${esc(g.name)}${g.available ? "" : " (no key)"}</option>`).join("");
  $("#generator").innerHTML = gens;
  $("#eval-gen").innerHTML = gens;
  $("#eval-gen").value = "extractive";
  const fillModels = () => {
    const g = STATUS.generators.find((x) => x.name === $("#generator").value);
    const ms = g && g.models.length ? g.models : [g ? g.model : ""];
    $("#model").innerHTML = ms.map((m) => `<option ${m === g.model ? "selected" : ""}>${esc(m)}</option>`).join("");
    $("#model").disabled = !g || !g.models.length;
  };
  $("#generator").addEventListener("change", fillModels);
  fillModels();
  $("#u-rel").innerHTML = STATUS.reliability_categories.map((c) => `<option ${c === "unknown" ? "selected" : ""}>${c}</option>`).join("");
  $("#topk").value = STATUS.settings.top_k;
  $("#rounds").value = STATUS.settings.max_rounds;
  $("#examples").innerHTML = EXAMPLES.map(([q, tag], i) => `<button type="button" class="chip" data-i="${i}">${esc(q)} <small>· ${esc(tag)}</small></button>`).join("");
  $("#examples").addEventListener("click", (e) => {
    const b = e.target.closest(".chip");
    if (!b) return;
    $("#question").value = EXAMPLES[b.dataset.i][0];
    $("#ask-form").requestSubmit();
  });
  renderMethod();
}

// ------------------------------------------------------------------ ask
$("#ask-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = $("#question").value.trim();
  if (!q) return;
  const btn = $("#ask-btn");
  btn.disabled = true;
  $("#result").innerHTML = `<div class="card"><span class="spinner"></span>Retrieving, scoring, generating and verifying…</div>`;
  try {
    const rec = await api("/api/ask", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q, condition: $("#condition").value, generator: $("#generator").value,
        model: $("#model").disabled ? null : $("#model").value, top_k: +$("#topk").value, max_rounds: +$("#rounds").value }),
    });
    LAST = rec;
    $("#result").innerHTML = renderRecord(rec, true);
  } catch (err) {
    $("#result").innerHTML = `<div class="card err">${esc(err.message)}</div>`;
  } finally {
    btn.disabled = false;
  }
});

function decisionBadge(d) {
  const label = { answer: "Answered", qualified: "Qualified answer", abstain: "Abstained", error: "Error" }[d] || d;
  return `<span class="decision d-${esc(d)}">${esc(label)}</span>`;
}

function renderAnswer(rec) {
  const claims = rec.claims || [];
  if (rec.decision === "error") return `<div class="reason err">${esc(rec.error?.type)}: ${esc(rec.error?.message)}</div>`;
  let html = "";
  if (rec.answer) {
    html += `<div class="answer">` + (claims.length
      ? claims.map((k) => {
          const cls = k.kind === "gap" ? "gap" : k.relation;
          const cites = (k.cited || []).map((c) => `<span class="cite">${esc(c)}</span>`).join("");
          return `<span class="claim ${cls}" data-kid="${k.kid}" title="${esc(k.kind === "gap" ? "gap statement" : k.relation)}">${esc(k.text)}</span>${cites} `;
        }).join("")
      : esc(rec.answer)) + `</div>`;
  }
  if (rec.qualification) html += `<div class="qual">${esc(rec.qualification)}</div>`;
  if (rec.reason) html += `<div class="reason">${esc(rec.reason)}</div>`;
  if (rec.withheld_draft) html += `<details><summary class="hint">Withheld draft (not shown as an answer)</summary><p class="hint">${esc(rec.withheld_draft)}</p></details>`;
  return html;
}

function renderRecord(rec, full) {
  const labels = rec.labels || {};
  const inv = Object.fromEntries(Object.entries(labels).map(([k, v]) => [v, k]));
  const ctx = new Set(rec.context || []);
  let h = `<div class="card">
    <div class="row" style="justify-content:space-between;align-items:center;margin:0">
      <div>${decisionBadge(rec.decision)} <span class="hint">${esc(rec.condition_label)}</span></div>
      <div class="meta"><span>${esc(rec.generator)}: ${esc(rec.generator_model)}</span><span>config ${esc(rec.config_hash)}</span><span>${fmt(rec.timing?.total_s)} s</span></div>
    </div>
    ${renderAnswer(rec)}
    <div class="legend"><span class="claim supported">supported</span><span class="claim partially_supported">partially supported</span><span class="claim contradicted">contradicted</span><span class="claim unsupported">unsupported</span><span class="hint">Click a claim for its evidence span and provenance.</span></div>
  </div>`;
  if (!full) return h;

  // Facets + sufficiency
  const cov = Object.fromEntries((rec.coverage || []).map((c) => [c.fid, c]));
  const suff = rec.sufficiency;
  h += `<div class="cols"><div class="card"><h2>Question facets</h2><table><tr><th>Facet</th><th>Covered by</th></tr>` +
    (rec.facets || []).map((f) => {
      const c = cov[f.fid];
      const by = c ? (c.covered ? c.by.map((id) => `<span class="tag ok">${esc(inv[id] || "")} ${esc(id)}</span>`).join("") : `<span class="tag bad">not covered</span>${c.missing_cues?.length ? ` <span class="hint">missing cue: ${esc(c.missing_cues.join(", "))}</span>` : ""}`) : "n/a";
      return `<tr><td><b>${f.fid}</b>${f.central ? " (central)" : ""} ${esc(f.text)}</td><td>${by}</td></tr>`;
    }).join("") + `</table><p class="hint">Temporal need: ${esc(rec.temporal_need)}</p></div>`;
  h += `<div class="card"><h2>Evidence sufficiency</h2>` + (suff
    ? `<p><b>E = ${fmt(suff.E)}</b> → <span class="tag ${suff.band === "high" ? "ok" : suff.band === "medium" ? "warn" : "bad"}">${suff.band}</span></p>
       <table>${[["T̄ aggregate trust", suff.T_bar, STATUS.settings.s_trust], ["COV facet coverage", suff.COV, STATUS.settings.s_coverage], ["AGR agreement", suff.AGR, STATUS.settings.s_agreement], ["IND independent sources", suff.IND, STATUS.settings.s_independence]]
         .map(([n, v, w]) => `<tr><td>${n}</td><td class="num">${fmt(v)}</td><td><span class="bar" style="width:${Math.round(v * 120)}px"></span></td><td class="num">× ${w}</td></tr>`).join("")}</table>
       ${suff.hard_checks?.length ? `<p class="hint">Hard checks: ${esc(suff.hard_checks.join("; "))}</p>` : ""}
       <p class="hint">Bands: high ≥ ${STATUS.settings.band_high}, medium ≥ ${STATUS.settings.band_medium}. A heuristic policy score, not a probability that the answer is true.</p>`
    : `<p class="hint">Not computed for this condition.</p>`) + `</div></div>`;

  // Rounds
  h += `<div class="card"><h2>Retrieval rounds <span class="hint">(${rec.counters?.retrieval_calls} of max ${$("#rounds").value || "?"})</span></h2>` +
    (rec.rounds || []).map((r) => `<div class="round"><div><b>Round ${r.round}</b> <span class="tag">${esc(r.trigger)}</span> ${r.hits} hits, ${r.new_chunks} new</div>
      <div class="q">${esc(r.query)}</div>
      ${r.sufficiency ? `<div class="hint">E = ${fmt(r.sufficiency.E)} (${r.sufficiency.band}) · coverage ${fmt(r.sufficiency.COV)}</div>` : ""}
      ${r.action ? `<div class="hint">→ ${esc(r.action)}</div>` : ""}</div>`).join("") +
    `<p class="meta"><span>generator calls ${rec.counters?.generator_calls}</span><span>NLI pairs ${rec.counters?.nli_pairs}</span><span>generated tokens ${rec.counters?.generated_tokens}</span>${Object.entries(rec.timing || {}).map(([k, v]) => `<span>${esc(k)} ${fmt(v)}</span>`).join("")}</p></div>`;

  // Conflicts
  if (rec.conflicts?.length) {
    h += `<div class="card"><h2>Detected conflicts</h2><table><tr><th>Passages</th><th>Claim</th><th class="num">P(contra.)</th><th>Resolution</th></tr>` +
      rec.conflicts.map((c) => `<tr><td class="mono">${esc(c.a)}<br>${esc(c.b)}</td><td>${esc(c.claim)}</td><td class="num">${fmt(c.contradiction)}</td><td>${c.resolved ? `<span class="tag ok">prefer ${esc(c.preferred)}</span>` : `<span class="tag warn">unresolved</span>`}<div class="hint">${esc(c.reason)}</div></td></tr>`).join("") +
      `</table><p class="hint">Contradictory evidence is retained in the record, not discarded.</p></div>`;
  }

  // Candidates
  h += `<div class="card"><h2>Candidate pool and trust scores</h2><p class="hint">T = ${STATUS.settings.w_relevance}·R + ${STATUS.settings.w_reliability}·S + ${STATUS.settings.w_consistency}·C + ${STATUS.settings.w_freshness}·F. Rows in bold were passed to the generator.</p><div class="scroll"><table>
    <tr><th>ID</th><th>Chunk</th><th>Source</th><th class="num">R</th><th class="num">S</th><th class="num">C</th><th class="num">F</th><th class="num">T</th><th>Flags</th></tr>` +
    (rec.candidates || []).slice(0, 15).map((c) => `<tr style="${ctx.has(c.chunk_id) ? "font-weight:600" : "opacity:.75"}">
      <td>${ctx.has(c.chunk_id) ? `<span class="tag ok">${esc(inv[c.chunk_id] || "ctx")}</span>` : ""}</td>
      <td><div class="mono">${esc(c.chunk_id)}</div><div class="hint">${esc(c.locator)}</div><div style="font-weight:400">${esc(c.text.slice(0, 180))}${c.text.length > 180 ? "…" : ""}</div></td>
      <td>${esc(c.title)}<div><span class="tag ${c.reliability === "official" ? "ok" : ["unverified", "blog", "unknown"].includes(c.reliability) ? "bad" : ""}">${esc(c.reliability)}</span><span class="tag">${esc(c.published || "undated")}</span>${c.synthetic ? '<span class="tag">synthetic</span>' : ""}</div></td>
      <td class="num">${fmt(c.R)}</td><td class="num">${fmt(c.S)}</td><td class="num">${fmt(c.C)}</td><td class="num">${fmt(c.F)}</td><td class="num"><b>${fmt(c.trust)}</b></td>
      <td>${(c.flags || []).map((f) => `<span class="tag ${/conflict/.test(f) ? "bad" : "warn"}">${esc(f)}</span>`).join("")}</td></tr>`).join("") +
    `</table></div></div>`;

  // Drafts
  if (rec.drafts?.length > 1) {
    h += `<div class="card"><h2>Drafts</h2>` + rec.drafts.map((d, i) => `<p><b>Draft ${i + 1}</b> ${d.stage ? `<span class="tag">${esc(d.stage)}</span>` : ""}<br><span class="hint">${esc(d.text)}</span></p>`).join("") + `</div>`;
  }
  return h;
}

document.addEventListener("click", (e) => {
  const el = e.target.closest(".claim[data-kid]");
  if (!el) return;
  const card = el.closest("[data-rec]");
  const rec = card ? CMP[card.dataset.rec] : LAST;
  const k = rec && (rec.claims || []).find((x) => x.kid === el.dataset.kid);
  if (!k) return;
  const cands = Object.fromEntries((rec.candidates || []).map((c) => [c.chunk_id, c]));
  const ids = (k.best_chunk || "").startsWith("joint:") ? k.best_chunk.slice(6).split("+") : [k.best_chunk];
  let h = `<h2>${esc(k.kid)}: ${k.kind === "gap" ? "gap statement" : esc(k.relation)}</h2><p>${esc(k.text)}</p>`;
  if (k.kind === "gap") h += `<p class="hint">This sentence reports missing evidence, so it is not counted as a factual claim.</p>`;
  else {
    h += `<h3>Best evidence span</h3><div class="span">${esc(k.span)}</div>
      <p class="meta"><span>entailment ${fmt(k.probs?.entailment)}</span><span>neutral ${fmt(k.probs?.neutral)}</span><span>contradiction ${fmt(k.probs?.contradiction)}</span></p>`;
    ids.filter(Boolean).forEach((id) => {
      const c = cands[id];
      if (c) h += `<h3>Provenance</h3><table><tr><td>Chunk</td><td class="mono">${esc(id)}</td></tr><tr><td>Document</td><td>${esc(c.title)} (v${c.version})</td></tr><tr><td>Locator</td><td>${esc(c.locator)}</td></tr><tr><td>Source</td><td>${esc(c.reliability)}, ${esc(c.published || "undated")}, group ${esc(c.source_group)}</td></tr><tr><td>Trust</td><td>${fmt(c.trust)}</td></tr></table>`;
    });
    if (k.contradicting) h += `<h3>Contradicting span</h3><div class="span">${esc(k.contradicting.span)}</div><p class="hint">${esc(k.contradicting.chunk_id)} · P(contradiction) ${fmt(k.contradicting.contradiction)}</p>`;
    if (k.citations?.length) h += `<h3>Citation check</h3><table>${k.citations.map((c) => `<tr><td>${esc(c.label)}</td><td class="mono">${esc(c.chunk_id || "invalid ID")}</td><td>${c.supports ? '<span class="tag ok">supports claim</span>' : '<span class="tag bad">does not support</span>'}</td></tr>`).join("")}</table>`;
    else h += `<p class="hint">No citation attached to this claim.</p>`;
    if (k.clauses?.length) h += `<h3>Clauses</h3><table>${k.clauses.map((c) => `<tr><td>${esc(c.text)}</td><td class="num">${fmt(c.entailment)}</td><td>${c.supported ? '<span class="tag ok">supported</span>' : '<span class="tag">not supported</span>'}</td></tr>`).join("")}</table>`;
    if (k.notes?.length) h += `<p class="hint">${esc(k.notes.join(" · "))}</p>`;
    h += `<p class="hint">Entailment shows the passage supports the claim; it does not prove the generator relied on it.</p>`;
  }
  $("#drawer-body").innerHTML = h;
  $("#drawer").hidden = false;
});
$("#drawer-close").addEventListener("click", () => ($("#drawer").hidden = true));
$("#drawer").addEventListener("click", (e) => { if (e.target.id === "drawer") $("#drawer").hidden = true; });
document.addEventListener("keydown", (e) => { if (e.key === "Escape") $("#drawer").hidden = true; });

// ------------------------------------------------------------------ compare
const CMP = {};
$("#cmp-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = $("#cmp-q").value.trim() || $("#question").value.trim();
  if (!q) return;
  $("#cmp-q").value = q;
  const gen = $("#generator").value;
  const conds = (gen === "extractive" ? [] : ["llm_only"]).concat(["vanilla", "trust", "sentinel"]);
  $("#cmp-result").innerHTML = conds.map((c) => `<div class="card" id="cmp-${c}"><h2>${esc(STATUS.conditions[c])}</h2><span class="spinner"></span></div>`).join("");
  for (const c of conds) {
    try {
      const rec = await api("/api/ask", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q, condition: c, generator: gen, model: $("#model").disabled ? null : $("#model").value }) });
      CMP[c] = rec;
      const fact = (rec.claims || []).filter((k) => k.kind === "factual");
      const sup = fact.filter((k) => k.relation === "supported").length;
      $("#cmp-" + c).outerHTML = `<div class="card" data-rec="${c}"><h2>${esc(rec.condition_label)}</h2>${decisionBadge(rec.decision)}${renderAnswer(rec)}
        <p class="meta"><span>${sup}/${fact.length} claims supported</span><span>${rec.counters.retrieval_calls} retrieval round(s)</span><span>${fmt(rec.timing.total_s)} s</span></p></div>`;
    } catch (err) {
      $("#cmp-" + c).innerHTML += `<p class="err">${esc(err.message)}</p>`;
    }
  }
});

// ------------------------------------------------------------------ corpus
async function loadDocs() {
  const { documents } = await api("/api/documents");
  $("#docs").innerHTML = documents.length ? `<div class="scroll"><table><tr><th>Document</th><th>Reliability</th><th>Published</th><th>Source group</th><th class="num">Chunks</th><th>Version / hash</th><th></th></tr>` +
    documents.map((d) => `<tr><td><b>${esc(d.title)}</b><div class="mono">${esc(d.doc_id)} · ${esc(d.filename)}</div><div class="hint">${esc(d.extraction)}</div>${(d.warnings || []).map((w) => `<span class="tag warn">${esc(w)}</span>`).join("")}${d.synthetic ? '<span class="tag">synthetic</span>' : ""}</td>
      <td><span class="tag ${d.reliability === "official" ? "ok" : ["unverified", "unknown", "blog"].includes(d.reliability) ? "bad" : ""}">${esc(d.reliability)}</span></td>
      <td>${esc(d.published || "missing")}</td><td class="mono">${esc(d.source_group)}</td><td class="num">${d.n_chunks}</td>
      <td class="mono">v${d.version} · ${esc(d.content_hash.slice(0, 10))}</td>
      <td><button class="danger" data-del="${esc(d.doc_id)}">Remove</button></td></tr>`).join("") + `</table></div>` : `<p class="hint">The corpus is empty. Upload a document or reload the demo corpus.</p>`;
}
$("#docs").addEventListener("click", async (e) => {
  const id = e.target.dataset.del;
  if (!id || !confirm(`Remove ${id} from the index?`)) return;
  await api("/api/documents/" + encodeURIComponent(id), { method: "DELETE" });
  loadDocs(); loadStatus();
});
$("#upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = $("#file").files[0];
  if (!f) return;
  const fd = new FormData();
  fd.append("file", f);
  fd.append("reliability", $("#u-rel").value);
  fd.append("published", $("#u-date").value.trim());
  fd.append("source_group", $("#u-group").value.trim());
  fd.append("title", $("#u-title").value.trim());
  $("#upload-msg").innerHTML = `<span class="spinner"></span>Parsing, chunking and embedding…`;
  try {
    const r = await api("/api/documents", { method: "POST", body: fd });
    $("#upload-msg").innerHTML = `<p>${esc(r.doc_id)}: <b>${esc(r.status)}</b>, ${r.n_chunks} chunks (v${r.version}).${(r.warnings || []).map((w) => ` <span class="tag warn">${esc(w)}</span>`).join("")}</p>`;
    loadDocs(); loadStatus();
  } catch (err) {
    $("#upload-msg").innerHTML = `<p class="err">${esc(err.message)}</p>`;
  }
});
$("#reload-demo").addEventListener("click", async () => {
  $("#upload-msg").innerHTML = `<span class="spinner"></span>Re-indexing the demo corpus…`;
  await api("/api/corpus/demo", { method: "POST" });
  $("#upload-msg").innerHTML = "Demo corpus reloaded.";
  loadDocs(); loadStatus();
});
$("#clear-corpus").addEventListener("click", async () => {
  if (!confirm("Remove every document from the index?")) return;
  await api("/api/corpus/clear", { method: "POST" });
  loadDocs(); loadStatus();
});

// ------------------------------------------------------------------ eval
const METRICS = [
  ["accuracy", "Accuracy"], ["answer_rate", "Answer rate"], ["selective_accuracy", "Selective acc."],
  ["abstention_on_unanswerable", "Abstain | unanswerable"], ["abstention_on_answerable", "Abstain | answerable"],
  ["faithfulness", "Faithfulness"], ["partial_rate", "Partial"], ["unsupported_rate", "Unsupported"],
  ["citation_correctness", "Citation corr."], ["evidence_coverage", "Evidence cov."], ["facet_coverage", "Facet cov."],
  ["precision_at_k", "P@k"], ["recall_at_k", "R@k"], ["mrr", "MRR"],
  ["latency_mean_s", "Latency mean s"], ["latency_p90_s", "p90 s"], ["retrieval_calls_mean", "Rounds"],
  ["generator_calls_mean", "Gen calls"], ["generated_tokens_total", "Tokens"], ["errors", "Errors"],
];
async function loadEval() {
  try {
    renderEval(await api("/api/eval/latest"));
  } catch {
    $("#eval-result").innerHTML = `<div class="card hint">No evaluation has been run yet.</div>`;
  }
}
function renderEval(res) {
  const m = res.metrics;
  const conds = Object.keys(m);
  let h = `<div class="card"><h2>Results <span class="hint">${esc(res.config.generator)} / ${esc(res.config.generator_model)} · split ${esc(res.config.split)} · config ${esc(res.config.config_hash)} · ${res.seconds}s · ${esc(res.config.started)}</span></h2>
    <p class="hint">Synthetic 24-question set over a fictional corpus. N per condition = ${m[conds[0]].N}. Claim metrics count only answered questions; abstentions are reported through answer rate, never as perfect faithfulness.</p>
    <div class="scroll"><table><tr><th>Metric</th>${conds.map((c) => `<th class="num">${esc(m[c].label)}</th>`).join("")}</tr>` +
    METRICS.map(([k, n]) => `<tr><td>${n}</td>${conds.map((c) => `<td class="num">${fmt(m[c][k], k.includes("tokens") || k === "errors" ? 0 : 2)}</td>`).join("")}</tr>`).join("") +
    `</table></div></div>`;
  const cmp = res.vs_vanilla || {};
  if (Object.keys(cmp).length) {
    h += `<div class="card"><h2>Paired bootstrap vs vanilla RAG</h2><p class="hint">Difference (condition − vanilla) with 95% CI from 2,000 resamples of question IDs; claims stay with their question.</p><table><tr><th>Condition</th><th>Accuracy</th><th>Faithfulness</th><th>Unsupported rate</th></tr>` +
      Object.entries(cmp).map(([c, v]) => `<tr><td>${esc(m[c].label)}</td>${["accuracy", "faithfulness", "unsupported_rate"].map((k) => `<td class="num">${v[k] ? `${v[k].diff >= 0 ? "+" : ""}${fmt(v[k].diff)} [${fmt(v[k].ci95[0])}, ${fmt(v[k].ci95[1])}]` : "n/a"}</td>`).join("")}</tr>`).join("") + `</table></div>`;
  }
  const cats = res.categories;
  const catNames = [...new Set(Object.values(cats).flatMap((x) => Object.keys(x)))];
  h += `<div class="card"><h2>Correct answers by category</h2><div class="scroll"><table><tr><th>Category</th>${conds.map((c) => `<th class="num">${esc(c)}</th>`).join("")}</tr>` +
    catNames.map((cat) => `<tr><td>${esc(cat)}</td>${conds.map((c) => { const x = cats[c][cat]; return `<td class="num">${x ? `${x.correct}/${x.n}` : ""}</td>`; }).join("")}</tr>`).join("") + `</table></div></div>`;
  h += `<div class="card"><h2>Per question</h2><div class="scroll"><table><tr><th>ID</th>${conds.map((c) => `<th>${esc(c)}</th>`).join("")}</tr>` +
    Object.entries(res.per_question).map(([qid, row]) => `<tr><td class="mono">${qid}</td>${conds.map((c) => { const x = row[c]; return x ? `<td title="${esc(x.answer || x.reason || "")}"><span class="tag ${x.correct ? "ok" : "bad"}">${x.correct ? "✓" : "✗"} ${esc(x.decision)}</span></td>` : "<td></td>"; }).join("")}</tr>`).join("") +
    `</table></div><p class="hint">Hover a cell for the delivered answer or abstention reason. Saved under ${esc(res.run_dir)}.</p></div>`;
  $("#eval-result").innerHTML = h;
}
$("#eval-run").addEventListener("click", async () => {
  const gen = $("#eval-gen").value;
  const main = ["llm_only", "vanilla", "trust", "sentinel"];
  const conds = $("#eval-ablations").checked ? main.concat(["vanilla_iterative", "sentinel_no_trust", "sentinel_no_adaptive", "sentinel_no_feedback"]) : main;
  try {
    await api("/api/eval", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ generator: gen, conditions: conds, split: $("#eval-split").value }) });
  } catch (err) {
    $("#eval-progress").innerHTML = `<p class="err">${esc(err.message)}</p>`;
    return;
  }
  $("#eval-run").disabled = true;
  const poll = async () => {
    const s = await api("/api/eval/status");
    $("#eval-progress").innerHTML = s.running ? `<p><span class="spinner"></span>${s.done}/${s.total || "?"} · ${esc(s.current || "starting")}</p>` : s.error ? `<p class="err">${esc(s.error)}</p>` : "<p>Done.</p>";
    if (s.running) setTimeout(poll, 1000);
    else { $("#eval-run").disabled = false; if (!s.error) loadEval(); }
  };
  poll();
});

// ------------------------------------------------------------------ method
function renderMethod() {
  const s = STATUS.settings;
  $("#method").innerHTML = `<h2>What this prototype implements</h2>
  <p>Sentinel-RAG extends a vanilla RAG baseline with evidence assessment and control. The original question q₀ is fixed; a separate search query qₜ is refined between rounds. All retrieval rounds, before or after drafting, share one budget of <b>M = ${s.max_rounds}</b>.</p>
  <ol>
    <li><b>Ingestion.</b> PDF (PyMuPDF, page locators), DOCX (python-docx, section + paragraph locators, tables kept as header: value rows), TXT/MD. Whitespace is normalised without touching numbers, negation or punctuation. Each document records a content hash, version, extraction method and timestamp; a changed upload creates a new version.</li>
    <li><b>Chunking.</b> The article's 500/50 setting had unresolved units and <code>${esc(s.embed_model)}</code> truncates at 256 word pieces, so chunks are measured with that model's tokenizer: at most ${s.chunk_tokens} word pieces with ${s.overlap_tokens}-piece sentence overlap, never crossing a heading.</li>
    <li><b>Retrieval.</b> Cosine similarity over L2-normalised embeddings (exact inner product). Candidate pool ${s.candidate_pool}, final context k = ${s.top_k}.</li>
    <li><b>Trust.</b> T = ${s.w_relevance}R + ${s.w_reliability}S + ${s.w_consistency}C + ${s.w_freshness}F. S comes from declared source categories (unknown → ${s.missing_reliability}, recorded as missing). C is NLI agreement with topically matched passages from other source groups; duplicates and copies of one source do not count. Exceptions qualify rather than contradict a rule. F depends on the question's temporal need (current, historical or neutral).</li>
    <li><b>Sufficiency.</b> E = ${s.s_trust}T̄ + ${s.s_coverage}COV + ${s.s_agreement}AGR + ${s.s_independence}IND, bands high ≥ ${s.band_high}, medium ≥ ${s.band_medium}. Hard checks: an uncovered central facet forces “low”; an unresolved contradiction caps the band at “medium”. Facets are extracted by rules; a facet is covered only if its specific cue (e.g. “exception”) is present in the passage.</li>
    <li><b>Generation.</b> One provider-independent interface: a deterministic extractive generator (no LLM) or a hosted LLM through an OpenAI-compatible API (Groq by default). The prompt allows citing only the evidence IDs shown.</li>
    <li><b>Verification.</b> Answers are split into claims; each is checked against sentence, sentence-pair, chunk and joint spans with <code>${esc(s.nli_model)}</code>: supported (entailment ≥ ${s.entail_threshold} and every number present in the span), partially supported (some clause supported), contradicted (contradiction ≥ ${s.contradict_threshold}) or unsupported.</li>
    <li><b>Decision.</b> Unsupported claims trigger targeted re-retrieval while rounds remain. At the limit only the supported scope is regenerated and verified again; otherwise the system abstains. Contradicted or unsupported claims are never shown as answers.</li>
  </ol>
  <h2>Honest limits</h2>
  <ul>
    <li>Weights, thresholds and coefficients are the article's provisional settings or documented engineering choices, not calibrated values.</li>
    <li>The evaluation set is small and synthetic. Results show the mechanisms work as designed on controlled cases; they do not establish general superiority.</li>
    <li>NLI entailment shows a passage supports a claim; it does not prove the generator relied on it. The verifier can make mistakes, especially with numbers, tables and long spans.</li>
    <li>Source reliability is only as good as the declared metadata.</li>
  </ul>`;
}

// ------------------------------------------------------------------ boot
(async () => {
  await loadStatus();
  if (STATUS) fillSelectors();
})();
