"use strict";

const assetSel = document.getElementById("asset");
const hoursSel = document.getElementById("hours");
const goBtn = document.getElementById("go");
const priceBox = document.getElementById("price");
const evBox = document.getElementById("evidence");
const datelineDate = document.getElementById("dateline-date");

// --- helpers ---------------------------------------------------------------
const fmtTs = (iso) =>
  new Date(iso).toISOString().slice(0, 16).replace("T", " ") + " UTC";
const fmtUsd = (v) =>
  "$" + Number(v).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const escapeHtml = (s) =>
  s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function detailText(d, status) {
  let msg = d && d.detail;
  if (Array.isArray(msg)) msg = msg.map((x) => x.msg || JSON.stringify(x)).join("; ");
  return msg || "HTTP " + status;
}

function currentQuestion() {
  const label = hoursSel.options[hoursSel.selectedIndex].text;
  return `What happened to ${assetSel.value} within the last ${label}?`;
}

// --- masthead date ---------------------------------------------------------
datelineDate.textContent = new Date().toLocaleDateString("en-US", {
  weekday: "long", year: "numeric", month: "long", day: "numeric",
});

// --- assets dropdown -------------------------------------------------------
async function loadAssets() {
  try {
    const r = await fetch("/api/assets");
    const d = await r.json();
    assetSel.innerHTML = "";
    for (const a of d.assets) {
      const o = document.createElement("option");
      o.value = a;
      o.textContent = a;
      if (a === d.default) o.selected = true;
      assetSel.appendChild(o);
    }
  } catch (e) {
    assetSel.innerHTML = '<option value="BTC" selected>BTC</option>';
  }
}

// --- renderers -------------------------------------------------------------
function priceHtml(d) {
  const arrow = d.direction === "up" ? "▲" : d.direction === "down" ? "▼" : "▬";
  const durH = ((new Date(d.move_end) - new Date(d.move_start)) / 3.6e6).toFixed(1);
  const sign = d.pct_change >= 0 ? "+" : "";
  priceBox.className = "report " + d.direction;
  return `
    <p class="kicker">The Move</p>
    <div class="move__head">
      <span class="move__ticker">${d.asset}</span>
      <span class="move__pct">${arrow} ${sign}${d.pct_change.toFixed(2)}%</span>
      <span class="move__dir">${d.direction}</span>
    </div>
    <dl class="ledger">
      <div><dt>Window</dt><dd>${fmtTs(d.window_start)} &rarr; ${fmtTs(d.window_end)} (${d.hours}h &middot; ${d.granularity})</dd></div>
      <div><dt>Price</dt><dd>${fmtUsd(d.price_start)} &rarr; ${fmtUsd(d.price_end)}</dd></div>
      <div><dt>Max drawdown</dt><dd>&minus;${d.max_drawdown_pct.toFixed(2)}%</dd></div>
      <div><dt>Steepest move</dt><dd>${fmtTs(d.move_start)} &rarr; ${fmtTs(d.move_end)} (${durH}h)</dd></div>
    </dl>`;
}

function evidenceHtml(d) {
  if (!d.chunks || d.chunks.length === 0) {
    const note = (d.notes && d.notes[0]) || "No record found in this window.";
    return `<p class="kicker">The Record</p><p class="placeholder">${escapeHtml(note)}</p>`;
  }
  const items = d.chunks
    .map(
      (c) => `
      <div class="dispatch">
        <div class="dispatch__meta">
          <span class="dispatch__source">${escapeHtml(c.source_name)}</span>
          <span class="dispatch__date">${fmtTs(c.published_at)}</span>
          <span class="dispatch__score" title="relevance (semantic ${c.semantic_score} + BM25 ${c.bm25_score})">rel. ${c.score.toFixed(3)}</span>
        </div>
        <p class="dispatch__body">${escapeHtml(c.snippet)}</p>
        <a class="dispatch__link" href="${encodeURI(c.url)}" target="_blank" rel="noopener">${escapeHtml(c.url)}</a>
      </div>`
    )
    .join("");
  return `<p class="kicker">The Record — ${d.chunks.length} source(s), by relevance</p>${items}`;
}

// --- fetchers --------------------------------------------------------------
async function loadPrice(asset, hours) {
  try {
    const r = await fetch(`/api/price-event?asset=${encodeURIComponent(asset)}&hours=${encodeURIComponent(hours)}`);
    const d = await r.json();
    priceBox.innerHTML = r.ok
      ? priceHtml(d)
      : `<p class="kicker">The Move</p><p class="error">${escapeHtml(detailText(d, r.status))}</p>`;
  } catch (e) {
    priceBox.innerHTML = `<p class="kicker">The Move</p><p class="error">Network error: ${escapeHtml(e.message)}</p>`;
  }
}

async function loadEvidence(question, asset, hours) {
  try {
    const url = `/api/evidence?question=${encodeURIComponent(question)}&asset=${encodeURIComponent(asset)}&hours=${encodeURIComponent(hours)}`;
    const r = await fetch(url);
    const d = await r.json();
    evBox.innerHTML = r.ok
      ? evidenceHtml(d)
      : `<p class="kicker">The Record</p><p class="error">${escapeHtml(detailText(d, r.status))}</p>`;
  } catch (e) {
    evBox.innerHTML = `<p class="kicker">The Record</p><p class="error">Network error: ${escapeHtml(e.message)}</p>`;
  }
}

// --- orchestration ---------------------------------------------------------
async function onAsk() {
  const asset = assetSel.value;
  const hours = hoursSel.value;
  const question = currentQuestion();

  goBtn.classList.add("loading");
  goBtn.disabled = true;
  priceBox.hidden = false;
  priceBox.className = "report";
  priceBox.innerHTML = '<p class="kicker">The Move</p><p class="placeholder">Confirming the move&hellip;</p>';
  evBox.hidden = false;
  evBox.innerHTML = '<p class="kicker">The Record</p><p class="placeholder">Gathering the record&hellip;</p>';

  await Promise.allSettled([loadPrice(asset, hours), loadEvidence(question, asset, hours)]);

  goBtn.classList.remove("loading");
  goBtn.disabled = false;
}

goBtn.addEventListener("click", onAsk);
loadAssets();
