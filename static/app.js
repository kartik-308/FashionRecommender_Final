/* ── Fashion Finder — frontend logic ─────────────────────────────────────── */

const $ = id => document.getElementById(id);

// ── State ─────────────────────────────────────────────────────────────────────
let currentMode     = "text";
let acceptedHistory = [];
let rejectedHistory = [];
let likedHistory    = [];
let droppedFile     = null;

// ── DOM refs ──────────────────────────────────────────────────────────────────
const statusBadge   = $("status-badge");
const btnSearch     = $("btn-search");
const btnReset      = $("btn-reset");
const searchForm    = $("search-form");
const textInput     = $("text-input");
const imageInput    = $("image-input");
const dropZone      = $("drop-zone");
const dropBody      = $("drop-body");
const previewImg    = $("preview-img");
const weightSlider  = $("weight-slider");
const twLabel       = $("tw-label");
const iwLabel       = $("iw-label");
const cardsGrid     = $("cards-grid");
const emptyState    = $("empty-state");
const queryBar      = $("query-bar");
const queryBarText  = $("query-bar-text");
const queryBarMode  = $("query-bar-mode");
const queryThumb    = $("query-thumb");
const historySection        = $("history-section");
const historyStripAccepted  = $("history-strip-accepted");
const historyStripRejected  = $("history-strip-rejected");
const historyStripLiked     = $("history-strip-liked");
const acceptedCount         = $("accepted-count");
const rejectedCount         = $("rejected-count");
const likedCount            = $("liked-count");
const toast                 = $("toast");
const driftBanner           = $("drift-banner");
const driftScore            = $("drift-score");

// ── Poll model status ─────────────────────────────────────────────────────────
const pageLoader    = $("page-loader");
const loaderMsg     = $("loader-msg");
const loaderBar     = $("loader-bar");

// Animate loader steps while waiting
let _stepIndex = 0;
const _steps   = ["step-clip", "step-embed", "step-faiss", "step-ready"];
const _stepMsgs = ["Loading CLIP model…", "Loading embeddings…", "Building FAISS index…", "Almost ready…"];
const _stepPcts = [20, 50, 80, 95];

function advanceLoaderStep() {
  if (_stepIndex > 0) {
    const prev = $(_steps[_stepIndex - 1]);
    if (prev) { prev.classList.remove("loader-step--active"); prev.classList.add("loader-step--done"); }
  }
  if (_stepIndex < _steps.length) {
    const cur = $(_steps[_stepIndex]);
    if (cur) cur.classList.add("loader-step--active");
    loaderBar.style.width = _stepPcts[_stepIndex] + "%";
    loaderMsg.textContent = _stepMsgs[_stepIndex];
    _stepIndex++;
  }
}

// Cycle through steps every 4s while loading
advanceLoaderStep();
const _stepTimer = setInterval(() => {
  if (_stepIndex < _steps.length) advanceLoaderStep();
  else clearInterval(_stepTimer);
}, 4000);

async function pollStatus() {
  try {
    const res  = await fetch("/api/status");
    const data = await res.json();

    if (data.ready) {
      clearInterval(_stepTimer);
      _steps.forEach(id => {
        const el = $(id);
        if (el) { el.classList.remove("loader-step--active"); el.classList.add("loader-step--done"); }
      });
      if (loaderBar) loaderBar.style.width = "100%";
      if (loaderMsg) loaderMsg.textContent = `✓ ${(data.total_images || 0).toLocaleString()} images ready`;

      statusBadge.className   = "badge badge--ready";
      statusBadge.textContent = `Ready · ${(data.total_images || 0).toLocaleString()} images`;
      btnSearch.disabled = false;

      setTimeout(() => { if (pageLoader) pageLoader.classList.add("hidden"); }, 1200);
      return;
    }

    if (data.error) {
      clearInterval(_stepTimer);
      statusBadge.className   = "badge badge--error";
      statusBadge.textContent = "Error";
      if (loaderMsg) loaderMsg.textContent = "⚠️ " + data.error;
      return;
    }
  } catch (_) {}
  setTimeout(pollStatus, 2500);
}

// ── Mode tabs ─────────────────────────────────────────────────────────────────
document.querySelectorAll(".tab").forEach(tab => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(t => t.classList.remove("tab--active"));
    tab.classList.add("tab--active");
    currentMode = tab.dataset.mode;
    applyMode();
  });
});

function applyMode() {
  const fieldText   = $("field-text");
  const fieldImage  = $("field-image");
  const fieldWeight = $("field-weight");

  fieldText.classList.toggle("hidden",   currentMode === "image");
  fieldImage.classList.toggle("hidden",  currentMode === "text");
  fieldWeight.classList.toggle("hidden", currentMode !== "multimodal");
}

// ── Weight slider ─────────────────────────────────────────────────────────────
weightSlider.addEventListener("input", () => {
  const tw = parseFloat(weightSlider.value);
  twLabel.textContent = Math.round(tw * 100) + "%";
  iwLabel.textContent = Math.round((1 - tw) * 100) + "%";
});

// ── Drop zone ─────────────────────────────────────────────────────────────────
dropZone.addEventListener("dragover", e => {
  e.preventDefault();
  dropZone.classList.add("drop-zone--over");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drop-zone--over"));
dropZone.addEventListener("drop", e => {
  e.preventDefault();
  dropZone.classList.remove("drop-zone--over");
  const file = e.dataTransfer.files[0];
  if (file) {
    droppedFile = file;          // store for FormData
    setImagePreview(file);
  }
});
imageInput.addEventListener("change", () => {
  if (imageInput.files[0]) {
    droppedFile = null;          // input has it, no need for droppedFile
    setImagePreview(imageInput.files[0]);
  }
});

function setImagePreview(file) {
  const reader = new FileReader();
  reader.onload = ev => {
    previewImg.src = ev.target.result;
    previewImg.classList.remove("hidden");
    dropBody.classList.add("hidden");
    dropZone.classList.add("drop-zone--has-file");
    $("btn-clear-img").classList.remove("hidden");
  };
  reader.readAsDataURL(file);
}

function clearImageState() {
  droppedFile = null;
  imageInput.value = "";
  previewImg.src = "";
  previewImg.classList.add("hidden");
  dropBody.classList.remove("hidden");
  dropZone.classList.remove("drop-zone--has-file");
  $("btn-clear-img").classList.add("hidden");
}

$("btn-clear-img").addEventListener("click", e => {
  e.stopPropagation();
  clearImageState();
});

// ── Example chips ─────────────────────────────────────────────────────────────
document.querySelectorAll(".chip").forEach(chip => {
  chip.addEventListener("click", () => {
    textInput.value = chip.textContent;
    // Switch to text mode if not already
    if (currentMode === "image") {
      document.querySelector('[data-mode="text"]').click();
    }
    textInput.focus();
  });
});

// ── Reset ─────────────────────────────────────────────────────────────────────
btnReset.addEventListener("click", async () => {
  await fetch("/api/reset", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });
  acceptedHistory = [];
  rejectedHistory = [];
  likedHistory    = [];
  historyStripAccepted.innerHTML = "";
  historyStripRejected.innerHTML = "";
  historyStripLiked.innerHTML    = "";
  acceptedCount.textContent = "0";
  rejectedCount.textContent = "0";
  likedCount.textContent    = "0";
  historySection.classList.add("hidden");
  driftBanner.classList.add("hidden");
  showToast("Preference memory cleared");
});

// ── Search form ───────────────────────────────────────────────────────────────
searchForm.addEventListener("submit", async e => {
  e.preventDefault();
  await doSearch();
});

async function doSearch() {
  const formData = new FormData();

  // Resolve text
  const text = (currentMode !== "image") ? (textInput.value.trim() || null) : null;

  // Resolve image — prefer droppedFile, fall back to <input>
  const imageFile = droppedFile || (imageInput.files[0] || null);
  const hasImage  = !!imageFile;

  // Validate
  if (currentMode === "text"  && !text)     { showToast("Please enter a text query."); return; }
  if (currentMode === "image" && !hasImage) { showToast("Please upload an image."); return; }
  if (currentMode === "multimodal" && !text && !hasImage) {
    showToast("Provide text, image, or both."); return;
  }

  if (text)      formData.append("text", text);
  if (hasImage)  formData.append("image", imageFile);
  formData.append("text_weight", weightSlider.value);

  setLoading(true);
  showSkeletons();

  try {
    const res  = await fetch("/api/search", { method: "POST", body: formData });
    const data = await res.json();

    if (!res.ok) { showToast("⚠️ " + (data.error || "Search failed")); setLoading(false); return; }

    renderResults(data, text);
  } catch (err) {
    showToast("⚠️ Network error: " + err.message);
  }

  setLoading(false);
}

// ── Render results ────────────────────────────────────────────────────────────
function renderResults(data, queryText) {
  cardsGrid.innerHTML = "";

  // Query bar
  queryBar.classList.remove("hidden");
  queryBarText.textContent = queryText || "(image query)";
  queryBarMode.textContent = data.mode;

  if (data.query_img_b64) {
    queryThumb.src = data.query_img_b64;
    queryThumb.classList.remove("hidden");
  } else {
    queryThumb.classList.add("hidden");
  }

  if (!data.results || data.results.length === 0) {
    cardsGrid.appendChild(emptyState);
    emptyState.classList.remove("hidden");
    return;
  }

  data.results.forEach((item, i) => {
    const card = buildCard(item, i + 1);
    cardsGrid.appendChild(card);

    // Add to shown history
    if (item.image_b64) {
      acceptedHistory.push({ b64: item.image_b64, filename: item.filename });
    }
  });

  // Handle drift notification
  if (data.drift_detected) {
    driftBanner.classList.remove("hidden");
    driftScore.textContent = `Drift: ${(data.drift_score * 100).toFixed(0)}%`;
  } else {
    driftBanner.classList.add("hidden");
  }

  updateHistory();
}

function buildCard(item, rank) {
  const score = item.final_score;
  const scoreClass = score > 0.35 ? "" : score > 0.2 ? "card__score--mid" : "card__score--low";

  const card = document.createElement("div");
  card.className = "card";
  card.dataset.filename = item.filename;

  card.innerHTML = `
    <div class="card__rank">${rank}</div>
    <div class="card__img-wrap">
      ${item.image_b64
        ? `<img class="card__img" src="${item.image_b64}" alt="${item.filename}" loading="lazy" />`
        : `<div class="card__img skeleton" style="width:100%;height:100%"></div>`
      }
    </div>
    <div class="card__body">
      <span class="card__filename" title="${item.filename}">${item.filename}</span>
      ${item.title ? `<span class="card__title" title="${item.title}">${item.title}</span>` : ""}
      <div style="display:flex;align-items:center;gap:.4rem;flex-wrap:wrap;margin-top:.1rem">
        <span class="card__score ${scoreClass}">${(score * 100).toFixed(1)}%</span>
        ${item.price ? `<span class="card__price">${item.price}</span>` : ""}
        ${item.source ? `<span class="card__source card__source--${item.dataset}">${item.source}</span>` : ""}
        ${item.category && item.category !== "unknown" ? `<span class="card__category">${item.category}</span>` : ""}
      </div>
      <div class="score-bars">
        ${scoreBar("Query",      item.sim_query,  "q",   "Similarity to your search query")}
        ${scoreBar("Preference", item.sim_pref,   "p",   "Similarity to your taste profile")}
        ${scoreBar("Redundancy", item.redundancy, "r",   "How similar to already-shown items")}
        ${scoreBar("Popularity", item.popularity, "pop", "Category popularity in dataset")}
      </div>
      ${item.blip2_caption ? `<div class="card__blip2-caption"><span class="card__blip2-tag">AI</span> ${item.blip2_caption}</div>` : ""}
      ${item.explanation ? `<div class="card__explanation">${item.explanation}</div>` : ""}
      <div class="card__actions">
        <button class="card__btn card__btn--accept" data-filename="${item.filename}">
          &#10003; Accept
        </button>
        <button class="card__btn card__btn--reject" data-filename="${item.filename}">
          &#10005; Reject
        </button>
      </div>
    </div>
  `;

  card.querySelector(".card__btn--accept").addEventListener("click", () => acceptItem(item.filename, item.image_b64, card));
  card.querySelector(".card__btn--reject").addEventListener("click", () => rejectItem(item.filename, card));
  return card;
}

function scoreBar(label, value, type, title) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return `
    <div class="score-bar" title="${title}: ${pct.toFixed(1)}%">
      <span class="score-bar__label">${label}</span>
      <div class="score-bar__track">
        <div class="score-bar__fill score-bar__fill--${type}" style="width:${pct}%"></div>
      </div>
      <span class="score-bar__val">${pct.toFixed(0)}%</span>
    </div>`;
}

// ── Accept ────────────────────────────────────────────────────────────────────
async function acceptItem(filename, b64, cardEl) {
  cardEl.classList.add("card--accepted");
  // Disable both buttons
  cardEl.querySelectorAll(".card__btn").forEach(btn => btn.disabled = true);

  const res = await fetch("/api/accept", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filename }),
  });
  const data = await res.json();

  // Move to liked history
  if (b64) {
    likedHistory.push({ b64, filename });
  }
  // Remove from shown (accepted) history
  const idx = acceptedHistory.findIndex(h => h.filename === filename);
  if (idx !== -1) acceptedHistory.splice(idx, 1);

  updateHistory();

  // Handle drift
  if (data.drift_detected) {
    driftBanner.classList.remove("hidden");
    driftScore.textContent = `Drift: ${(data.drift_score * 100).toFixed(0)}%`;
  }

  showToast(`Liked: ${filename}`);
}

// ── Reject ────────────────────────────────────────────────────────────────────
async function rejectItem(filename, cardEl) {
  cardEl.classList.add("card--rejected");
  // Disable both buttons
  cardEl.querySelectorAll(".card__btn").forEach(btn => btn.disabled = true);

  const res = await fetch("/api/reject", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filename }),
  });
  const data = await res.json();

  // Move from accepted -> rejected in history
  const idx = acceptedHistory.findIndex(h => h.filename === filename);
  if (idx !== -1) {
    const [item] = acceptedHistory.splice(idx, 1);
    rejectedHistory.push(item);
  }

  updateHistory();

  // Handle drift
  if (data.drift_detected) {
    driftBanner.classList.remove("hidden");
    driftScore.textContent = `Drift: ${(data.drift_score * 100).toFixed(0)}%`;
  }

  showToast(`Rejected: ${filename}`);
}

// ── History ───────────────────────────────────────────────────────────────────
function makeThumb(item, type) {
  const wrap = document.createElement("div");
  wrap.className = `history-thumb history-thumb--${type}`;
  wrap.title = item.filename;
  const img = document.createElement("img");
  img.src = item.b64;
  img.alt = item.filename;
  wrap.appendChild(img);
  return wrap;
}

function updateHistory() {
  const hasAny = acceptedHistory.length > 0 || rejectedHistory.length > 0 || likedHistory.length > 0;
  historySection.classList.toggle("hidden", !hasAny);

  // Liked strip
  historyStripLiked.innerHTML = "";
  likedHistory.slice(-20).forEach(item => {
    historyStripLiked.appendChild(makeThumb(item, "liked"));
  });
  historyStripLiked.scrollLeft = historyStripLiked.scrollWidth;
  likedCount.textContent = likedHistory.length;

  // Shown strip
  historyStripAccepted.innerHTML = "";
  acceptedHistory.slice(-20).forEach(item => {
    historyStripAccepted.appendChild(makeThumb(item, "accepted"));
  });
  historyStripAccepted.scrollLeft = historyStripAccepted.scrollWidth;
  acceptedCount.textContent = acceptedHistory.length;

  // Rejected strip
  historyStripRejected.innerHTML = "";
  rejectedHistory.slice(-20).forEach(item => {
    historyStripRejected.appendChild(makeThumb(item, "rejected"));
  });
  historyStripRejected.scrollLeft = historyStripRejected.scrollWidth;
  rejectedCount.textContent = rejectedHistory.length;

  // Hide rows if empty
  $("history-row-liked").style.display    = likedHistory.length ? "" : "none";
  $("history-row-accepted").style.display = acceptedHistory.length ? "" : "none";
  $("history-row-rejected").style.display = rejectedHistory.length ? "" : "none";
}

// ── Skeletons ─────────────────────────────────────────────────────────────────
function showSkeletons() {
  cardsGrid.innerHTML = "";
  for (let i = 0; i < 5; i++) {
    const card = document.createElement("div");
    card.className = "card card--skeleton";
    card.innerHTML = `
      <div class="card__img-wrap skeleton"></div>
      <div class="card__body">
        <div class="skel-line skeleton" style="width:70%"></div>
        <div class="skel-line skeleton" style="width:40%"></div>
        <div class="skel-line skeleton" style="width:90%"></div>
      </div>`;
    cardsGrid.appendChild(card);
  }
}

// ── Loading state ─────────────────────────────────────────────────────────────
function setLoading(on) {
  btnSearch.disabled = on || statusBadge.classList.contains("badge--loading");
  btnSearch.querySelector(".btn-text").classList.toggle("hidden", on);
  btnSearch.querySelector(".btn-spinner").classList.toggle("hidden", !on);
}

// ── Toast ─────────────────────────────────────────────────────────────────────
let toastTimer;
function showToast(msg, duration = 2800) {
  toast.textContent = msg;
  toast.classList.remove("hidden");
  toast.classList.add("toast--show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    toast.classList.remove("toast--show");
    setTimeout(() => toast.classList.add("hidden"), 300);
  }, duration);
}

// ── Init ──────────────────────────────────────────────────────────────────────
// Populate header from server-injected user context
const user = window.__USER__ || {};
const headerUsername  = $("header-username");
const headerAdminLink = $("header-admin-link");

if (headerUsername)  headerUsername.textContent = user.username || "";
if (headerAdminLink && user.role === "admin") headerAdminLink.style.display = "";

// Sign out — direct navigation
const btnSignout = $("btn-signout");
if (btnSignout) {
  btnSignout.addEventListener("click", e => {
    e.preventDefault();
    window.location.href = "/logout";
  });
}

pollStatus();
applyMode();
