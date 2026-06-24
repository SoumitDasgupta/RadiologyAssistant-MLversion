/* =========================================================
   AI Radiology Assistant — app.js
   Frontend → FastAPI → Gemini AI → SQLite
   ========================================================= */

const API = "http://127.0.0.1:8000";

let selectedFile = null;
let zoom = 100;
let currentTool = "select";
let inverted = false;
let currentPresetFilter = "";   // tracks the active preset's filter string

/* ── Login ──────────────────────────────────────────────── */
function setRole(btn, role) {
  document.querySelectorAll(".role-tab").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
}

function togglePw() {
  const pw = document.getElementById("loginPassword");
  pw.type = pw.type === "password" ? "text" : "password";
}

function doLogin() {
  document.getElementById("loginPage").style.display  = "none";
  document.getElementById("appPage").style.display    = "flex";
}

function logout() {
  document.getElementById("appPage").style.display   = "none";
  document.getElementById("loginPage").style.display = "flex";
}

/* ── Image loading ─────────────────────────────────────── */
document.addEventListener("DOMContentLoaded", () => {
  const input = document.getElementById("imageInput");
  if (input) input.addEventListener("change", onFileSelected);

  const dz = document.getElementById("dropZone");
  if (dz) {
    dz.addEventListener("dragover", e => { e.preventDefault(); dz.classList.add("dragover"); });
    dz.addEventListener("dragleave", () => dz.classList.remove("dragover"));
    dz.addEventListener("drop", e => {
      e.preventDefault();
      dz.classList.remove("dragover");
      if (e.dataTransfer.files[0]) loadFile(e.dataTransfer.files[0]);
    });
  }
});

function onFileSelected() {
  const input = document.getElementById("imageInput");
  if (input.files[0]) loadFile(input.files[0]);
}

function loadFile(file) {
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    const preview = document.getElementById("preview");
    preview.src = e.target.result;
    preview.onload = () => {
      document.getElementById("dropZone").style.display       = "none";
      document.getElementById("imgContainer").style.display   = "flex";
      document.getElementById("emptyState").style.display     = "none";

      const mb = (file.size / 1024 / 1024).toFixed(1);
      document.getElementById("fileInfo").textContent  = "FILE — " + file.name;
      document.getElementById("sizeInfo").textContent  = "SIZE — " + mb + " MB";
      document.getElementById("dimsInfo").textContent  = "DIMS — " + preview.naturalWidth + " × " + preview.naturalHeight;
      document.getElementById("aiStatusBox").textContent = "Image loaded. Click 'Analyze with Gemini AI' to begin.";
    };
  };
  reader.readAsDataURL(file);
  addRecentItem(file.name);
}

function addRecentItem(name) {
  const list = document.getElementById("recentList");
  const item = document.createElement("div");
  item.className = "recent-item";
  item.innerHTML = `
    <span class="dot teal"></span>
    <div>
      <div class="recent-name">${name} <span class="recent-time">just now</span></div>
      <div class="recent-desc">Pending analysis…</div>
    </div>`;
  list.insertBefore(item, list.firstChild);
}

/* ── Sliders ───────────────────────────────────────────── */
function applyFilters() {
  const b = +document.getElementById("brightness").value;
  const c = +document.getElementById("contrast").value;
  const s = +document.getElementById("sharpness").value;
  document.getElementById("brightnessVal").textContent = b;
  document.getElementById("contrastVal").textContent   = c;
  document.getElementById("sharpnessVal").textContent  = s;
  const preview = document.getElementById("preview");
  if (preview) {
    // Build slider filter on top of the active preset filter
    const sliderPart = `brightness(${1 + b/100}) contrast(${1 + c/100}) ${s > 0 ? `saturate(${1 + s*0.01})` : ""}`;
    preview.style.filter = (currentPresetFilter ? currentPresetFilter + " " : "") + sliderPart;
  }
}

function resetSliders() {
  ["brightness","contrast","sharpness"].forEach(id => {
    document.getElementById(id).value = 0;
    const el = document.getElementById(id + "Val");
    if (el) el.textContent = "0";
  });
  // Restore to current preset (don't clear the preset)
  const preview = document.getElementById("preview");
  if (preview) preview.style.filter = currentPresetFilter;
  inverted = false;
}

/* ── Window presets ────────────────────────────────────── */
const PRESETS = {
  default:    "",
  invert:     "invert(1)",
  bone:       "grayscale(1) contrast(1.5) brightness(1.1)",
  lung:       "grayscale(1) contrast(2) brightness(0.7)",
  softtissue: "grayscale(1) contrast(1.2) brightness(1.2)",
  vascular:   "grayscale(1) contrast(2.2) brightness(0.85)",
};

function setPreset(btn, name) {
  document.querySelectorAll(".preset").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");

  // Store the preset filter separately so sliders can layer on top
  currentPresetFilter = PRESETS[name] || "";

  const preview = document.getElementById("preview");
  if (preview) preview.style.filter = currentPresetFilter;

  document.getElementById("presetInfo").textContent = "PRESET — " + btn.textContent;

  // Reset sliders to 0 without wiping the preset
  ["brightness","contrast","sharpness"].forEach(id => {
    document.getElementById(id).value = 0;
    const el = document.getElementById(id + "Val");
    if (el) el.textContent = "0";
  });
  inverted = false;
}

/* ── Tabs ──────────────────────────────────────────────── */
function setTab(btn, tab) {
  // Toggle active button
  document.querySelectorAll(".vtab").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");

  // Show the matching panel, hide the others
  const panels = { viewer: "viewerTab", chat: "chatTab", dashboard: "dashboardTab" };
  Object.entries(panels).forEach(([key, id]) => {
    const el = document.getElementById(id);
    if (el) el.style.display = key === tab ? (key === "viewer" ? "block" : "flex") : "none";
  });

  // When switching to dashboard, load latest data
  if (tab === "dashboard") loadDashboard();
}

/* ── Tools ─────────────────────────────────────────────── */
function setTool(btn, tool) {
  document.querySelectorAll(".tool-btn").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  currentTool = tool;
}

/* ── Zoom ──────────────────────────────────────────────── */
function zoomIn()  { applyZoom(zoom + 10); }
function zoomOut() { applyZoom(zoom - 10); }
function fitView() { applyZoom(100); }

function applyZoom(z) {
  zoom = Math.max(20, Math.min(400, z));
  const preview = document.getElementById("preview");
  if (preview) { preview.style.transform = `scale(${zoom/100})`; preview.style.transformOrigin = "center center"; }
  document.getElementById("zoomLevel").textContent = zoom + "%";
  document.getElementById("zoomInfo").textContent  = "ZOOM — " + zoom + "%";
}

/* ── Nav actions ───────────────────────────────────────── */
function resetView() {
  currentPresetFilter = "";
  resetSliders();
  applyZoom(100);
  clearAnnotations();
  document.querySelectorAll(".preset").forEach(b => b.classList.remove("active"));
  document.querySelector(".preset")?.classList.add("active");
  document.getElementById("presetInfo").textContent = "PRESET — Default";
}

function toggleHalf() {
  const p = document.getElementById("preview");
  if (!p) return;
  p.style.clipPath = p.style.clipPath ? "" : "inset(0 50% 0 0)";
}

function toggleInvert() {
  inverted = !inverted;
  const p = document.getElementById("preview");
  if (!p) return;
  p.style.filter = inverted ? (p.style.filter || "") + " invert(1)" : (p.style.filter || "").replace(/\s?invert\(1\)/g, "");
}

function togglePoints() {
  const layer = document.getElementById("annotations");
  if (layer) layer.style.display = layer.style.display === "none" ? "" : "none";
}

function clearAnnotations() {
  const layer = document.getElementById("annotations");
  if (layer) layer.innerHTML = "";
}

/* ── Theme ─────────────────────────────────────────────── */
function toggleTheme() {
  document.body.classList.toggle("light");
  document.querySelector(".theme-btn").textContent = document.body.classList.contains("light") ? "🌙 Dark" : "☀ Light";
}

/* ══════════════════════════════════════════════════════════
   ANALYZE — sends image to FastAPI → Gemini
══════════════════════════════════════════════════════════ */
async function analyzeImage() {
  if (!selectedFile) { alert("Please load an image first."); return; }

  const btn       = document.getElementById("analyzeBtn");
  const statusBox = document.getElementById("aiStatusBox");

  btn.disabled = true;
  btn.innerHTML = "⏳ Analyzing with Gemini...";
  statusBox.textContent = "Sending to Gemini AI...";

  try {
    const formData = new FormData();
    formData.append("file", selectedFile);

    const res = await fetch(`${API}/analyze`, { method: "POST", body: formData });

    if (!res.ok) {
      const err = await res.text();
      throw new Error(`Server ${res.status}: ${err}`);
    }

    const data = await res.json();

    if (data.error) throw new Error(data.error + " — raw: " + data.raw_response);

    renderAnalysis(data.analysis);
    updateRecentItem(selectedFile.name, data.analysis);

  } catch (err) {
    console.error(err);
    statusBox.textContent = "❌ " + err.message;
  } finally {
    btn.disabled = false;
    btn.innerHTML = "<span>⊕</span> Analyze with Gemini AI";
  }
}

/* ── Render analysis results ────────────────────────────── */
function renderAnalysis(a) {
  if (!a) return;

  document.getElementById("emptyState").style.display = "none";

  const pct = typeof a.confidence === "number" ? a.confidence : parseInt(a.confidence) || 85;
  document.getElementById("confidenceBlock").style.display = "block";
  document.getElementById("confidencePct").textContent  = pct + "%";
  document.getElementById("confidenceBar").style.width  = pct + "%";

  const summary = a.impression || a.summary || "";
  if (summary) {
    document.getElementById("summarySection").style.display = "block";
    document.getElementById("summaryText").textContent = summary;
  }

  const findings = a.findings_list || parseFindingsText(a.findings);
  if (findings && findings.length) {
    document.getElementById("findingsSection").style.display = "block";
    const list   = document.getElementById("findingsList");
    const colors = ["#f97316","#7c5cbf","#22c55e","#3b82f6","#eab308"];
    list.innerHTML = "";
    findings.forEach((f, i) => {
      const color = colors[i % colors.length];
      const card  = document.createElement("div");
      card.className = "finding-card";
      card.innerHTML = `
        <div class="finding-title">
          <span class="finding-dot" style="background:${color}"></span>
          ${f.title || f}
        </div>
        ${f.details ? `<div class="finding-details">${f.details}</div>` : ""}`;
      list.appendChild(card);
    });
    placeAnnotation(findings[0]?.title || String(findings[0]));
  }

  const recs = a.recommendations_list || parseRecsText(a.recommendation);
  if (recs && recs.length) {
    document.getElementById("recsSection").style.display = "block";
    document.getElementById("recsList").innerHTML = recs.map(r => `<li>${r}</li>`).join("");
  }

  document.getElementById("aiStatusBox").textContent = "✓ Analysis complete — " + (findings?.length || 0) + " finding(s) detected.";
}

/* ── Helpers ───────────────────────────────────────────── */
function parseFindingsText(text) {
  if (!text) return [];
  return text.split(/\n|;/).map(s => s.trim()).filter(Boolean).map(s => ({ title: s }));
}

function parseRecsText(text) {
  if (!text) return [];
  return text.split(/\n|;|\.(?=\s)/).map(s => s.trim()).filter(Boolean);
}

function placeAnnotation(label) {
  if (!label) return;
  clearAnnotations();
  const layer   = document.getElementById("annotations");
  const preview = document.getElementById("preview");
  if (!layer || !preview) return;

  const rect = preview.getBoundingClientRect();
  const cx   = rect.width  * 0.42;
  const cy   = rect.height * 0.37;

  const dot = document.createElement("div");
  dot.className = "annotation-dot";
  dot.style.cssText = `left:${cx}px; top:${cy}px;`;

  const bubble = document.createElement("div");
  bubble.className = "annotation-bubble";
  bubble.textContent = label.length > 24 ? label.slice(0, 24) + "…" : label;
  bubble.style.cssText = `left:${cx + 18}px; top:${cy - 24}px;`;

  layer.appendChild(dot);
  layer.appendChild(bubble);
}

function updateRecentItem(name, analysis) {
  const items = document.querySelectorAll(".recent-item");
  for (const item of items) {
    if (item.querySelector(".recent-name")?.textContent.includes(name)) {
      const desc = item.querySelector(".recent-desc");
      if (desc) desc.textContent = (analysis?.impression || "Analysis complete.").slice(0, 80) + "…";
      break;
    }
  }
}

/* ── Generate Report ───────────────────────────────────── */
function generateReport() {
  const summary  = document.getElementById("summaryText")?.textContent || "No summary.";
  const findings = [...document.querySelectorAll(".finding-title")].map(el => el.textContent.trim()).join("\n  - ");
  const recs     = [...document.querySelectorAll("#recsList li")].map(li => li.textContent.trim()).join("\n  - ");
  const conf     = document.getElementById("confidencePct")?.textContent || "N/A";

  const report = [
    "═══════════════════════════════════════",
    "       AI RADIOLOGY REPORT",
    "═══════════════════════════════════════",
    "Generated: " + new Date().toLocaleString(),
    "AI Model:  Gemini 2.5 Flash",
    "Confidence: " + conf,
    "",
    "CLINICAL IMPRESSION",
    "───────────────────",
    summary,
    "",
    "FINDINGS",
    "────────",
    findings ? "  - " + findings : "  None documented.",
    "",
    "RECOMMENDATIONS",
    "───────────────",
    recs ? "  - " + recs : "  None.",
    "",
    "═══════════════════════════════════════",
    "⚠  This report is AI-generated and must",
    "   be reviewed by a qualified radiologist.",
    "═══════════════════════════════════════",
  ].join("\n");

  const blob = new Blob([report], { type: "text/plain" });
  const url  = URL.createObjectURL(blob);
  const a    = Object.assign(document.createElement("a"), { href: url, download: "radiology_report.txt" });
  a.click();
  URL.revokeObjectURL(url);
}

/* ── Ask AI ────────────────────────────────────────────── */
function askAI() {
  const summary = document.getElementById("summaryText")?.textContent;
  if (!summary) { alert("Please analyze an image first."); return; }
  const q = prompt("Ask about the findings:");
  if (!q) return;
  alert("Based on the analysis:\n\n" + summary + "\n\nYour question: " + q + "\n\nPlease consult a qualified radiologist for clinical decisions.");
}

/* ══════════════════════════════════════════════════════════
   CHAT TAB
══════════════════════════════════════════════════════════ */
const chatHistory = [];   // { role: "user"|"assistant", content: string }

function chatKeydown(e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendChatMessage();
  }
}

async function sendChatMessage() {
  const input = document.getElementById("chatInput");
  const text  = input.value.trim();
  if (!text) return;

  input.value = "";
  input.style.height = "auto";

  appendChatMessage("user", text);
  chatHistory.push({ role: "user", content: text });

  const typingId = appendChatTyping();

  try {
    // Grab Gemini analysis context from the UI
    const impression     = document.getElementById("summaryText")?.textContent?.trim() || null;
    const findings       = [...document.querySelectorAll(".finding-title")]
                            .map(el => el.textContent.trim()).join("; ") || null;
    const confidenceText = document.getElementById("confidencePct")?.textContent?.trim();
    const confidence     = confidenceText ? parseInt(confidenceText) : null;
    const recommendation = [...document.querySelectorAll("#recsList li")]
                            .map(el => el.textContent.trim()).join("; ") || null;

    const analysisContext = (impression || findings)
      ? { impression, findings, confidence, recommendation }
      : null;

    const res = await fetch(`${API}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages: chatHistory.slice(-10).map(m => ({ role: m.role, content: m.content })),
        analysis_context: analysisContext,
      }),
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.error || `Server error ${res.status}`);
    }

    const data  = await res.json();
    const reply = data.reply || "Sorry, I couldn't generate a response.";

    removeTyping(typingId);
    appendChatMessage("assistant", reply);
    chatHistory.push({ role: "assistant", content: reply });

  } catch (err) {
    removeTyping(typingId);
    appendChatMessage("assistant", "⚠️ Error connecting to Groq AI: " + err.message);
  }
}

function appendChatMessage(role, content) {
  const messages = document.getElementById("chatMessages");
  const div = document.createElement("div");
  div.className = "chat-msg chat-msg-" + role;

  // Convert newlines to paragraphs for assistant messages
  const formatted = role === "assistant"
    ? content.split(/\n{2,}/).map(p => `<p>${p.replace(/\n/g, "<br>")}</p>`).join("")
    : `<p>${content.replace(/\n/g, "<br>")}</p>`;

  div.innerHTML = `
    <div class="chat-bubble">
      <div class="chat-bubble-role">${role === "user" ? "You" : "✦ AI Assistant"}</div>
      <div class="chat-bubble-content">${formatted}</div>
    </div>`;
  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
  return div;
}

let typingCounter = 0;
function appendChatTyping() {
  const id = "typing-" + (++typingCounter);
  const messages = document.getElementById("chatMessages");
  const div = document.createElement("div");
  div.id = id;
  div.className = "chat-msg chat-msg-assistant";
  div.innerHTML = `<div class="chat-bubble"><div class="chat-bubble-role">✦ AI Assistant</div><div class="chat-typing"><span></span><span></span><span></span></div></div>`;
  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
  return id;
}

function removeTyping(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

/* ══════════════════════════════════════════════════════════
   DASHBOARD TAB
══════════════════════════════════════════════════════════ */
async function loadDashboard() {
  const listEl = document.getElementById("dashReportsList");
  listEl.innerHTML = '<div class="dash-empty">Loading…</div>';

  try {
    const res  = await fetch(`${API}/history`);
    if (!res.ok) throw new Error("Backend unreachable (status " + res.status + ")");
    const rows = await res.json();

    if (!rows || rows.length === 0) {
      listEl.innerHTML = '<div class="dash-empty">No analyses yet. Load an image and click Analyze to get started.</div>';
      document.getElementById("statTotal").textContent    = "0";
      document.getElementById("statAvgConf").textContent  = "—";
      document.getElementById("statLatest").textContent   = "—";
      return;
    }

    // Stats
    document.getElementById("statTotal").textContent = rows.length;
    const avgConf = Math.round(rows.reduce((s, r) => s + (r[4] || 0), 0) / rows.length);
    document.getElementById("statAvgConf").textContent = avgConf + "%";
    document.getElementById("statLatest").textContent  = rows[0][1] || "—";

    // Reports list — rows: [id, filename, findings, impression, confidence, recommendation]
    listEl.innerHTML = "";
    rows.forEach(row => {
      const [id, filename, findings, impression, confidence, recommendation] = row;
      const card = document.createElement("div");
      card.className = "dash-report-card";
      card.setAttribute("role", "button");
      card.setAttribute("tabindex", "0");
      card.title = "Click to view full report";
      card.innerHTML = `
        <div class="dash-report-header">
          <span class="dash-report-name">📋 ${filename || "Unknown file"}</span>
          <span class="dash-report-conf ${confidence >= 80 ? "teal" : confidence >= 60 ? "orange" : "red"}">${confidence || 0}%</span>
        </div>
        <div class="dash-report-impression">${impression || "No impression recorded."}</div>
        ${recommendation ? `<div class="dash-report-rec">💡 ${recommendation.slice(0, 120)}${recommendation.length > 120 ? "…" : ""}</div>` : ""}
        <div class="dash-report-footer-row">
          <div class="dash-report-id">Report #${id}</div>
          <span class="dash-view-link">View details →</span>
        </div>`;
      card.addEventListener("click", () => openReportModal(id));
      card.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") openReportModal(id); });
      listEl.appendChild(card);
    });

  } catch (err) {
    listEl.innerHTML = `<div class="dash-empty">⚠️ Could not load history: ${err.message}<br><small>Make sure the backend is running at ${API}</small></div>`;
    document.getElementById("statTotal").textContent   = "—";
    document.getElementById("statAvgConf").textContent = "—";
    document.getElementById("statLatest").textContent  = "—";
  }
}

/* ── Report Detail Modal ───────────────────────────────── */
async function openReportModal(reportId) {
  const modal = document.getElementById("reportModal");
  modal.style.display = "flex";

  // Show loading state
  document.getElementById("modalFilename").textContent   = "Loading…";
  document.getElementById("modalConfBadge").textContent  = "";
  document.getElementById("modalImpression").textContent = "Fetching report…";
  document.getElementById("modalFindings").innerHTML     = "";
  document.getElementById("modalRecs").innerHTML         = "";
  document.getElementById("modalReportId").textContent   = "Report #" + reportId;

  try {
    const res  = await fetch(`${API}/report/${reportId}`);
    if (!res.ok) throw new Error("Status " + res.status);
    const row  = await res.json();

    if (row.error) throw new Error(row.error);

    // row = [id, filename, findings, impression, confidence, recommendation]
    const [id, filename, findings, impression, confidence, recommendation] = row;
    const confNum   = confidence || 0;
    const confClass = confNum >= 80 ? "teal" : confNum >= 60 ? "orange" : "red";

    document.getElementById("modalFilename").textContent = filename || "Unknown file";

    const badge = document.getElementById("modalConfBadge");
    badge.textContent  = confNum + "% confidence";
    badge.className    = "modal-conf-badge " + confClass;

    // Impression
    document.getElementById("modalImpression").textContent = impression || "No impression recorded.";

    // Findings — try to split on newline/semicolon into a list
    const findingsEl = document.getElementById("modalFindings");
    if (findings) {
      const items = findings.split(/\n|;/).map(s => s.trim()).filter(Boolean);
      if (items.length > 1) {
        findingsEl.innerHTML = items.map(f => `<div class="modal-finding-item"><span class="modal-finding-dot"></span>${f}</div>`).join("");
      } else {
        findingsEl.textContent = findings;
      }
    } else {
      findingsEl.textContent = "No findings recorded.";
    }

    // Recommendations
    const recsEl = document.getElementById("modalRecs");
    if (recommendation) {
      const items = recommendation.split(/\n|;|\.(?=\s)/).map(s => s.trim()).filter(Boolean);
      if (items.length > 1) {
        recsEl.innerHTML = "<ul>" + items.map(r => `<li>${r}</li>`).join("") + "</ul>";
      } else {
        recsEl.textContent = recommendation;
      }
    } else {
      recsEl.textContent = "No recommendations recorded.";
    }

    // Export button
    document.getElementById("modalExportBtn").onclick = () => exportModalReport({ id, filename, findings, impression, confidence: confNum, recommendation });

  } catch (err) {
    document.getElementById("modalImpression").textContent = "⚠️ Could not load report: " + err.message;
  }
}

function closeReportModal(event) {
  // Close if clicking overlay background or close button (no event = button click)
  if (event && event.target !== document.getElementById("reportModal")) return;
  document.getElementById("reportModal").style.display = "none";
}

// Also close on Escape key
document.addEventListener("keydown", e => {
  if (e.key === "Escape") document.getElementById("reportModal").style.display = "none";
});

function exportModalReport({ id, filename, findings, impression, confidence, recommendation }) {
  const report = [
    "═══════════════════════════════════════",
    "       AI RADIOLOGY REPORT",
    "═══════════════════════════════════════",
    "Report ID:  #" + id,
    "File:       " + (filename || "Unknown"),
    "Generated:  " + new Date().toLocaleString(),
    "AI Model:   Gemini 2.5 Flash",
    "Confidence: " + confidence + "%",
    "",
    "CLINICAL IMPRESSION",
    "───────────────────",
    impression || "None.",
    "",
    "FINDINGS",
    "────────",
    findings || "None.",
    "",
    "RECOMMENDATIONS",
    "───────────────",
    recommendation || "None.",
    "",
    "═══════════════════════════════════════",
    "⚠  This report is AI-generated and must",
    "   be reviewed by a qualified radiologist.",
    "═══════════════════════════════════════",
  ].join("\n");

  const blob = new Blob([report], { type: "text/plain" });
  const url  = URL.createObjectURL(blob);
  const a    = Object.assign(document.createElement("a"), { href: url, download: `report_${id}_${filename || "export"}.txt` });
  a.click();
  URL.revokeObjectURL(url);
}
