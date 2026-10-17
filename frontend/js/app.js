/**
 * app.js — Chat interface logic
 * Handles: sending queries, rendering messages, source citations, typing indicator
 */

const API_BASE = ""; // Same origin — Flask serves both frontend and API

// ─── State ────────────────────────────────────────────────────────────────────
let isWaiting = false;
let chatHistory = [];

// ─── Tab Switching ────────────────────────────────────────────────────────────
function switchTab(tab) {
  // Hide all panels
  document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
  document.querySelectorAll(".nav-item").forEach(n => n.classList.remove("active"));

  // Show selected panel
  document.getElementById(`panel-${tab}`).classList.add("active");
  document.getElementById(`nav-${tab}`).classList.add("active");

  // Load panel data
  if (tab === "documents") refreshDocuments();
  if (tab === "status") refreshStatus();
}

// ─── Chat Functions ───────────────────────────────────────────────────────────
function fillQuery(text) {
  const input = document.getElementById("chat-input");
  input.value = text;
  input.focus();
  autoResizeTextarea(input);
}

function clearChat() {
  chatHistory = [];
  const msgs = document.getElementById("chat-messages");
  msgs.innerHTML = `
    <div class="chat-welcome" id="chat-welcome">
      <div class="welcome-icon">🔍</div>
      <div class="welcome-title">Ask your documents anything</div>
      <div class="welcome-subtitle">
        Upload documents to the <strong>Documents</strong> tab or drop files in the
        <code>documents/</code> folder. All processing happens locally — your data never leaves this machine.
      </div>
      <div class="welcome-hints">
        <div class="hint-chip" onclick="fillQuery('Summarize the main findings')">Summarize the main findings</div>
        <div class="hint-chip" onclick="fillQuery('What are the key requirements?')">What are the key requirements?</div>
        <div class="hint-chip" onclick="fillQuery('Explain the methodology used')">Explain the methodology</div>
        <div class="hint-chip" onclick="fillQuery('List all defined terms')">List all defined terms</div>
      </div>
    </div>`;
  showToast("Chat cleared", "info");
}

async function sendMessage() {
  if (isWaiting) return;

  const input = document.getElementById("chat-input");
  const query = input.value.trim();
  if (!query) return;

  // Hide welcome screen
  const welcome = document.getElementById("chat-welcome");
  if (welcome) welcome.remove();

  // Append user message
  appendMessage("user", query);
  input.value = "";
  autoResizeTextarea(input);
  chatHistory.push({ role: "user", content: query });

  // Show typing indicator
  const typingId = showTypingIndicator();

  // Disable send button
  isWaiting = true;
  document.getElementById("send-btn").disabled = true;
  document.getElementById("input-status").textContent = "Retrieving relevant chunks…";

  try {
    const response = await fetch(`${API_BASE}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });

    const data = await response.json();

    removeTypingIndicator(typingId);

    if (!response.ok) {
      appendMessage("assistant", `⚠️ Error: ${data.error || "Unknown error"}`, []);
      showToast(`Error: ${data.error}`, "error");
    } else {
      appendMessage("assistant", data.answer, data.sources, data.chunks_found);
      chatHistory.push({ role: "assistant", content: data.answer });

      // Update chunk label
      document.getElementById("chunk-count-label").textContent =
        data.chunks_found > 0 ? `${data.chunks_found} chunks retrieved` : "";

      if (data.error) {
        showToast(`Warning: ${data.error}`, "warning");
      }
    }

  } catch (err) {
    removeTypingIndicator(typingId);
    appendMessage("assistant", `⚠️ Could not connect to server. Is the Flask app running?\n\nError: ${err.message}`, []);
    showToast("Server connection failed", "error");
  } finally {
    isWaiting = false;
    document.getElementById("send-btn").disabled = false;
    document.getElementById("input-status").textContent = "Hybrid retrieval (semantic + BM25) · Grounded by Mistral";
  }
}

function appendMessage(role, text, sources = [], chunksFound = 0) {
  const msgs = document.getElementById("chat-messages");

  const avatarContent = role === "user" ? "You" : "AI";
  const avatarEmoji = role === "user" ? "👤" : "🤖";

  const sourcesHtml = buildSourcesHtml(sources);

  const div = document.createElement("div");
  div.className = `message ${role}`;
  div.innerHTML = `
    <div class="msg-avatar">${role === "user" ? "U" : "🤖"}</div>
    <div class="msg-body">
      <div class="msg-bubble">${formatAnswer(text)}</div>
      ${sourcesHtml}
    </div>`;

  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
}

function formatAnswer(text) {
  // Basic markdown-like formatting
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.*?)\*/g, "<em>$1</em>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\n\n/g, "</p><p>")
    .replace(/\n/g, "<br>")
    .replace(/^(.+)$/, "<p>$1</p>");
}

function buildSourcesHtml(sources) {
  if (!sources || sources.length === 0) return "";

  const icons = { pdf: "📄", docx: "📝", txt: "📋", java: "☕", cs: "🔷", py: "🐍", md: "📖" };

  const cards = sources.map(s => {
    const icon = icons[s.file_type] || "📄";
    const pageHtml = s.page ? `<span class="source-page">Page ${s.page}</span>` : "";
    const scoreText = s.score ? ` · score: ${typeof s.score === 'number' ? s.score.toFixed(4) : s.score}` : "";
    return `
      <div class="source-card">
        <div class="source-header">
          <span class="source-icon">${icon}</span>
          <span class="source-filename">${escHtml(s.filename)}</span>
          ${pageHtml}
        </div>
        <div class="source-excerpt">${escHtml(s.excerpt || "")}</div>
      </div>`;
  }).join("");

  return `
    <div class="sources-section">
      <div class="sources-label">📌 Sources (${sources.length})</div>
      ${cards}
    </div>`;
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function showTypingIndicator() {
  const msgs = document.getElementById("chat-messages");
  const id = "typing-" + Date.now();
  const div = document.createElement("div");
  div.className = "message assistant";
  div.id = id;
  div.innerHTML = `
    <div class="msg-avatar">🤖</div>
    <div class="msg-body">
      <div class="msg-bubble">
        <div class="typing-indicator">
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
        </div>
      </div>
    </div>`;
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
  return id;
}

function removeTypingIndicator(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

// ─── Textarea Auto-resize ─────────────────────────────────────────────────────
function autoResizeTextarea(el) {
  el.style.height = "auto";
  el.style.height = Math.min(el.scrollHeight, 120) + "px";
}

// ─── Toast Notifications ──────────────────────────────────────────────────────
function showToast(message, type = "info") {
  const container = document.getElementById("toast-container");
  const icons = { success: "✅", error: "❌", warning: "⚠️", info: "ℹ️" };
  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  toast.innerHTML = `<span>${icons[type] || "ℹ️"}</span><span>${escHtml(message)}</span>`;
  container.appendChild(toast);
  setTimeout(() => toast.remove(), 4200);
}

// ─── Health Polling ───────────────────────────────────────────────────────────
async function pollHealth() {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    if (!res.ok) throw new Error("HTTP " + res.status);
    const data = await res.json();

    // Ollama pill
    const ollamaPill = document.getElementById("ollama-pill");
    const ollamaLabel = document.getElementById("ollama-pill-label");
    if (data.ollama.available && data.ollama.model_found) {
      ollamaPill.classList.remove("offline", "loading");
      ollamaLabel.textContent = `${data.ollama.model} · ready`;
    } else if (data.ollama.available) {
      ollamaPill.classList.add("loading");
      ollamaPill.classList.remove("offline");
      ollamaLabel.textContent = `Ollama on · model missing`;
    } else {
      ollamaPill.classList.add("offline");
      ollamaPill.classList.remove("loading");
      ollamaLabel.textContent = "Ollama offline";
    }

    // Chunk count pill
    document.getElementById("index-pill-label").textContent =
      `${data.vector_store.total_chunks} chunks indexed`;

    // Sidebar LLM model name
    document.getElementById("sidebar-llm").textContent = data.ollama.model;

    // Badge in nav
    // (doc count badge is updated by documents.js)

  } catch (e) {
    document.getElementById("ollama-pill").classList.add("offline");
    document.getElementById("ollama-pill-label").textContent = "Server offline";
  }
}

// ─── Keyboard Shortcuts ───────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  const input = document.getElementById("chat-input");

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  input.addEventListener("input", () => autoResizeTextarea(input));

  // Initial health check
  pollHealth();

  // Poll every 10 seconds
  setInterval(pollHealth, 10000);
});
