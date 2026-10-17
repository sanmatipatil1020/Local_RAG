/**
 * status.js — System status panel
 * Fetches /api/health and populates all status rows
 */

async function refreshStatus() {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    if (!res.ok) throw new Error("HTTP " + res.status);
    const data = await res.json();

    const set = (id, val, cls) => {
      const el = document.getElementById(id);
      if (!el) return;
      el.textContent = val;
      if (cls) el.className = `status-row-value ${cls}`;
    };

    // Ollama
    set("s-ollama-status",
      data.ollama.available ? "🟢 Online" : "🔴 Offline",
      data.ollama.available ? "online-chip" : "offline-chip");
    set("s-ollama-model", data.ollama.model);
    set("s-model-found",
      data.ollama.model_found ? "✅ Found" : "❌ Not found",
      data.ollama.model_found ? "online-chip" : "offline-chip");

    // ChromaDB
    set("s-chroma-status",
      data.vector_store.ok ? "🟢 Online" : "🔴 Error",
      data.vector_store.ok ? "online-chip" : "offline-chip");
    set("s-total-chunks", data.vector_store.total_chunks.toLocaleString());
    set("s-chroma-path", shortenPath(data.vector_store.directory));

    // Embedding model
    set("s-emb-model", data.embedding.model);

    // Monitor
    const m = data.monitor;
    set("s-monitor-status",
      m.running ? "🟢 Running" : "🔴 Stopped",
      m.running ? "online-chip" : "offline-chip");
    set("s-watched-dir", shortenPath(m.watched_dir));
    set("s-events", (m.events_processed || 0).toString());
    set("s-last-event", m.last_event || "None");

  } catch (e) {
    showToast("Failed to load status: " + e.message, "error");
  }
}

function shortenPath(p) {
  if (!p) return "—";
  // Show last 2 path segments for readability
  const parts = p.replace(/\\/g, "/").split("/");
  if (parts.length > 3) return "…/" + parts.slice(-2).join("/");
  return p;
}
