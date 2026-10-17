/**
 * documents.js — Document management panel
 * Handles: file listing, upload, delete, rescan, status badges
 */

const TYPE_ICONS = {
  pdf: "📄", docx: "📝", txt: "📋",
  java: "☕", cs: "🔷", py: "🐍", md: "📖"
};

// ─── Refresh document list ────────────────────────────────────────────────────
async function refreshDocuments() {
  const btn = document.getElementById("refresh-btn");
  if (btn) btn.textContent = "⏳ Loading…";

  try {
    const res = await fetch(`${API_BASE}/api/documents`);
    const data = await res.json();

    renderDocTable(data.documents || []);
    updateDocStats(data);

    // Update nav badge
    const indexed = (data.documents || []).filter(d => d.status === "indexed").length;
    const badge = document.getElementById("doc-count-badge");
    if (badge) badge.textContent = indexed;

  } catch (e) {
    showToast("Failed to load documents: " + e.message, "error");
  } finally {
    if (btn) btn.textContent = "🔄 Refresh";
  }
}

function updateDocStats(data) {
  const docs = data.documents || [];
  const indexed = docs.filter(d => d.status === "indexed").length;
  const errors = docs.filter(d => d.status === "error").length;

  const el = (id, val) => { const e = document.getElementById(id); if (e) e.textContent = val; };
  el("stat-total-docs", docs.length);
  el("stat-indexed", indexed);
  el("stat-chunks", data.total_chunks ?? "—");
  el("stat-errors", errors);
}

function renderDocTable(docs) {
  const tbody = document.getElementById("doc-table-body");
  if (!tbody) return;

  if (!docs || docs.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="5">
          <div class="empty-state">
            <div class="empty-state-icon">📭</div>
            <p>No documents indexed yet.<br>
            Upload files above or drop them in the <code>documents/</code> folder.</p>
          </div>
        </td>
      </tr>`;
    return;
  }

  tbody.innerHTML = docs.map(doc => {
    const ext = doc.file_type ||
      ((doc.filename || "").split(".").pop() || "").toLowerCase();

    const icon = TYPE_ICONS[ext] || "📄";
    const badgeClass = `badge-${ext}`;
    const statusHtml = buildStatusBadge(doc);

    const errorHtml = doc.error
      ? `<div class="error-text" title="${escHtml(doc.error)}">
          ⚠ ${escHtml(doc.error.slice(0, 60))}${doc.error.length > 60 ? "…" : ""}
        </div>`
      : "";

    const filepath = doc.filepath || "";
    const safeFilepath = encodeURIComponent(filepath);

    return `
      <tr id="doc-row-${safeFilepath}">
        <td>
          <span class="doc-filename">${icon} ${escHtml(doc.filename || "")}</span>
          ${errorHtml}
        </td>
        <td>
          <span class="doc-type-badge ${badgeClass}">${ext.toUpperCase()}</span>
        </td>
        <td>${doc.chunk_count ?? "—"}</td>
        <td>${statusHtml}</td>
        <td>
          <button
            type="button"
            class="btn-icon delete-doc-btn"
            data-filepath="${safeFilepath}"
            title="Remove from index"
            aria-label="Delete ${escHtml(doc.filename || "document")}">
            🗑️
          </button>
        </td>
      </tr>`;
  }).join("");
}

function buildStatusBadge(doc) {
  switch (doc.status) {
    case "indexed":
      return `<span class="status-badge status-indexed">✅ Indexed</span>`;
    case "indexing":
      return `<span class="status-badge status-indexing"><span class="dot-spin"></span> Indexing…</span>`;
    case "error":
      return `<span class="status-badge status-error">❌ Error</span>`;
    case "deleted":
      return `<span class="status-badge status-deleted">🗑 Deleted</span>`;
    default:
      return `<span class="status-badge">${escHtml(doc.status || "unknown")}</span>`;
  }
}

// ─── Delete a document from index ────────────────────────────────────────────
// ─── Delete a document ──────────────────────────────────────
async function deleteDocument(filepath) {
    if (!filepath) {
        showToast("Cannot delete: document path is missing.", "error");
        return;
    }

    const filename = filepath.split(/[\\/]/).pop();

    const confirmed = confirm(
        `Delete "${filename}" from the index?\n\n` +
        "The original file will remain on disk unless your backend deletes it."
    );

    if (!confirmed) return;

    // Find the corresponding table row and disable its delete button.
    const rowId = `doc-row-${encodeURIComponent(filepath)}`;
    const row = document.getElementById(rowId);
    const deleteButton = row?.querySelector("button");

    if (deleteButton) {
        deleteButton.disabled = true;
        deleteButton.textContent = "⏳";
    }

    try {
        const response = await fetch(`${API_BASE}/api/documents/delete`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ filepath })
        });

        // Read the response safely, even if the server returns non-JSON.
        const responseText = await response.text();

        let data = {};

        try {
            data = responseText ? JSON.parse(responseText) : {};
        } catch {
            data = { error: responseText || "Invalid server response." };
        }

        if (!response.ok) {
            throw new Error(
                data.error ||
                data.message ||
                `Server returned HTTP ${response.status}`
            );
        }

        showToast(
            data.message || `Successfully removed "${filename}".`,
            "success"
        );

        // Refresh the table to reflect the backend's actual state.
        await refreshDocuments();

    } catch (error) {
        console.error("Document deletion failed:", error);

        showToast(
            `Delete failed: ${error.message}`,
            "error"
        );

        if (deleteButton) {
            deleteButton.disabled = false;
            deleteButton.textContent = "🗑️";
        }
    }
}

// ─── Manual Rescan ────────────────────────────────────────────────────────────
async function triggerRescan() {
  const btn = document.getElementById("rescan-btn");
  if (btn) {
    btn.disabled = true;
    btn.textContent = "⏳ Rescanning…";
  }

  try {
    const res = await fetch(`${API_BASE}/api/documents/rescan`, { method: "POST" });
    const data = await res.json();
    showToast(data.message || "Rescan started", "info");

    // Poll for updates every 2 seconds for 10 seconds
    let polls = 0;
    const interval = setInterval(async () => {
      await refreshDocuments();
      polls++;
      if (polls >= 5) clearInterval(interval);
    }, 2000);

  } catch (e) {
    showToast("Rescan failed: " + e.message, "error");
  } finally {
    setTimeout(() => {
      if (btn) { btn.disabled = false; btn.textContent = "⚡ Rescan Folder"; }
    }, 3000);
  }
}

// ─── File Upload ──────────────────────────────────────────────────────────────
async function handleFileUpload(files) {
  if (!files || files.length === 0) return;

  const progressWrapper = document.getElementById("upload-progress-wrapper");
  const progressFill = document.getElementById("upload-progress-fill");
  const progressLabel = document.getElementById("upload-progress-label");

  progressWrapper.classList.remove("hidden");

  let completed = 0;
  const total = files.length;

  for (const file of Array.from(files)) {
    progressLabel.textContent = `Uploading ${file.name}… (${completed + 1}/${total})`;
    progressFill.style.width = `${Math.round((completed / total) * 100)}%`;

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${API_BASE}/api/documents/upload`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();

      if (res.ok) {
        showToast(`"${file.name}" uploaded — indexing in background`, "success");
      } else {
        showToast(`Upload failed for "${file.name}": ${data.error}`, "error");
      }
    } catch (e) {
      showToast(`Upload error for "${file.name}": ${e.message}`, "error");
    }

    completed++;
  }

  progressFill.style.width = "100%";
  progressLabel.textContent = `Done! ${completed} file(s) uploaded.`;

  // Refresh after 1.5s to show indexing status
  setTimeout(() => {
    progressWrapper.classList.add("hidden");
    progressFill.style.width = "0%";
    refreshDocuments();
  }, 2000);

  // Poll for a few seconds to catch indexing completion
  let polls = 0;
  const interval = setInterval(() => {
    refreshDocuments();
    polls++;
    if (polls >= 6) clearInterval(interval);
  }, 2000);

  // Reset file input so same file can be re-uploaded
  document.getElementById("file-input").value = "";
}

// ─── Drag & Drop ──────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  const zone = document.getElementById("upload-zone");
  if (!zone) return;

  zone.addEventListener("dragover", (e) => {
    e.preventDefault();
    zone.classList.add("drag-over");
  });

  zone.addEventListener("dragleave", () => zone.classList.remove("drag-over"));

  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("drag-over");
    const files = e.dataTransfer?.files;
    if (files && files.length > 0) handleFileUpload(files);
  });
});
// Handle clicks on document delete buttons
document.addEventListener("click", function (event) {
  const button = event.target.closest(".delete-doc-btn");

  if (!button) return;

  const filepath = decodeURIComponent(
    button.getAttribute("data-filepath") || ""
  );

  if (!filepath) {
    console.error("Delete failed: document filepath is empty.");
    alert("Cannot delete this document because its filepath is missing.");
    return;
  }

  deleteDocument(filepath);
});
