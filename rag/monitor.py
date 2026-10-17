"""
rag/monitor.py
───────────────
Uses the 'watchdog' library to watch the documents/ folder for:
  - New files        → index them
  - Modified files   → re-index them
  - Deleted files    → remove them from the vector store

The monitor runs in a background daemon thread so it doesn't
block the Flask web server.

How watchdog works:
  - Observer: the watchdog thread that polls the filesystem.
  - EventHandler: our subclass that reacts to file events.
"""

import os
import time
import logging
import threading
from typing import Optional

from watchdog.observers import Observer
from watchdog.events import (
    FileSystemEventHandler,
    FileCreatedEvent,
    FileModifiedEvent,
    FileDeletedEvent,
    FileMovedEvent,
)

import config
from rag.indexer import index_file, remove_file

logger = logging.getLogger(__name__)

# ─── Global monitor state ─────────────────────────────────────────────────────
_observer: Optional[Observer] = None
_monitor_status = {
    "running": False,
    "watched_dir": config.DOCUMENTS_DIR,
    "events_processed": 0,
    "last_event": None,
}
_status_lock = threading.Lock()


def get_monitor_status() -> dict:
    with _status_lock:
        return dict(_monitor_status)


def _update_status(event_description: str = None):
    with _status_lock:
        if event_description:
            _monitor_status["events_processed"] += 1
            _monitor_status["last_event"] = event_description


# ─── Event Handler ────────────────────────────────────────────────────────────

class DocumentEventHandler(FileSystemEventHandler):
    """
    React to file system events in the watched directory.
    We add a small debounce delay to avoid double-processing rapid events
    (e.g. some editors write a temp file then rename it on save).
    """

    def __init__(self):
        super().__init__()
        # Debounce: track last-seen modification time per filepath
        self._last_modified: dict = {}
        self._debounce_seconds = 1.5

    def _is_supported(self, filepath: str) -> bool:
        ext = os.path.splitext(filepath)[1].lower()
        return ext in config.SUPPORTED_EXTENSIONS

    def _debounce(self, filepath: str) -> bool:
        """Return True if enough time has passed since last event for this file."""
        now = time.time()
        last = self._last_modified.get(filepath, 0)
        if now - last < self._debounce_seconds:
            return False
        self._last_modified[filepath] = now
        return True

    def on_created(self, event: FileCreatedEvent):
        if event.is_directory:
            return
        filepath = str(event.src_path)
        if self._is_supported(filepath) and self._debounce(filepath):
            logger.info(f"[Monitor] New file detected: {os.path.basename(filepath)}")
            _update_status(f"Created: {os.path.basename(filepath)}")
            # Small delay to ensure file write is complete
            time.sleep(0.5)
            index_file(filepath)

    def on_modified(self, event: FileModifiedEvent):
        if event.is_directory:
            return
        filepath = str(event.src_path)
        if self._is_supported(filepath) and self._debounce(filepath):
            logger.info(f"[Monitor] File modified: {os.path.basename(filepath)}")
            _update_status(f"Modified: {os.path.basename(filepath)}")
            time.sleep(0.5)
            index_file(filepath)

    def on_deleted(self, event: FileDeletedEvent):
        if event.is_directory:
            return
        filepath = str(event.src_path)
        if self._is_supported(filepath):
            logger.info(f"[Monitor] File deleted: {os.path.basename(filepath)}")
            _update_status(f"Deleted: {os.path.basename(filepath)}")
            remove_file(filepath)

    def on_moved(self, event: FileMovedEvent):
        """Handle file renames: remove old path, index new path."""
        if event.is_directory:
            return
        src_path = str(event.src_path)
        dest_path = str(event.dest_path)

        # Remove old entry
        if self._is_supported(src_path):
            logger.info(f"[Monitor] File moved/renamed (old): {os.path.basename(src_path)}")
            remove_file(src_path)

        # Index new entry
        if self._is_supported(dest_path) and self._debounce(dest_path):
            logger.info(f"[Monitor] File moved/renamed (new): {os.path.basename(dest_path)}")
            _update_status(f"Renamed: {os.path.basename(dest_path)}")
            time.sleep(0.5)
            index_file(dest_path)


# ─── Start / Stop ─────────────────────────────────────────────────────────────

def start_monitor(watched_dir: str = None) -> bool:
    """
    Start the file system monitor as a background daemon thread.
    Returns True if started successfully, False if already running.
    """
    global _observer

    if _observer is not None and _observer.is_alive():
        logger.warning("Monitor is already running.")
        return False

    watched_dir = watched_dir or config.DOCUMENTS_DIR
    os.makedirs(watched_dir, exist_ok=True)

    handler = DocumentEventHandler()
    _observer = Observer()
    _observer.schedule(handler, watched_dir, recursive=True)
    _observer.daemon = True  # Dies when the main process exits
    _observer.start()

    with _status_lock:
        _monitor_status["running"] = True
        _monitor_status["watched_dir"] = watched_dir

    logger.info(f"[Monitor] Started watching: {watched_dir}")
    return True


def stop_monitor():
    """Stop the file system monitor gracefully."""
    global _observer
    if _observer is not None:
        _observer.stop()
        _observer.join(timeout=5)
        _observer = None

    with _status_lock:
        _monitor_status["running"] = False

    logger.info("[Monitor] Stopped.")
