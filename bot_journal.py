"""Local per-run JSONL journal. No broker reads, screen input or order APIs."""
import copy
import hashlib
import json
import queue
import re
import sys
import threading
import time
import uuid
import traceback as traceback_module
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PINE_FOLDER = ROOT / "אינדיקטוריםTRADING VIEW"
SOURCES = {
    "1sec-after-3bars": "instant_breakout_rsi_vwap_1sec_3bars.pine",
    "5sec": "instant_breakout_rsi_vwap_5sec.pine",
    "5sec-reversed": "instant_breakout_rsi_vwap_5sec_reversed.pine",
    "5sec-regime": "instant_breakout_rsi_vwap_5sec_regime.pine",
    "5sec-reversed-regime": "instant_breakout_rsi_vwap_5sec_reversed_regime.pine",
}


class RunJournal:
    def __init__(self, strategy_tag, dry, config, folder=None):
        self.strategy_tag = strategy_tag or "unlabeled"
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", self.strategy_tag):
            raise ValueError("Strategy label must be 1-80 ASCII letters, digits, dots, underscores or hyphens")
        directory = Path(folder) if folder is not None else ROOT / "nq_bot_logs"
        source = PINE_FOLDER / SOURCES.get(self.strategy_tag, "unavailable")
        metadata = dict(config=copy.deepcopy(config),
                        strategy_source_file=source.name if source.is_file() else None,
                        strategy_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else None,
                        bot_sha256=hashlib.sha256((ROOT / "nq_triangle_bot.py").read_bytes()).hexdigest(),
                        position_guard_sha256=hashlib.sha256((ROOT / "position_guard.py").read_bytes()).hexdigest(),
                        entry_surface_sha256=hashlib.sha256((ROOT / "entry_surface.py").read_bytes()).hexdigest(),
                        position_source="Windows UI Automation + desktop quote pixels; no broker API",
                        strategy_label_is_user_declared=True, chart_settings_verified=False,
                        fills_and_account_position_verified=False)
        directory.mkdir(parents=True, exist_ok=True)
        self.session_id = uuid.uuid4().hex
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = directory / f"{stamp}_{self.strategy_tag}_{self.session_id[:8]}.jsonl"
        self._stream = self.path.open("x", encoding="utf-8")
        self._queue = queue.SimpleQueue()
        self._error = None
        self._closed = False
        self.dry = dry
        self._thread = threading.Thread(target=self._write, daemon=True, name="nq-run-journal")
        self._thread.start()
        self.record("session_start", **metadata)

    def record(self, event, **fields):
        if self._closed:
            raise OSError("Journal is already closed")
        if self._error is not None:
            raise OSError(f"Journal writer failed: {self._error}") from self._error
        item = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "epoch_ns": time.time_ns(),
                "session_id": self.session_id, "declared_strategy": self.strategy_tag,
                "mode": "TEST" if self.dry else "RUN", "event": event, **fields}
        # Encoding and disk writes happen on the worker, not before a click.
        self._queue.put(item)

    def _write(self):
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    break
                self._stream.write(json.dumps(item, ensure_ascii=False) + "\n")
                self._stream.flush()
        except Exception as error:
            self._error = error
            print(f"Journal error: {error}", file=sys.stderr, flush=True)
        finally:
            self._stream.close()

    def close(self, outcome="stopped"):
        if self._closed:
            return
        if self._error is None:
            self.record("session_end", outcome=outcome)
        self._closed = True
        self._queue.put(None)
        self._thread.join(timeout=3)
        if self._thread.is_alive():
            print("Journal has not finished flushing; recent records may be incomplete", file=sys.stderr, flush=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            self.record("session_exception", error_type=exc_type.__name__, message=str(exc_value),
                        traceback="".join(traceback_module.format_exception(exc_type, exc_value, traceback)))
        self.close("interrupted" if exc_type is KeyboardInterrupt else "error" if exc_type else "stopped")
        return False
