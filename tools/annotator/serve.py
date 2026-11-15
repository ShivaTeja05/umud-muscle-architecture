"""Tiny local server for the UMUD annotation tool (127.0.0.1 only).

    .venv/bin/python tools/annotator/serve.py            # http://127.0.0.1:8765/
    .venv/bin/python tools/annotator/serve.py --port 9000

Routes
    GET  /               -> tools/annotator/index.html (+ app.js, style.css)
    GET  /data/<path>    -> data/annotator/<path> (manifest.json, calib_manifest.json, img/*.png,
                            calib/*.png).  The calibration answers / scores (calib_answers.json,
                            calib_score.json, calib_pipeline.json) and backups/ are NEVER served.
    GET  /api/state      -> data/annotator/autosave.json (or {} if none)
    POST /api/state      -> body {"state": {...}}; merged per image (newest wins) with the
                            copy on disk, then writes autosave.json, labels.csv (computed by
                            labels.py), calib_labels.csv (calibration tasks, if calib_manifest.json
                            exists) and an hourly copy in data/annotator/backups/

Only stdlib, single process, no caching.  The API refuses cross-origin requests (it needs a
custom header and a same-origin Origin), so other web pages cannot overwrite your labels.
Competition images are served to this machine only; never expose or upload them.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import threading
import time
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

import labels as labels_mod  # tools/annotator/labels.py (script dir is on sys.path)

TOOL_DIR = Path(__file__).resolve().parent
ROOT = TOOL_DIR.parents[1]
DEFAULT_DATA = ROOT / "data" / "annotator"
MAX_BODY = 32 * 1024 * 1024
SAVE_LOCK = threading.Lock()
STATIC = {"/": "index.html", "/index.html": "index.html", "/app.js": "app.js",
          "/style.css": "style.css", "/favicon.ico": None}
# Files under data/annotator/ the browser must never get (answers would anchor the annotator).
DENY_PREFIXES = ("calib_answers", "calib_score", "calib_pipeline",
                 # manual-labelling kit answer files (they live in data/manual_labelling/private/, outside
                 # every served data dir; denied here too in case one is ever copied into a data dir)
                 "check_key", "pass_bar", "practice_key", "check_result", "selection", "work_selection",
                 "build_cache")
DENY_DIRS = {"backups", "private", "check_results"}  # compared in lower case (the macOS disk ignores case)


def servable(data_dir: Path, rel: str) -> Path | None:
    """The file under data_dir for a /data/<rel> request, or None when it must not be served."""

    root = data_dir.resolve()
    try:
        target = (root / rel).resolve()
    except (ValueError, OSError):  # e.g. an embedded NUL byte (%00) in the request path
        return None
    if not target.is_relative_to(root) or target.suffix.lower() not in {".png", ".json", ".csv"}:
        return None
    parts = target.relative_to(root).parts
    if any(p.lower() in DENY_DIRS for p in parts[:-1]) or target.name.lower().startswith(DENY_PREFIXES):
        return None
    return target


def atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class Handler(SimpleHTTPRequestHandler):
    data_dir: Path = DEFAULT_DATA

    def log_message(self, fmt: str, *args) -> None:  # quieter: only API calls and errors
        if "/api/" in self.path or (args and str(args[1])[:1] in "45"):
            super().log_message(fmt, *args)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def translate_path(self, path: str) -> str:
        url_path = unquote(urlsplit(path).path)
        if url_path in STATIC:
            name = STATIC[url_path]
            return str(TOOL_DIR / name) if name else str(TOOL_DIR / "__missing__")
        if url_path.startswith("/data/"):
            target = servable(self.data_dir, url_path[len("/data/"):])
            if target is not None:
                return str(target)
        return str(TOOL_DIR / "__missing__")

    def list_directory(self, path):  # never list directories
        self.send_error(HTTPStatus.NOT_FOUND)
        return None

    def _send_json(self, payload: object, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _host_ok(self) -> bool:
        """Reject DNS-rebinding requests: the Host must be the loopback address we bound."""

        host = self.headers.get("Host", "")
        port = self.server.server_address[1]
        return host in {f"127.0.0.1:{port}", f"localhost:{port}"}

    def _same_origin(self) -> bool:
        if not self._host_ok() or self.headers.get("X-Annotator") != "1":
            return False
        origin = self.headers.get("Origin")
        host = self.headers.get("Host", "")
        return origin is None or origin in {f"http://{host}"}

    def do_GET(self) -> None:  # noqa: N802
        if not self._host_ok():
            return self._send_json({"error": "bad host"}, 403)
        if urlsplit(self.path).path == "/api/state":
            if not self._same_origin():
                return self._send_json({"error": "forbidden"}, 403)
            path = self.data_dir / "autosave.json"
            state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            return self._send_json(state)
        return super().do_GET()

    def do_HEAD(self) -> None:  # noqa: N802
        if not self._host_ok():
            return self._send_json({"error": "bad host"}, 403)
        return super().do_HEAD()

    def do_POST(self) -> None:  # noqa: N802
        if urlsplit(self.path).path != "/api/state":
            return self._send_json({"error": "not found"}, 404)
        if not self._same_origin():
            return self._send_json({"error": "forbidden"}, 403)
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY:
            return self._send_json({"error": "bad length"}, 400)
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            state = payload["state"]
            if not isinstance(state, dict) or not isinstance(state.get("labels"), dict):
                raise ValueError("state.labels missing")
        except (ValueError, KeyError, UnicodeDecodeError) as exc:
            return self._send_json({"error": f"bad payload: {exc}"}, 400)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        auto = self.data_dir / "autosave.json"
        with SAVE_LOCK:
            disk = json.loads(auto.read_text(encoding="utf-8")) if auto.exists() else {}
            merged, newer = labels_mod.merge_states(disk, state)
            atomic_write(auto, json.dumps(merged, separators=(",", ":")))
            manifest = json.loads((self.data_dir / "manifest.json").read_text(encoding="utf-8"))
            rows = labels_mod.build_rows(merged, manifest)
            atomic_write(self.data_dir / "labels.csv", labels_mod.rows_to_csv(rows))
            calib_path = self.data_dir / "calib_manifest.json"
            crows = []
            if calib_path.exists():
                calib = json.loads(calib_path.read_text(encoding="utf-8"))
                crows = labels_mod.build_calib_rows(merged, calib)
                atomic_write(self.data_dir / "calib_labels.csv",
                             labels_mod.rows_to_csv(crows, labels_mod.CALIB_HEADER))
            backups = self.data_dir / "backups"
            backups.mkdir(exist_ok=True)
            hourly = backups / f"autosave_{time.strftime('%Y%m%d_%H')}.json"
            if not hourly.exists():
                shutil.copyfile(auto, hourly)
        n_complete = len({r["task_id"] for r in rows if r["complete"] == 1})
        return self._send_json({"ok": True, "saved": time.strftime("%H:%M:%S"),
                                "n_labels": len(merged["labels"]), "n_complete": n_complete,
                                "n_calib_complete": sum(r["complete"] == 1 for r in crows),
                                "newer_on_disk": newer})


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the UMUD annotator on 127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    if not (args.data / "manifest.json").exists():
        raise SystemExit(f"{args.data}/manifest.json missing: run tools/annotator/prepare.py")
    Handler.data_dir = args.data.resolve()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.daemon_threads = True
    print(f"UMUD annotator: http://127.0.0.1:{args.port}/   (data: {Handler.data_dir})")
    print(f"Ctrl+C to stop. Labels autosave to {Handler.data_dir}/autosave.json + labels.csv"
          + (" + calib_labels.csv" if (args.data / "calib_manifest.json").exists() else ""))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
