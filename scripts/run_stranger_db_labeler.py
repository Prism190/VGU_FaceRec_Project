#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fas_kd.pipeline import append_known_identity, ensure_face_db_layout, load_session_group_embeddings


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _resolve_face_db_root(path_arg: str) -> Path:
    root = Path(path_arg)
    if not root.is_absolute():
        root = (PROJECT_ROOT / root).resolve()
    return root


def _resolve_session_dir(stranger_sessions_dir: Path, session_arg: str) -> Path:
    token = str(session_arg).strip()
    if token.lower() == "latest":
        candidates = sorted(
            (p for p in stranger_sessions_dir.iterdir() if p.is_dir() and (p / "manifest.json").exists()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(f"No stranger sessions found in {stranger_sessions_dir}")
        return candidates[0].resolve()

    candidate = Path(token)
    if candidate.is_absolute():
        session_dir = candidate.resolve()
    else:
        session_dir = (stranger_sessions_dir / candidate).resolve()

    if not session_dir.exists() or not session_dir.is_dir():
        raise FileNotFoundError(f"session directory not found: {session_dir}")
    if not (session_dir / "manifest.json").exists():
        raise FileNotFoundError(f"manifest.json not found in session directory: {session_dir}")
    return session_dir


@dataclass
class UIState:
    db_root: Path
    known_identities_dir: Path
    session_dir: Path
    manifest_path: Path
    show_promoted: bool


class StrangerLabelHandler(BaseHTTPRequestHandler):
    state: UIState

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _load_manifest(self) -> dict[str, Any]:
        payload = json.loads(self.state.manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("manifest.json must be a JSON object")
        groups = payload.get("groups", [])
        if not isinstance(groups, list):
            payload["groups"] = []
        return payload

    def _save_manifest(self, payload: dict[str, Any]) -> None:
        self.state.manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    def _safe_session_rel(self, rel_path: str) -> Path | None:
        rel = Path(rel_path)
        if rel.is_absolute():
            return None
        candidate = (self.state.session_dir / rel).resolve()
        try:
            candidate.relative_to(self.state.session_dir)
        except ValueError:
            return None
        return candidate

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == "/":
            body = self._render_html().encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path == "/api/data":
            manifest = self._load_manifest()
            labels = {}
            clusters: list[dict[str, Any]] = []
            for group in manifest.get("groups", []):
                if not isinstance(group, dict):
                    continue
                promoted = bool(group.get("promoted", False))
                if promoted and not self.state.show_promoted:
                    continue

                try:
                    gid = int(group.get("group_id"))
                except Exception:
                    continue

                out = dict(group)
                out["group_id"] = gid
                out["sample_urls"] = [
                    "/samples/" + urllib.parse.quote(str(rel), safe="/") for rel in out.get("samples", [])
                ]
                out["num_samples"] = int(len(out.get("samples", [])))
                out["promoted"] = promoted
                if promoted and group.get("promoted_identity_name"):
                    labels[str(gid)] = str(group.get("promoted_identity_name"))
                clusters.append(out)

            self._send_json(
                {
                    "session_name": str(manifest.get("session_name") or self.state.session_dir.name),
                    "manifest_path": str(self.state.manifest_path),
                    "known_identities_dir": str(self.state.known_identities_dir),
                    "show_promoted": bool(self.state.show_promoted),
                    "labels": labels,
                    "groups": clusters,
                }
            )
            return

        if parsed.path.startswith("/samples/"):
            rel = urllib.parse.unquote(parsed.path[len("/samples/") :])
            path = self._safe_session_rel(rel)
            if path is None or not path.exists() or not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            data = path.read_bytes()
            if path.suffix.lower() in {".jpg", ".jpeg"}:
                mime = "image/jpeg"
            elif path.suffix.lower() == ".png":
                mime = "image/png"
            elif path.suffix.lower() == ".webp":
                mime = "image/webp"
            else:
                mime = "application/octet-stream"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/api/promote":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception:
            self._send_json({"ok": False, "error": "invalid_json"}, status=400)
            return

        raw_labels = payload.get("labels", {})
        if not isinstance(raw_labels, dict):
            self._send_json({"ok": False, "error": "labels_must_be_object"}, status=400)
            return

        labels: dict[int, str] = {}
        for raw_gid, raw_name in raw_labels.items():
            try:
                gid = int(raw_gid)
            except Exception:
                continue
            name = str(raw_name).strip()
            if not name:
                continue
            labels[gid] = name

        manifest = self._load_manifest()
        groups = manifest.get("groups", [])
        if not isinstance(groups, list):
            groups = []
            manifest["groups"] = groups

        group_by_id: dict[int, dict[str, Any]] = {}
        for group in groups:
            if not isinstance(group, dict):
                continue
            try:
                gid = int(group.get("group_id"))
            except Exception:
                continue
            group_by_id[gid] = group

        emb_map = load_session_group_embeddings(
            session_dir=self.state.session_dir,
            embeddings_file=str(manifest.get("embeddings_file") or "group_embeddings.npz"),
        )

        promoted_count = 0
        skipped_missing_embedding = 0
        skipped_missing_group = 0
        skipped_already_promoted = 0
        promoted: list[dict[str, Any]] = []

        for gid in sorted(labels.keys()):
            name = labels[gid]
            group = group_by_id.get(gid)
            if group is None:
                skipped_missing_group += 1
                continue
            if bool(group.get("promoted", False)):
                skipped_already_promoted += 1
                continue

            vec = emb_map.get(gid)
            if vec is None:
                skipped_missing_embedding += 1
                continue

            sample_paths = []
            for rel in group.get("samples", []):
                p = self._safe_session_rel(str(rel))
                if p is not None and p.exists() and p.is_file():
                    sample_paths.append(p)

            append_out = append_known_identity(
                db_root=self.state.db_root,
                identity_name=name,
                embeddings=vec.reshape(1, -1),
                sample_image_paths=sample_paths,
            )

            group["promoted"] = True
            group["promoted_identity_id"] = int(append_out["identity_id"])
            group["promoted_identity_name"] = str(name)
            group["promoted_at"] = _utc_now_iso()

            promoted_count += 1
            promoted.append(
                {
                    "group_id": int(gid),
                    "identity_id": int(append_out["identity_id"]),
                    "identity_name": str(name),
                    "copied_photos": int(append_out["copied_photos"]),
                }
            )

        manifest["last_promoted_at"] = _utc_now_iso()
        self._save_manifest(manifest)

        self._send_json(
            {
                "ok": True,
                "promoted_count": int(promoted_count),
                "skipped_missing_embedding": int(skipped_missing_embedding),
                "skipped_missing_group": int(skipped_missing_group),
                "skipped_already_promoted": int(skipped_already_promoted),
                "promoted": promoted,
                "manifest_path": str(self.state.manifest_path),
                "known_identities_dir": str(self.state.known_identities_dir),
            }
        )

    def log_message(self, fmt: str, *args: Any) -> None:
        print("[stranger-ui] " + (fmt % args))

    def _render_html(self) -> str:
        return """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Stranger Session Labeler</title>
  <style>
    :root {
      --bg: #f0ebe2;
      --card: #ffffff;
      --ink: #1f2a37;
      --muted: #6b7280;
      --accent: #0f766e;
      --warn: #b45309;
      --border: #d6d3d1;
    }
    body {
      margin: 0;
      font-family: "Segoe UI", "Helvetica Neue", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(1200px 500px at 0% 0%, #e4f1ea 0%, transparent 62%),
        radial-gradient(1000px 460px at 100% 0%, #fcd9a8 0%, transparent 60%),
        var(--bg);
    }
    .wrap {
      max-width: 1320px;
      margin: 0 auto;
      padding: 24px;
    }
    h1 {
      margin: 0 0 4px;
      font-size: 30px;
    }
    p {
      margin: 0 0 12px;
      color: var(--muted);
    }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 14px;
    }
    button {
      border: 1px solid var(--border);
      border-radius: 10px;
      background: var(--card);
      color: var(--ink);
      padding: 10px 14px;
      cursor: pointer;
      font-weight: 700;
      letter-spacing: 0.2px;
    }
    button:hover {
      border-color: var(--accent);
      color: var(--accent);
    }
    .status {
      margin-bottom: 12px;
      font-size: 14px;
      color: var(--muted);
      white-space: pre-wrap;
      background: rgba(255,255,255,0.62);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(310px, 1fr));
      gap: 14px;
    }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 14px;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.05);
    }
    .card.promoted {
      border-color: #9ca3af;
      opacity: 0.8;
    }
    .title {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      margin-bottom: 6px;
      gap: 8px;
    }
    .stats {
      font-size: 13px;
      color: var(--muted);
      line-height: 1.36;
      margin-bottom: 8px;
    }
    .samples {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 6px;
      margin-bottom: 10px;
    }
    .samples img {
      width: 100%;
      aspect-ratio: 1;
      object-fit: cover;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: #f8fafc;
    }
    label {
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 4px;
    }
    input[type=text] {
      width: 100%;
      box-sizing: border-box;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 8px 10px;
      font-size: 14px;
    }
    .badge {
      display: inline-block;
      border-radius: 999px;
      padding: 2px 8px;
      font-size: 11px;
      font-weight: 700;
      color: white;
      background: var(--warn);
    }
    .badge.ok {
      background: var(--accent);
    }
    .empty {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px;
      color: var(--muted);
    }
  </style>
</head>
<body>
  <div class=\"wrap\">
    <h1>Stranger Session Labeler</h1>
    <p>Assign a name to each stranger group and promote to known face database.</p>
    <div class=\"toolbar\">
      <button id=\"btnPromote\">Promote Labeled Groups</button>
      <button id=\"btnReload\">Reload</button>
    </div>
    <div id=\"status\" class=\"status\"></div>
    <div id=\"container\"></div>
  </div>

  <script>
    const statusEl = document.getElementById('status');
    const container = document.getElementById('container');
    let state = null;
    let labels = {};

    function setStatus(text) {
      statusEl.textContent = text;
    }

    function render() {
      const groups = (state && Array.isArray(state.groups)) ? state.groups : [];
      if (!groups.length) {
        container.innerHTML = '<div class="empty">No groups to label in this session.</div>';
        return;
      }

      const grid = document.createElement('div');
      grid.className = 'grid';

      groups.forEach((group) => {
        const gid = Number(group.group_id);
        const key = String(gid);
        const promoted = !!group.promoted;

        const card = document.createElement('div');
        card.className = promoted ? 'card promoted' : 'card';

        const title = document.createElement('div');
        title.className = 'title';
        const left = document.createElement('strong');
        left.textContent = `Group-${gid}`;

        const right = document.createElement('span');
        right.className = promoted ? 'badge ok' : 'badge';
        right.textContent = promoted ? 'promoted' : 'pending';

        title.appendChild(left);
        title.appendChild(right);

        const stats = document.createElement('div');
        stats.className = 'stats';
        const sim = group.max_similarity_to_group;
        const simText = (sim == null) ? 'n/a' : Number(sim).toFixed(3);
        stats.textContent =
          `tracks=${Number(group.num_tracks || 0)} | samples=${Number(group.num_samples || 0)}\n` +
          `frames=${group.first_frame}..${group.last_frame} | maxGroupSim=${simText}`;

        const samples = document.createElement('div');
        samples.className = 'samples';
        const urls = Array.isArray(group.sample_urls) ? group.sample_urls : [];
        urls.forEach((u) => {
          const img = document.createElement('img');
          img.loading = 'lazy';
          img.src = u;
          img.alt = `Group-${gid}`;
          samples.appendChild(img);
        });

        const label = document.createElement('label');
        label.textContent = 'Identity label';

        const input = document.createElement('input');
        input.type = 'text';
        input.placeholder = 'Example: Sarah';
        input.value = String(labels[key] || '');
        input.disabled = promoted;
        input.addEventListener('change', () => {
          const v = input.value.trim();
          if (v) {
            labels[key] = v;
          } else {
            delete labels[key];
          }
        });

        card.appendChild(title);
        card.appendChild(stats);
        card.appendChild(samples);
        card.appendChild(label);
        card.appendChild(input);
        grid.appendChild(card);
      });

      container.innerHTML = '';
      container.appendChild(grid);
    }

    async function loadData() {
      setStatus('Loading session...');
      const res = await fetch('/api/data');
      state = await res.json();
      labels = Object.assign({}, state.labels || {});
      render();
      setStatus(
        `session: ${state.session_name}\n` +
        `manifest: ${state.manifest_path}\n` +
        `known identities dir: ${state.known_identities_dir}`
      );
    }

    async function promote() {
      setStatus('Promoting labeled groups into known DB...');
      const res = await fetch('/api/promote', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({labels})
      });
      const out = await res.json();
      if (!out.ok) {
        setStatus(`Promotion failed: ${JSON.stringify(out)}`);
        return;
      }
      setStatus(
        `Promoted: ${out.promoted_count}\n` +
        `Skipped (missing embedding): ${out.skipped_missing_embedding}\n` +
        `Skipped (missing group): ${out.skipped_missing_group}\n` +
        `Skipped (already promoted): ${out.skipped_already_promoted}\n` +
        `manifest: ${out.manifest_path}\n` +
        `known identities dir: ${out.known_identities_dir}`
      );
      await loadData();
    }

    document.getElementById('btnPromote').addEventListener('click', promote);
    document.getElementById('btnReload').addEventListener('click', loadData);
    loadData();
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Label stranger groups from DB session and promote to known DB")
    parser.add_argument("--face-db-root", default="data/face_db")
    parser.add_argument("--session", default="latest", help="Session name under strangers/sessions or full session path")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--show-promoted", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--open-browser", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    face_db_root = _resolve_face_db_root(args.face_db_root)
    layout = ensure_face_db_layout(face_db_root)

    session_dir = _resolve_session_dir(layout["stranger_sessions"], args.session)
    manifest_path = (session_dir / "manifest.json").resolve()

    state = UIState(
        db_root=face_db_root,
        known_identities_dir=layout["known_identities"],
        session_dir=session_dir,
        manifest_path=manifest_path,
        show_promoted=bool(args.show_promoted),
    )

    def _handler(*h_args, **h_kwargs):
        StrangerLabelHandler.state = state
        return StrangerLabelHandler(*h_args, **h_kwargs)

    server = ThreadingHTTPServer((args.host, int(args.port)), _handler)
    url = f"http://{args.host}:{int(args.port)}/"

    print("[stranger-ui] ready")
    print(f"[stranger-ui] face_db_root={face_db_root}")
    print(f"[stranger-ui] session_dir={session_dir}")
    print(f"[stranger-ui] known_identities_dir={layout['known_identities']}")
    print(f"[stranger-ui] url={url}")

    if bool(args.open_browser):
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[stranger-ui] stopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
