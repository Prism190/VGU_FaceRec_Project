#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import urllib.parse
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_checkpoint(config_path: Path, checkpoint_arg: str) -> Path:
    import yaml

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_root = Path(cfg["experiment"]["output_root"])
    if not output_root.is_absolute():
        output_root = (PROJECT_ROOT / output_root).resolve()
    ckpt_dir = output_root / "checkpoints"

    alias = checkpoint_arg.strip().lower()
    if alias in {"latest", "best", "swa"}:
        path = ckpt_dir / f"{alias}.pt"
        if not path.exists():
            raise FileNotFoundError(f"checkpoint alias not found: {path}")
        return path

    candidate = Path(checkpoint_arg)
    if not candidate.is_absolute():
        candidate = (PROJECT_ROOT / candidate).resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"checkpoint not found: {candidate}")
    return candidate


def _build_identity_name_map(
    labels: dict[str, str],
    *,
    unknown_manifest: dict[str, Any],
    gallery_manifest: dict[str, Any],
) -> dict[int, str]:
    track_to_identity: dict[int, int] = {}
    for item in gallery_manifest.get("tracks", []):
        if "track_id" not in item or "assigned_identity_id" not in item:
            continue
        try:
            tid = int(item["track_id"])
            iid = int(item["assigned_identity_id"])
        except Exception:
            continue
        track_to_identity[tid] = iid

    group_to_tracks: dict[int, list[int]] = {}
    for cluster in unknown_manifest.get("clusters", []):
        if "group_id" not in cluster:
            continue
        try:
            gid = int(cluster["group_id"])
        except Exception:
            continue
        tracks: list[int] = []
        for tid in cluster.get("track_ids", []):
            try:
                tracks.append(int(tid))
            except Exception:
                continue
        group_to_tracks[gid] = tracks

    identity_names: dict[int, str] = {}
    for raw_gid, raw_name in labels.items():
        name = str(raw_name).strip()
        if not name:
            continue
        try:
            gid = int(raw_gid)
        except Exception:
            continue
        for tid in group_to_tracks.get(gid, []):
            iid = track_to_identity.get(tid)
            if iid is not None:
                identity_names[int(iid)] = name

    return identity_names


def _default_python_bin() -> Path:
    venv_py = PROJECT_ROOT / "venv" / "bin" / "python"
    if venv_py.exists():
        return venv_py
    return Path(sys.executable)


def _build_pass1_command(args: argparse.Namespace, outputs: dict[str, Path]) -> list[str]:
    py_bin = str(Path(args.python_bin))
    cmd = [
        py_bin,
        str(PROJECT_ROOT / "scripts" / "run_face_pipeline.py"),
        "--config",
        str(args.config),
        "--checkpoint",
        str(args.checkpoint),
        "--source",
        str(args.source),
        "--detector-model",
        str(args.detector_model),
        "--det-conf",
        str(args.det_conf),
        "--det-iou",
        str(args.det_iou),
        "--det-imgsz",
        str(args.det_imgsz),
        "--det-rescue-conf",
        str(args.det_rescue_conf),
        "--det-rescue-imgsz",
        str(args.det_rescue_imgsz),
        "--det-rescue-min-primary",
        str(args.det_rescue_min_primary),
        "--tracker-backend",
        "deepsort",
        "--track-max-missed-frames",
        str(args.track_max_missed_frames),
        "--track-n-init",
        str(args.track_n_init),
        "--track-max-iou-distance",
        str(args.track_max_iou_distance),
        "--track-max-cosine-distance",
        str(args.track_max_cosine_distance),
        "--track-nn-budget",
        str(args.track_nn_budget),
        "--quality-min",
        str(args.quality_min),
        "--quality-max",
        str(args.quality_max),
        "--match-threshold",
        "0.99",
        "--match-topk",
        str(args.match_topk),
        "--match-min-margin",
        str(args.match_min_margin),
        "--reid-min-track-frames",
        str(args.reid_min_track_frames),
        "--reid-once-per-track",
        "--unknown-group-threshold",
        str(args.unknown_group_threshold),
        "--unknown-min-track-frames",
        str(args.unknown_min_track_frames),
        "--unknown-min-mean-magnitude",
        str(args.unknown_min_mean_magnitude),
        "--unknown-max-samples-per-group",
        str(args.unknown_max_samples_per_group),
        "--unknown-sample-min-gap",
        str(args.unknown_sample_min_gap),
        "--gallery-min-track-frames",
        str(args.gallery_min_track_frames),
        "--gallery-dedupe-threshold",
        str(args.gallery_dedupe_threshold),
        "--gallery-min-mean-magnitude",
        str(args.gallery_min_mean_magnitude),
        "--out-jsonl",
        str(outputs["pass1_jsonl"]),
        "--out-summary",
        str(outputs["pass1_summary"]),
        "--out-gallery-npz",
        str(outputs["gallery_npz"]),
        "--out-gallery-manifest",
        str(outputs["gallery_manifest"]),
        "--out-unknown-manifest",
        str(outputs["unknown_manifest"]),
        "--out-unknown-review-html",
        str(outputs["legacy_unknown_html"]),
        "--max-frames",
        str(args.max_frames),
        "--print-every",
        str(args.print_every),
    ]
    return cmd


def _write_pass2_command(args: argparse.Namespace, outputs: dict[str, Path], identity_names_json: Path) -> Path:
    cmd = [
        str(Path(args.python_bin)),
        str(PROJECT_ROOT / "scripts" / "run_face_pipeline.py"),
        "--config",
        str(args.config),
        "--checkpoint",
        str(args.checkpoint),
        "--source",
        str(args.source),
        "--detector-model",
        str(args.detector_model),
        "--det-conf",
        str(args.det_conf),
        "--det-iou",
        str(args.det_iou),
        "--det-imgsz",
        str(args.det_imgsz),
        "--det-rescue-conf",
        str(args.det_rescue_conf),
        "--det-rescue-imgsz",
        str(args.det_rescue_imgsz),
        "--det-rescue-min-primary",
        str(args.det_rescue_min_primary),
        "--gallery-npz",
        str(outputs["gallery_npz"]),
        "--identity-names-json",
        str(identity_names_json),
        "--tracker-backend",
        "deepsort",
        "--track-max-missed-frames",
        str(args.track_max_missed_frames),
        "--track-n-init",
        str(args.track_n_init),
        "--track-max-iou-distance",
        str(args.track_max_iou_distance),
        "--track-max-cosine-distance",
        str(args.track_max_cosine_distance),
        "--track-nn-budget",
        str(args.track_nn_budget),
        "--quality-min",
        str(args.quality_min),
        "--quality-max",
        str(args.quality_max),
        "--match-threshold",
        str(args.pass2_match_threshold),
        "--match-topk",
        str(args.match_topk),
        "--match-min-margin",
        str(args.match_min_margin),
        "--reid-min-track-frames",
        str(args.reid_min_track_frames),
        "--reid-once-per-track",
        "--unknown-group-threshold",
        str(args.unknown_group_threshold),
        "--unknown-min-track-frames",
        str(args.unknown_min_track_frames),
        "--unknown-min-mean-magnitude",
        str(args.unknown_min_mean_magnitude),
        "--max-frames",
        str(args.max_frames),
        "--out-jsonl",
        str(outputs["pass2_jsonl"]),
        "--out-summary",
        str(outputs["pass2_summary"]),
        "--out-video",
        str(outputs["pass2_video"]),
        "--out-unknown-manifest",
        str(outputs["pass2_unknown_manifest"]),
        "--out-unknown-review-html",
        str(outputs["pass2_unknown_html"]),
        "--print-every",
        str(args.print_every),
    ]

    out_script = outputs["pass2_script"]
    out_script.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + " ".join(shlex.quote(x) for x in cmd) + "\n", encoding="utf-8")
    out_script.chmod(0o755)
    return out_script


@dataclass
class UIState:
    manifest_dir: Path
    unknown_manifest_path: Path
    gallery_manifest_path: Path
    labels_path: Path
    identity_names_path: Path
    pass2_script_path: Path


class LabelHandler(BaseHTTPRequestHandler):
    state: UIState

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _safe_rel_path(self, rel_path: str) -> Path | None:
        rel = Path(rel_path)
        if rel.is_absolute():
            return None
        candidate = (self.state.manifest_dir / rel).resolve()
        try:
            candidate.relative_to(self.state.manifest_dir.resolve())
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
            unknown_manifest = json.loads(self.state.unknown_manifest_path.read_text(encoding="utf-8"))
            labels = {}
            if self.state.labels_path.exists():
                payload = json.loads(self.state.labels_path.read_text(encoding="utf-8"))
                labels = payload.get("labels", payload)

            clusters = []
            for cluster in unknown_manifest.get("clusters", []):
                c = dict(cluster)
                sample_urls: list[str] = []
                for rel in cluster.get("samples", []):
                    rel_str = str(rel)
                    sample_urls.append("/samples/" + urllib.parse.quote(rel_str, safe="/"))
                c["sample_urls"] = sample_urls
                clusters.append(c)

            self._send_json(
                {
                    "unknown_name_prefix": unknown_manifest.get("unknown_name_prefix", "Stranger"),
                    "clusters": clusters,
                    "labels": labels,
                    "labels_path": str(self.state.labels_path),
                    "identity_names_path": str(self.state.identity_names_path),
                    "pass2_script": str(self.state.pass2_script_path),
                }
            )
            return

        if parsed.path.startswith("/samples/"):
            rel = urllib.parse.unquote(parsed.path[len("/samples/") :])
            path = self._safe_rel_path(rel)
            if path is None or not path.exists() or not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            data = path.read_bytes()
            if path.suffix.lower() in {".jpg", ".jpeg"}:
                mime = "image/jpeg"
            elif path.suffix.lower() == ".png":
                mime = "image/png"
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
        if parsed.path != "/api/save_labels":
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

        labels: dict[str, str] = {}
        for k, v in raw_labels.items():
            key = str(k).strip()
            if not key:
                continue
            value = str(v).strip()
            labels[key] = value

        self.state.labels_path.write_text(
            json.dumps({"labels": labels}, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

        unknown_manifest = json.loads(self.state.unknown_manifest_path.read_text(encoding="utf-8"))
        gallery_manifest = json.loads(self.state.gallery_manifest_path.read_text(encoding="utf-8"))
        identity_map = _build_identity_name_map(
            labels=labels,
            unknown_manifest=unknown_manifest,
            gallery_manifest=gallery_manifest,
        )

        self.state.identity_names_path.write_text(
            json.dumps({int(k): v for k, v in sorted(identity_map.items())}, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

        self._send_json(
            {
                "ok": True,
                "saved_groups": int(sum(1 for v in labels.values() if str(v).strip())),
                "identity_names_count": int(len(identity_map)),
                "labels_path": str(self.state.labels_path),
                "identity_names_path": str(self.state.identity_names_path),
                "pass2_script": str(self.state.pass2_script_path),
            }
        )

    def log_message(self, fmt: str, *args: Any) -> None:
        # Keep terminal output compact.
        print("[ui] " + (fmt % args))

    def _render_html(self) -> str:
        return """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Unknown Group Labeling</title>
  <style>
    :root {
      --bg: #f5f3ef;
      --card: #ffffff;
      --ink: #1f2937;
      --muted: #6b7280;
      --accent: #0f766e;
      --border: #d6d3d1;
    }
    body {
      margin: 0;
      font-family: "Segoe UI", "Helvetica Neue", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(900px 420px at 0% 0%, #e7f4ea 0%, transparent 58%),
        radial-gradient(900px 420px at 100% 0%, #fde68a 0%, transparent 55%),
        var(--bg);
    }
    .wrap {
      max-width: 1320px;
      margin: 0 auto;
      padding: 24px;
    }
    h1 {
      margin: 0 0 6px;
      font-size: 28px;
    }
    p {
      margin: 0 0 12px;
      color: var(--muted);
    }
    .toolbar {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-bottom: 16px;
    }
    button {
      border: 1px solid var(--border);
      border-radius: 10px;
      background: var(--card);
      color: var(--ink);
      padding: 10px 14px;
      cursor: pointer;
      font-weight: 600;
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
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
      gap: 14px;
    }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 14px;
      box-shadow: 0 8px 20px rgba(0, 0, 0, 0.05);
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
      line-height: 1.35;
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
    <h1>Unknown Group Labeling Chain</h1>
    <p>Labels are saved directly by server. Identity names JSON is generated automatically.</p>
    <div class=\"toolbar\">
      <button id=\"btnSave\">Save Labels</button>
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
      const clusters = (state && Array.isArray(state.clusters)) ? state.clusters : [];
      if (!clusters.length) {
        container.innerHTML = '<div class="empty">No unknown groups found.</div>';
        return;
      }

      const prefix = String(state.unknown_name_prefix || 'Stranger');
      const grid = document.createElement('div');
      grid.className = 'grid';

      clusters.forEach((cluster) => {
        const gid = Number(cluster.group_id);
        const key = String(gid);

        const card = document.createElement('div');
        card.className = 'card';

        const title = document.createElement('div');
        title.className = 'title';
        const left = document.createElement('strong');
        left.textContent = `${prefix}-${gid}`;
        const right = document.createElement('span');
        right.style.color = '#6b7280';
        right.style.fontSize = '12px';
        right.textContent = `tracks=${Number(cluster.num_tracks || 0)}`;
        title.appendChild(left);
        title.appendChild(right);

        const stats = document.createElement('div');
        stats.className = 'stats';
        const sim = cluster.max_similarity_to_group;
        const simText = sim == null ? 'n/a' : Number(sim).toFixed(3);
        const avgMag = Number(cluster.avg_track_magnitude || 0).toFixed(2);
        stats.textContent = `frames=${cluster.first_frame}..${cluster.last_frame} | avgMag=${avgMag} | maxGroupSim=${simText}`;

        const samples = document.createElement('div');
        samples.className = 'samples';
        const urls = Array.isArray(cluster.sample_urls) ? cluster.sample_urls : [];
        urls.forEach((u) => {
          const img = document.createElement('img');
          img.loading = 'lazy';
          img.src = u;
          img.alt = `${prefix}-${gid}`;
          samples.appendChild(img);
        });

        const label = document.createElement('label');
        label.textContent = 'Manual label';
        const input = document.createElement('input');
        input.type = 'text';
        input.placeholder = 'Example: Sarah';
        input.value = String(labels[key] || '');
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
      setStatus('Loading...');
      const res = await fetch('/api/data');
      state = await res.json();
      labels = Object.assign({}, state.labels || {});
      render();
      setStatus(
        `labels_json: ${state.labels_path}\n` +
        `identity_names_json: ${state.identity_names_path}\n` +
        `pass2_script: ${state.pass2_script}`
      );
    }

    async function saveData() {
      setStatus('Saving labels and generating identity map...');
      const res = await fetch('/api/save_labels', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({labels})
      });
      const out = await res.json();
      if (!out.ok) {
        setStatus(`Save failed: ${JSON.stringify(out)}`);
        return;
      }
      setStatus(
        `Saved groups: ${out.saved_groups}\n` +
        `Identity names count: ${out.identity_names_count}\n` +
        `labels_json: ${out.labels_path}\n` +
        `identity_names_json: ${out.identity_names_path}\n` +
        `pass2_script: ${out.pass2_script}`
      );
    }

    document.getElementById('btnSave').addEventListener('click', saveData);
    document.getElementById('btnReload').addEventListener('click', loadData);
    loadData();
  </script>
</body>
</html>
"""


def _build_outputs(logs_dir: Path, base_name: str) -> dict[str, Path]:
    base = base_name.strip()
    return {
        "pass1_jsonl": logs_dir / f"{base}_pass1.jsonl",
        "pass1_summary": logs_dir / f"{base}_pass1.summary.json",
        "gallery_npz": logs_dir / f"{base}_gallery.npz",
        "gallery_manifest": logs_dir / f"{base}_gallery.manifest.json",
        "unknown_manifest": logs_dir / f"{base}_unknown_groups.json",
        "legacy_unknown_html": logs_dir / f"{base}_unknown_groups.static.html",
        "labels_json": logs_dir / f"{base}_group_labels.json",
        "identity_names_json": logs_dir / f"{base}_identity_names.json",
        "pass2_script": logs_dir / f"{base}_run_pass2.sh",
        "pass2_jsonl": logs_dir / f"{base}_pass2.jsonl",
        "pass2_summary": logs_dir / f"{base}_pass2.summary.json",
        "pass2_video": logs_dir / f"{base}_pass2.mp4",
        "pass2_unknown_manifest": logs_dir / f"{base}_pass2_unknown_groups.json",
        "pass2_unknown_html": logs_dir / f"{base}_pass2_unknown_groups.html",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="One-command stranger-labeling chain")
    parser.add_argument("--config", default="configs/train_ms1m_magface_phase1_cplus_aplus_v1.yaml")
    parser.add_argument("--checkpoint", default="latest", help="latest/best/swa or explicit checkpoint path")
    parser.add_argument("--source", required=True)
    parser.add_argument("--detector-model", required=True)
    parser.add_argument("--base-name", default="pipeline_label_chain")
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--python-bin", default=str(_default_python_bin()))

    parser.add_argument("--det-conf", type=float, default=0.08)
    parser.add_argument("--det-iou", type=float, default=0.45)
    parser.add_argument("--det-imgsz", type=int, default=1280)
    parser.add_argument("--det-rescue-conf", type=float, default=0.05)
    parser.add_argument("--det-rescue-imgsz", type=int, default=1920)
    parser.add_argument("--det-rescue-min-primary", type=int, default=2)

    parser.add_argument("--track-max-missed-frames", type=int, default=140)
    parser.add_argument("--track-n-init", type=int, default=2)
    parser.add_argument("--track-max-iou-distance", type=float, default=0.9)
    parser.add_argument("--track-max-cosine-distance", type=float, default=0.42)
    parser.add_argument("--track-nn-budget", type=int, default=200)

    parser.add_argument("--quality-min", type=float, default=10.0)
    parser.add_argument("--quality-max", type=float, default=110.0)
    parser.add_argument("--match-topk", type=int, default=7)
    parser.add_argument("--match-min-margin", type=float, default=0.12)
    parser.add_argument("--pass2-match-threshold", type=float, default=0.46)
    parser.add_argument("--reid-min-track-frames", type=int, default=6)

    parser.add_argument("--unknown-group-threshold", type=float, default=0.72)
    parser.add_argument("--unknown-min-track-frames", type=int, default=6)
    parser.add_argument("--unknown-min-mean-magnitude", type=float, default=11.0)
    parser.add_argument("--unknown-max-samples-per-group", type=int, default=12)
    parser.add_argument("--unknown-sample-min-gap", type=int, default=10)

    parser.add_argument("--gallery-min-track-frames", type=int, default=6)
    parser.add_argument("--gallery-dedupe-threshold", type=float, default=0.72)
    parser.add_argument("--gallery-min-mean-magnitude", type=float, default=11.0)

    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=60)

    parser.add_argument("--run-pass1", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--unknown-manifest-path", default="", help="Optional existing unknown manifest when --no-run-pass1")
    parser.add_argument("--gallery-manifest-path", default="", help="Optional existing gallery manifest when --no-run-pass1")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open-browser", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    args.config = str(config_path)

    checkpoint_path = _resolve_checkpoint(config_path, args.checkpoint)
    args.checkpoint = str(checkpoint_path)

    source = Path(args.source)
    if not source.is_absolute():
        source = (PROJECT_ROOT / source).resolve()
    if not source.exists():
        raise FileNotFoundError(f"source not found: {source}")
    args.source = str(source)

    detector_model = Path(args.detector_model)
    if not detector_model.is_absolute():
        detector_model = (PROJECT_ROOT / detector_model).resolve()
    if not detector_model.exists():
        raise FileNotFoundError(f"detector model not found: {detector_model}")
    args.detector_model = str(detector_model)

    logs_dir = Path(args.logs_dir)
    if not logs_dir.is_absolute():
        logs_dir = (PROJECT_ROOT / logs_dir).resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)

    outputs = _build_outputs(logs_dir=logs_dir, base_name=args.base_name)

    python_bin = Path(args.python_bin).expanduser()
    if not python_bin.is_absolute():
        python_bin = (PROJECT_ROOT / python_bin)
    if not python_bin.exists():
        raise FileNotFoundError(f"python bin not found: {python_bin}")
    args.python_bin = str(python_bin)

    if args.unknown_manifest_path:
        p = Path(args.unknown_manifest_path)
        if not p.is_absolute():
            p = (PROJECT_ROOT / p).resolve()
        outputs["unknown_manifest"] = p

    if args.gallery_manifest_path:
        p = Path(args.gallery_manifest_path)
        if not p.is_absolute():
            p = (PROJECT_ROOT / p).resolve()
        outputs["gallery_manifest"] = p

    if bool(args.run_pass1):
        cmd = _build_pass1_command(args=args, outputs=outputs)
        print("[chain] running pass1 build+dedupe...", flush=True)
        print("[chain] command:")
        print(" ".join(shlex.quote(x) for x in cmd))
        subprocess.run(cmd, check=True)

    if not outputs["unknown_manifest"].exists():
        raise FileNotFoundError(f"unknown manifest not found: {outputs['unknown_manifest']}")
    if not outputs["gallery_manifest"].exists():
        raise FileNotFoundError(f"gallery manifest not found: {outputs['gallery_manifest']}")

    pass2_script = _write_pass2_command(args=args, outputs=outputs, identity_names_json=outputs["identity_names_json"])

    state = UIState(
        manifest_dir=outputs["unknown_manifest"].parent.resolve(),
        unknown_manifest_path=outputs["unknown_manifest"].resolve(),
        gallery_manifest_path=outputs["gallery_manifest"].resolve(),
        labels_path=outputs["labels_json"].resolve(),
        identity_names_path=outputs["identity_names_json"].resolve(),
        pass2_script_path=pass2_script.resolve(),
    )

    def _handler(*h_args, **h_kwargs):
        LabelHandler.state = state
        return LabelHandler(*h_args, **h_kwargs)

    server = ThreadingHTTPServer((args.host, int(args.port)), _handler)
    url = f"http://{args.host}:{int(args.port)}/"

    print("[chain] ready")
    print(f"[chain] labeling_ui={url}")
    print(f"[chain] labels_json={state.labels_path}")
    print(f"[chain] identity_names_json={state.identity_names_path}")
    print(f"[chain] run_pass2_script={state.pass2_script_path}")

    if bool(args.open_browser):
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[chain] stopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
