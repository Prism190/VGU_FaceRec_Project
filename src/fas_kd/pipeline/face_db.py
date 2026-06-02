from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_IDENTITY_DIR_RE = re.compile(r"^id_(\d+)(?:__(.*))?$")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _normalize_embedding(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm < eps:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr / norm).astype(np.float32)


def _slugify(text: str, max_len: int = 60) -> str:
    lowered = text.strip().lower()
    cleaned = re.sub(r"[^a-z0-9]+", "_", lowered)
    cleaned = cleaned.strip("_")
    if not cleaned:
        cleaned = "person"
    return cleaned[:max_len]


def _safe_read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _count_identity_photos(identity_dir: Path) -> int:
    photos_dir = identity_dir / "photos"
    if not photos_dir.exists():
        return 0
    return int(
        sum(
            1
            for p in photos_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
    )


def _parse_identity_id_from_dir_name(name: str) -> int | None:
    match = _IDENTITY_DIR_RE.match(name)
    if match is None:
        return None
    return int(match.group(1))


def _identity_dir_for_id(identities_dir: Path, identity_id: int) -> Path | None:
    prefix = f"id_{int(identity_id):06d}"
    candidates = sorted(p for p in identities_dir.glob(f"{prefix}*") if p.is_dir())
    return candidates[0] if candidates else None


def _load_identity_embeddings(identity_dir: Path, expected_dim: int) -> np.ndarray:
    emb_path = identity_dir / "embeddings.npz"
    if not emb_path.exists():
        return np.zeros((0, expected_dim), dtype=np.float32)

    with np.load(emb_path) as data:
        if "embeddings" not in data:
            return np.zeros((0, expected_dim), dtype=np.float32)
        arr = np.asarray(data["embeddings"], dtype=np.float32)

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        return np.zeros((0, expected_dim), dtype=np.float32)
    if arr.shape[1] != expected_dim:
        return np.zeros((0, expected_dim), dtype=np.float32)
    return arr


@dataclass
class KnownIdentityRecord:
    identity_id: int
    name: str
    identity_dir: Path
    embedding_count: int
    photo_count: int
    prototype_embedding: np.ndarray | None


def ensure_face_db_layout(db_root: Path) -> dict[str, Path]:
    root = Path(db_root).resolve()
    known_root = root / "known"
    known_identities = known_root / "identities"
    strangers_root = root / "strangers"
    stranger_sessions = strangers_root / "sessions"

    known_identities.mkdir(parents=True, exist_ok=True)
    stranger_sessions.mkdir(parents=True, exist_ok=True)

    readme_path = root / "README.txt"
    if not readme_path.exists():
        readme_path.write_text(
            "Face database layout\n"
            "- known/identities/id_XXXXXX__name: per-identity photos + embeddings\n"
            "- strangers/sessions/<session_name>: grouped stranger samples + pooled embeddings\n",
            encoding="utf-8",
        )

    return {
        "root": root,
        "known_root": known_root,
        "known_identities": known_identities,
        "strangers_root": strangers_root,
        "stranger_sessions": stranger_sessions,
    }


def _load_known_identity_records(db_root: Path, expected_dim: int) -> list[KnownIdentityRecord]:
    layout = ensure_face_db_layout(db_root)
    identities_dir = layout["known_identities"]
    out: list[KnownIdentityRecord] = []

    for identity_dir in sorted(p for p in identities_dir.iterdir() if p.is_dir()):
        meta = _safe_read_json(identity_dir / "meta.json")

        raw_id = meta.get("identity_id")
        if raw_id is None:
            raw_id = _parse_identity_id_from_dir_name(identity_dir.name)
        if raw_id is None:
            continue

        identity_id = int(raw_id)
        name = str(meta.get("name") or f"Person-{identity_id}").strip() or f"Person-{identity_id}"

        embeddings = _load_identity_embeddings(identity_dir=identity_dir, expected_dim=expected_dim)
        if embeddings.shape[0] > 0:
            normalized_rows = np.stack([_normalize_embedding(row) for row in embeddings], axis=0)
            proto = _normalize_embedding(np.mean(normalized_rows, axis=0))
        else:
            proto = None

        out.append(
            KnownIdentityRecord(
                identity_id=identity_id,
                name=name,
                identity_dir=identity_dir,
                embedding_count=int(embeddings.shape[0]),
                photo_count=_count_identity_photos(identity_dir),
                prototype_embedding=proto,
            )
        )

    out.sort(key=lambda item: item.identity_id)
    return out


def load_known_face_gallery(
    db_root: Path,
    *,
    expected_dim: int,
    retrieval_mode: str = "pooled",
) -> tuple[np.ndarray, np.ndarray, dict[int, str], dict[str, int]]:
    mode = str(retrieval_mode).strip().lower()
    if mode not in {"pooled", "all"}:
        raise ValueError(f"Unsupported retrieval_mode: {retrieval_mode}")

    records = _load_known_identity_records(db_root=db_root, expected_dim=expected_dim)

    vectors: list[np.ndarray] = []
    ids: list[int] = []
    name_map: dict[int, str] = {}

    for rec in records:
        name_map[int(rec.identity_id)] = str(rec.name)

        if mode == "pooled":
            if rec.prototype_embedding is not None:
                vectors.append(np.asarray(rec.prototype_embedding, dtype=np.float32))
                ids.append(int(rec.identity_id))
            continue

        embeddings = _load_identity_embeddings(identity_dir=rec.identity_dir, expected_dim=expected_dim)
        for row in embeddings:
            vectors.append(_normalize_embedding(row))
            ids.append(int(rec.identity_id))

    if vectors:
        emb_arr = np.stack(vectors, axis=0).astype(np.float32)
    else:
        emb_arr = np.zeros((0, expected_dim), dtype=np.float32)

    id_arr = np.asarray(ids, dtype=np.int64)
    stats = {
        "identities_total": int(len(records)),
        "vectors_loaded": int(emb_arr.shape[0]),
        "identities_with_embeddings": int(sum(1 for rec in records if rec.embedding_count > 0)),
        "photos_total": int(sum(rec.photo_count for rec in records)),
    }
    return emb_arr, id_arr, name_map, stats


def append_known_identity(
    db_root: Path,
    *,
    identity_name: str,
    embeddings: np.ndarray,
    sample_image_paths: Iterable[Path] | None = None,
    identity_id: int | None = None,
    min_identity_id: int = 1000,
) -> dict[str, object]:
    name = str(identity_name).strip()
    if not name:
        raise ValueError("identity_name must be non-empty")

    emb = np.asarray(embeddings, dtype=np.float32)
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)
    if emb.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape {emb.shape}")

    emb = np.stack([_normalize_embedding(row) for row in emb], axis=0).astype(np.float32)
    dim = int(emb.shape[1])

    layout = ensure_face_db_layout(db_root)
    identities_dir = layout["known_identities"]

    records = _load_known_identity_records(db_root=db_root, expected_dim=dim)

    target_record: KnownIdentityRecord | None = None
    if identity_id is not None:
        for rec in records:
            if int(rec.identity_id) == int(identity_id):
                target_record = rec
                break

    if target_record is None:
        wanted_key = name.lower()
        for rec in records:
            if rec.name.lower() == wanted_key:
                target_record = rec
                break

    if target_record is None:
        used_ids = {int(rec.identity_id) for rec in records}
        new_id = max(max(used_ids) + 1 if used_ids else int(min_identity_id), int(min_identity_id))
        if identity_id is not None:
            new_id = int(identity_id)
        existing_dir = _identity_dir_for_id(identities_dir=identities_dir, identity_id=new_id)
        if existing_dir is None:
            identity_dir = identities_dir / f"id_{new_id:06d}__{_slugify(name)}"
            identity_dir.mkdir(parents=True, exist_ok=True)
        else:
            identity_dir = existing_dir
        resolved_id = int(new_id)
    else:
        identity_dir = target_record.identity_dir
        resolved_id = int(target_record.identity_id)

    old_meta = _safe_read_json(identity_dir / "meta.json")

    old_embeddings = _load_identity_embeddings(identity_dir=identity_dir, expected_dim=dim)
    if old_embeddings.shape[0] > 0:
        combined = np.concatenate([old_embeddings, emb], axis=0)
    else:
        combined = emb

    np.savez(identity_dir / "embeddings.npz", embeddings=combined.astype(np.float32))

    photos_dir = identity_dir / "photos"
    photos_dir.mkdir(parents=True, exist_ok=True)
    copied_photos = 0
    if sample_image_paths is not None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        for idx, raw_src in enumerate(sample_image_paths):
            src = Path(raw_src)
            if not src.exists() or not src.is_file():
                continue
            ext = src.suffix.lower()
            if ext not in IMAGE_EXTENSIONS:
                continue
            stem = _slugify(src.stem, max_len=24)
            dst = photos_dir / f"{ts}_{idx:03d}_{stem}{ext}"
            suffix_idx = 1
            while dst.exists():
                dst = photos_dir / f"{ts}_{idx:03d}_{stem}_{suffix_idx:02d}{ext}"
                suffix_idx += 1
            shutil.copy2(src, dst)
            copied_photos += 1

    created_at = old_meta.get("created_at") if isinstance(old_meta.get("created_at"), str) else _utc_now_iso()
    updated_at = _utc_now_iso()

    meta = {
        "identity_id": int(resolved_id),
        "name": name,
        "embedding_count": int(combined.shape[0]),
        "photo_count": _count_identity_photos(identity_dir),
        "created_at": created_at,
        "updated_at": updated_at,
    }
    _write_json(identity_dir / "meta.json", meta)

    return {
        "identity_id": int(resolved_id),
        "identity_dir": identity_dir,
        "added_embeddings": int(emb.shape[0]),
        "total_embeddings": int(combined.shape[0]),
        "copied_photos": int(copied_photos),
    }


def persist_stranger_session(
    db_root: Path,
    *,
    session_name: str,
    unknown_manifest: dict,
    unknown_manifest_parent: Path,
    group_embeddings: dict[int, np.ndarray],
) -> Path:
    layout = ensure_face_db_layout(db_root)
    sessions_dir = layout["stranger_sessions"]

    base = _slugify(session_name, max_len=80)
    if not base:
        base = datetime.now(timezone.utc).strftime("session_%Y%m%dT%H%M%SZ")

    session_dir = sessions_dir / base
    idx = 1
    while session_dir.exists():
        session_dir = sessions_dir / f"{base}_{idx:02d}"
        idx += 1

    groups_root = session_dir / "groups"
    groups_root.mkdir(parents=True, exist_ok=True)

    exported_groups: list[dict] = []
    emb_ids: list[int] = []
    emb_rows: list[np.ndarray] = []

    for cluster in unknown_manifest.get("clusters", []):
        try:
            group_id = int(cluster.get("group_id"))
        except Exception:
            continue

        group_dir = groups_root / f"group_{group_id:04d}"
        samples_dir = group_dir / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)

        copied_samples: list[str] = []
        for sample_idx, raw_sample in enumerate(cluster.get("samples", [])):
            src = Path(str(raw_sample))
            if not src.is_absolute():
                src = (unknown_manifest_parent / src).resolve()
            if not src.exists() or not src.is_file():
                continue

            ext = src.suffix.lower() or ".jpg"
            stem = _slugify(src.stem, max_len=32)
            dst = samples_dir / f"{sample_idx:03d}_{stem}{ext}"
            suffix_idx = 1
            while dst.exists():
                dst = samples_dir / f"{sample_idx:03d}_{stem}_{suffix_idx:02d}{ext}"
                suffix_idx += 1
            shutil.copy2(src, dst)
            copied_samples.append(dst.relative_to(session_dir).as_posix())

        exported_groups.append(
            {
                "group_id": int(group_id),
                "name": str(cluster.get("name") or f"Stranger-{group_id}"),
                "num_tracks": int(cluster.get("num_tracks") or 0),
                "track_ids": [int(tid) for tid in cluster.get("track_ids", [])],
                "first_frame": int(cluster.get("first_frame") or 0),
                "last_frame": int(cluster.get("last_frame") or 0),
                "avg_track_magnitude": float(cluster.get("avg_track_magnitude") or 0.0),
                "max_similarity_to_group": cluster.get("max_similarity_to_group"),
                "samples": copied_samples,
                "promoted": False,
                "promoted_identity_id": None,
                "promoted_identity_name": None,
                "promoted_at": None,
            }
        )

        vec = group_embeddings.get(int(group_id))
        if vec is None:
            continue
        norm_vec = _normalize_embedding(np.asarray(vec, dtype=np.float32))
        if norm_vec.size == 0 or not np.isfinite(norm_vec).all():
            continue
        emb_ids.append(int(group_id))
        emb_rows.append(norm_vec)

    if emb_rows:
        emb_arr = np.stack(emb_rows, axis=0).astype(np.float32)
    else:
        emb_arr = np.zeros((0, 0), dtype=np.float32)

    emb_file = session_dir / "group_embeddings.npz"
    np.savez(
        emb_file,
        group_ids=np.asarray(emb_ids, dtype=np.int64),
        embeddings=emb_arr,
    )

    manifest = {
        "session_name": session_dir.name,
        "created_at": _utc_now_iso(),
        "source_unknown_manifest": str(unknown_manifest.get("manifest_path") or ""),
        "source_unknown_name_prefix": str(unknown_manifest.get("unknown_name_prefix") or "Stranger"),
        "num_groups": int(len(exported_groups)),
        "embeddings_file": emb_file.name,
        "groups": exported_groups,
    }
    _write_json(session_dir / "manifest.json", manifest)
    return session_dir


def load_session_group_embeddings(session_dir: Path, embeddings_file: str) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    emb_path = session_dir / embeddings_file
    if not emb_path.exists():
        return out

    with np.load(emb_path) as data:
        if "group_ids" not in data or "embeddings" not in data:
            return out
        group_ids = np.asarray(data["group_ids"], dtype=np.int64).reshape(-1)
        embeddings = np.asarray(data["embeddings"], dtype=np.float32)

    if embeddings.ndim != 2:
        return out
    if group_ids.shape[0] != embeddings.shape[0]:
        return out

    for gid, row in zip(group_ids, embeddings):
        out[int(gid)] = _normalize_embedding(row)
    return out


def reset_face_db(db_root: Path) -> dict[str, Path]:
    root = Path(db_root).resolve()
    if root.exists():
        shutil.rmtree(root)
    return ensure_face_db_layout(root)


def iter_image_files(root: Path) -> list[Path]:
    base = Path(root)
    if not base.exists():
        return []
    return sorted(
        p
        for p in base.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
