
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import json
import numpy as np

@dataclass
class PairData:
    pair_id: int
    area: int
    red_area: int
    blue_area: int
    runs_red: list[list[int]]  # [y, x0, x1)
    runs_blue: list[list[int]]


def _runs_to_coords(runs: list[list[int]]) -> np.ndarray:
    """Expand RLE runs to Nx2 coords (y,x), sorted by (y,x)."""
    coords = []
    for y, x0, x1 in runs:
        # x range [x0, x1)
        # append (y,x)
        coords.extend([(y, x) for x in range(int(x0), int(x1))])
    if not coords:
        return np.zeros((0,2), dtype=np.int16)
    arr = np.array(coords, dtype=np.int16)
    # already in row-major order by construction
    return arr


def _load_pairs_json(p: Path) -> list[PairData]:
    doc = json.loads(p.read_text(encoding="utf-8"))
    pairs = []
    for it in doc.get("pairs", []):
        pairs.append(PairData(
            pair_id=int(it["pair_id"]),
            area=int(it.get("area", 0)),
            red_area=int(it.get("side_red_area", 0)),
            blue_area=int(it.get("side_blue_area", 0)),
            runs_red=it.get("runs_red") or [],
            runs_blue=it.get("runs_blue") or [],
        ))
    # stable order: largest first
    pairs.sort(key=lambda x: (-x.area, x.pair_id))
    return pairs


def resolve_pairs_file(templates_dir: Path, blueprint: str) -> Optional[Path]:
    """Find Pairs_Dino_Parts/<Base>/<Blueprint>.pairs.lite.json (preferred) or raw."""
    root = templates_dir / "Pairs_Dino_Parts"
    if not root.exists():
        return None

    # common layout: Pairs_Dino_Parts/<Base>/<Blueprint>.pairs.lite.json
    # We don't know base_name reliably, so search.
    lite_name = f"{blueprint}.pairs.lite.json"
    raw_name  = f"{blueprint}.pairs.raw.json"

    # Prefer lite if found
    matches_lite = list(root.rglob(lite_name))
    if matches_lite:
        return matches_lite[0]
    matches_raw = list(root.rglob(raw_name))
    if matches_raw:
        return matches_raw[0]
    return None


def apply_pairs_symmetry(
    image_rgba: np.ndarray,
    *,
    pairs: list[PairData],
    visibility_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Deterministic symmetry: for each pair, choose canonical side = side with larger area,
    then project source side pixels onto target side by index-resampling.
    """
    h, w, c = image_rgba.shape
    out = image_rgba.copy()

    total_changed = 0
    total_pairs_used = 0

    # cache expanded coords per pair_id per side to avoid recompute
    # local dict: pair_id -> (coords_red, coords_blue)
    coords_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def filter_mask(coords: np.ndarray) -> np.ndarray:
        if visibility_mask is None or coords.size == 0:
            return coords
        # visibility_mask is h×w bool
        ys = coords[:,0].astype(np.int32)
        xs = coords[:,1].astype(np.int32)
        keep = visibility_mask[ys, xs]
        return coords[keep]

    for p in pairs:
        if not p.runs_red or not p.runs_blue:
            continue
        if p.area <= 0:
            continue

        if p.pair_id not in coords_cache:
            cr = _runs_to_coords(p.runs_red)
            cb = _runs_to_coords(p.runs_blue)
            coords_cache[p.pair_id] = (cr, cb)
        cr, cb = coords_cache[p.pair_id]
        cr = filter_mask(cr)
        cb = filter_mask(cb)
        if cr.size == 0 or cb.size == 0:
            continue

        # choose canonical side by area
        if p.red_area >= p.blue_area:
            src = cr
            dst = cb
        else:
            src = cb
            dst = cr

        n_src = len(src)
        n_dst = len(dst)
        if n_src == 0 or n_dst == 0:
            continue

        # take snapshot of source pixels from original (not from out, to avoid feedback)
        src_pixels = image_rgba[src[:,0], src[:,1], :]

        # resample source indices across destination
        # idx_src = floor(i * n_src / n_dst)
        idx_src = (np.arange(n_dst, dtype=np.int64) * n_src) // n_dst
        mapped = src_pixels[idx_src]

        # count changed pixels (cheap compare)
        before = out[dst[:,0], dst[:,1], :]
        changed = np.any(before != mapped, axis=1)
        total_changed += int(changed.sum())

        out[dst[:,0], dst[:,1], :] = mapped
        total_pairs_used += 1

    report = {
        "pairs_used": total_pairs_used,
        "pixels_changed": total_changed,
    }
    return out, report


def try_apply_dino_pairs_symmetry(
    image_rgba: np.ndarray,
    *,
    blueprint: Optional[str],
    templates_dir: Path,
    visibility_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Auto: if a pairs file exists for blueprint, apply symmetry.
    No GUI options. Deterministic.
    """
    report: Dict[str, Any] = {"applied": False}
    if blueprint is None:
        return image_rgba, report

    pfile = resolve_pairs_file(templates_dir, blueprint)
    if pfile is None:
        return image_rgba, report

    try:
        pairs = _load_pairs_json(pfile)
    except Exception as e:
        report.update({"error": f"failed_to_load_pairs: {e}"})
        return image_rgba, report

    out, r = apply_pairs_symmetry(image_rgba, pairs=pairs, visibility_mask=visibility_mask)
    report.update({"applied": True, "pairs_file": str(pfile), **r})
    return out, report
