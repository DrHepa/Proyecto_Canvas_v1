from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
from PIL import Image


def _rgb_key(rgb: Tuple[int, int, int]) -> str:
    return f"{rgb[0]},{rgb[1]},{rgb[2]}"


def _load_rgb_png(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGBA")
    arr = np.array(img, dtype=np.uint8)
    return arr


def _runs_from_bool(mask: np.ndarray) -> List[List[int]]:
    h, w = mask.shape
    out: List[List[int]] = []
    for y in range(h):
        row = mask[y]
        x = 0
        while x < w:
            if not row[x]:
                x += 1
                continue
            x0 = x
            x += 1
            while x < w and row[x]:
                x += 1
            out.append([int(y), int(x0), int(x)])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Build pairs.raw.json + pairs.lite.json from mask PNGs.")
    ap.add_argument("--blueprint", required=True, help="Blueprint, ej. Raptor_Character_BP_C")
    ap.add_argument("--mask", required=True, type=str, help="mask.png (paintable alpha>0 o negro)")
    ap.add_argument("--mask_p", required=True, type=str, help="mask_P.png (RGB pair ids)")
    ap.add_argument("--mask_s", required=True, type=str, help="mask_S.png (side: red/blue)")
    ap.add_argument("--out_dir", required=True, type=str, help="Output dir")
    ap.add_argument("--side_red", default="255,0,0", help="RGB for 'red side' in mask_S")
    ap.add_argument("--side_blue", default="0,0,255", help="RGB for 'blue side' in mask_S")
    args = ap.parse_args()

    blueprint = args.blueprint.strip()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    side_red = tuple(int(x) for x in args.side_red.split(","))
    side_blue = tuple(int(x) for x in args.side_blue.split(","))

    mask = _load_rgb_png(Path(args.mask))
    mask_p = _load_rgb_png(Path(args.mask_p))
    mask_s = _load_rgb_png(Path(args.mask_s))

    if mask.shape[:2] != mask_p.shape[:2] or mask.shape[:2] != mask_s.shape[:2]:
        raise SystemExit("Resoluciones no coinciden entre mask/mask_P/mask_S.")

    h, w = mask.shape[:2]

    paintable = mask[..., 3] > 0
    # If mask is encoded as black on transparent, still alpha works. If user provides opaque, allow black pixels:
    if not paintable.any():
        paintable = (mask[..., 0:3].sum(axis=2) < 30)

    # Side detection: exact RGB match + alpha>0
    s_rgb = mask_s[..., 0:3]
    s_a = mask_s[..., 3] > 0
    is_red = s_a & (s_rgb[..., 0] == side_red[0]) & (s_rgb[..., 1] == side_red[1]) & (s_rgb[..., 2] == side_red[2])
    is_blue = s_a & (s_rgb[..., 0] == side_blue[0]) & (s_rgb[..., 1] == side_blue[1]) & (s_rgb[..., 2] == side_blue[2])

    # Pair id = exact RGB of mask_P + alpha>0
    p_rgb = mask_p[..., 0:3]
    p_a = mask_p[..., 3] > 0

    active = paintable & p_a
    # Orphans: active but no side
    orphan_mask = active & ~(is_red | is_blue)

    pairs: Dict[str, Dict[str, Any]] = {}

    # Iterate pixels (256x256 typical); deterministic scanning
    for y in range(h):
        for x in range(w):
            if not active[y, x]:
                continue
            k = _rgb_key(tuple(int(v) for v in p_rgb[y, x]))
            slot = pairs.get(k)
            if slot is None:
                slot = {
                    "pair_key": k,
                    "pixels_red": [],
                    "pixels_blue": [],
                }
                pairs[k] = slot

            if is_red[y, x]:
                slot["pixels_red"].append([int(x), int(y)])
            elif is_blue[y, x]:
                slot["pixels_blue"].append([int(x), int(y)])
            else:
                # orphan handled separately
                pass

    # Build lite structure
    lite_pairs: List[Dict[str, Any]] = []
    for k in sorted(pairs.keys()):
        slot = pairs[k]
        red_px = slot["pixels_red"]
        blue_px = slot["pixels_blue"]
        area_red = len(red_px)
        area_blue = len(blue_px)
        area = area_red + area_blue

        # bbox + centroid computed over both sides
        all_px = red_px + blue_px
        if all_px:
            xs = [p[0] for p in all_px]
            ys = [p[1] for p in all_px]
            bbox = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
            cx = float(sum(xs)) / float(len(xs))
            cy = float(sum(ys)) / float(len(ys))
        else:
            bbox = [0, 0, 0, 0]
            cx, cy = 0.0, 0.0

        # runs by side
        red_mask = np.zeros((h, w), dtype=bool)
        blue_mask = np.zeros((h, w), dtype=bool)
        for x, y in red_px:
            red_mask[y, x] = True
        for x, y in blue_px:
            blue_mask[y, x] = True

        lite_pairs.append({
            "pair_key": k,
            "area": int(area),
            "area_red": int(area_red),
            "area_blue": int(area_blue),
            "bbox": bbox,
            "centroid": [cx, cy],
            "runs_red": _runs_from_bool(red_mask),
            "runs_blue": _runs_from_bool(blue_mask),
            "members": [k],
        })

    # Sort by area desc (useful)
    lite_pairs.sort(key=lambda d: int(d.get("area", 0)), reverse=True)

    # Orphans list as unique RGB keys
    orphan_keys = set()
    oy, ox = np.where(orphan_mask)
    for y, x in zip(oy.tolist(), ox.tolist()):
        orphan_keys.add(_rgb_key(tuple(int(v) for v in p_rgb[y, x])))
    orphans = sorted(orphan_keys)

    lite = {
        "blueprint": blueprint,
        "version": 1,
        "resolution": [int(w), int(h)],
        "sides": {"red": list(side_red), "blue": list(side_blue)},
        "pairs": lite_pairs,
        "orphans": orphans,
    }

    raw = {
        "blueprint": blueprint,
        "version": 1,
        "resolution": [int(w), int(h)],
        "note": "Raw includes pixel lists per side (for debugging).",
        "pairs": pairs,
        "orphans": orphans,
    }

    (out_dir / f"{blueprint}.pairs.lite.json").write_text(json.dumps(lite, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / f"{blueprint}.pairs.raw.json").write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")

    # Pair color table: rgb -> pair_key (identity, but useful for tooling)
    table = {p["pair_key"]: p["pair_key"] for p in lite_pairs}
    (out_dir / f"{blueprint}.pair_color_table.json").write_text(json.dumps(table, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"OK: {blueprint} pairs={len(lite_pairs)} orphans={len(orphans)} -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
