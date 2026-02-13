from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
from PIL import Image

from MaskExtractor import extract_mask_from_pnt


def _parse_size(s: str) -> Tuple[Optional[int], Optional[int]]:
    s = (s or "").strip().lower().replace(" ", "")
    if not s:
        return None, None
    if "x" not in s:
        raise ValueError("Size debe ser WxH, ej. 256x256")
    a, b = s.split("x", 1)
    return int(a), int(b)


def _connected_components(mask: np.ndarray) -> List[dict]:
    # 4-neighborhood components on boolean mask; returns bbox+area
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or seen[y, x]:
                continue
            stack = [(y, x)]
            seen[y, x] = True
            area = 0
            x0 = x1 = x
            y0 = y1 = y
            while stack:
                cy, cx = stack.pop()
                area += 1
                x0 = min(x0, cx); x1 = max(x1, cx)
                y0 = min(y0, cy); y1 = max(y1, cy)
                for ny, nx in ((cy-1,cx),(cy+1,cx),(cy,cx-1),(cy,cx+1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            comps.append({"area": int(area), "bbox": [int(x0), int(y0), int(x1), int(y1)]})
    comps.sort(key=lambda c: c["area"], reverse=True)
    return comps


def main() -> int:
    ap = argparse.ArgumentParser(description="Diff mask PNG vs mask inferred from .pnt (debug).")
    ap.add_argument("pnt", type=str, help=".pnt")
    ap.add_argument("png", type=str, help="mask.png (alpha)")
    ap.add_argument("--size", type=str, default="", help="Para ASA GUID-header: WxH, ej. 256x256")
    ap.add_argument("--out_png", type=str, default="diff.png", help="Output diff image")
    ap.add_argument("--out_json", type=str, default="diff_regions.json", help="Output regions json")
    args = ap.parse_args()

    w, h = _parse_size(args.size)

    # pnt mask (alpha)
    img_pnt, info = extract_mask_from_pnt(Path(args.pnt), width=w, height=h)
    pnt_a = np.array(img_pnt, dtype=np.uint8)[..., 3] > 0

    # png mask (alpha)
    png_a = np.array(Image.open(args.png).convert("RGBA"), dtype=np.uint8)[..., 3] > 0

    if pnt_a.shape != png_a.shape:
        raise SystemExit(f"Shape mismatch: pnt {pnt_a.shape} vs png {png_a.shape}")

    only_png = png_a & ~pnt_a
    only_pnt = pnt_a & ~png_a
    both = png_a & pnt_a

    # Diff visualization:
    #   - only_png: red
    #   - only_pnt: blue
    #   - both: green
    out = np.zeros((pnt_a.shape[0], pnt_a.shape[1], 4), dtype=np.uint8)
    out[..., 3] = 255
    out[only_png] = [255, 0, 0, 255]
    out[only_pnt] = [0, 0, 255, 255]
    out[both] = [0, 255, 0, 255]

    Image.fromarray(out, mode="RGBA").save(args.out_png)

    comps_png = _connected_components(only_png)
    comps_pnt = _connected_components(only_pnt)

    report = {
        "pnt": str(args.pnt),
        "png": str(args.png),
        "pnt_kind": info.kind,
        "pnt_bg": info.background_value,
        "counts": {
            "only_png": int(only_png.sum()),
            "only_pnt": int(only_pnt.sum()),
            "both": int(both.sum()),
        },
        "regions": {
            "only_png": comps_png[:200],
            "only_pnt": comps_pnt[:200],
        },
    }
    Path(args.out_json).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"OK: wrote {args.out_png} and {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
