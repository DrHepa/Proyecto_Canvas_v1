from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def _merge_runs(runs: List[List[int]]) -> List[List[int]]:
    # Merge overlapping/adjacent runs per scanline, deterministic.
    by_y: Dict[int, List[Tuple[int,int]]] = {}
    for y, x0, x1 in runs:
        by_y.setdefault(int(y), []).append((int(x0), int(x1)))
    merged: List[List[int]] = []
    for y in sorted(by_y.keys()):
        segs = sorted(by_y[y])
        cur0, cur1 = segs[0]
        for x0, x1 in segs[1:]:
            if x0 <= cur1:  # overlap or touch
                cur1 = max(cur1, x1)
            else:
                merged.append([y, cur0, cur1])
                cur0, cur1 = x0, x1
        merged.append([y, cur0, cur1])
    return merged


def main() -> int:
    ap = argparse.ArgumentParser(description="Build part_masks.json from pairs.lite.json + anatomy.json")
    ap.add_argument("pairs_lite", type=str, help="Input <Blueprint>.pairs.lite.json")
    ap.add_argument("anatomy", type=str, help="Input anatomy.(auto|user).json")
    ap.add_argument("--out", type=str, default="", help="Output part_masks.json (default next to input)")
    args = ap.parse_args()

    pairs_path = Path(args.pairs_lite)
    anatomy_path = Path(args.anatomy)
    pairs = json.loads(pairs_path.read_text(encoding="utf-8"))
    anatomy = json.loads(anatomy_path.read_text(encoding="utf-8"))

    blueprint = anatomy.get("blueprint") or pairs.get("blueprint") or ""
    pairs_list = pairs.get("pairs") or []

    # Build map pair_key -> runs
    pair_runs = {}
    for p in pairs_list:
        k = p.get("pair_key")
        if not k:
            continue
        pair_runs[k] = {
            "red": p.get("runs_red") or [],
            "blue": p.get("runs_blue") or [],
        }

    parts = anatomy.get("parts") or {}
    out_parts: Dict[str, Dict[str, List[List[int]]]] = {}

    for part_name, ks in parts.items():
        red_runs: List[List[int]] = []
        blue_runs: List[List[int]] = []
        for k in ks or []:
            r = pair_runs.get(k)
            if not r:
                continue
            red_runs.extend(r["red"])
            blue_runs.extend(r["blue"])
        out_parts[str(part_name)] = {
            "red": _merge_runs(red_runs),
            "blue": _merge_runs(blue_runs),
        }

    out = {
        "blueprint": blueprint,
        "version": 1,
        "parts": out_parts,
    }

    out_path = Path(args.out) if args.out else pairs_path.with_name(f"{blueprint}.part_masks.json")
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"OK: {blueprint} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
