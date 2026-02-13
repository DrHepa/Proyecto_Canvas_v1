from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description="Autogenerate anatomy.auto.json from pairs.lite.json (bootstrap).")
    ap.add_argument("pairs_lite", type=str, help="Input <Blueprint>.pairs.lite.json")
    ap.add_argument("--out", type=str, default="", help="Output anatomy.auto.json (default next to input)")
    args = ap.parse_args()

    in_path = Path(args.pairs_lite)
    data = json.loads(in_path.read_text(encoding="utf-8"))

    blueprint = data.get("blueprint") or ""
    pairs = data.get("pairs") or []

    # Weighted PCA over centroids
    pts = []
    ws = []
    keys = []
    for p in pairs:
        c = p.get("centroid") or [0.0, 0.0]
        area = float(p.get("area") or 0)
        if area <= 0:
            continue
        pts.append([float(c[0]), float(c[1])])
        ws.append(area)
        keys.append(p.get("pair_key"))

    if not pts:
        raise SystemExit("No pairs with area > 0.")

    X = np.asarray(pts, dtype=np.float64)
    w = np.asarray(ws, dtype=np.float64)
    w = w / max(1e-12, w.sum())

    mu = (X * w[:, None]).sum(axis=0)
    Xc = X - mu[None, :]
    C = (Xc * w[:, None]).T @ Xc  # 2x2 cov

    # principal axis (largest eigenvalue)
    eigvals, eigvecs = np.linalg.eigh(C)
    axis = eigvecs[:, int(np.argmax(eigvals))]
    axis = axis / max(1e-12, np.linalg.norm(axis))

    proj = Xc @ axis  # 1D

    # Area-weighted terciles along projection
    order = np.argsort(proj)
    keys_sorted = [keys[i] for i in order]
    proj_sorted = proj[order]
    w_sorted = w[order]

    cum = np.cumsum(w_sorted)
    # cut positions at 1/3 and 2/3
    cut1 = int(np.searchsorted(cum, 1/3))
    cut2 = int(np.searchsorted(cum, 2/3))

    head = keys_sorted[:cut1]
    torso = keys_sorted[cut1:cut2]
    tail = keys_sorted[cut2:]

    out = {
        "blueprint": blueprint,
        "version": 1,
        "swap_head_tail": False,
        "parts": {
            "Head": head,
            "Torso": torso,
            "Tail": tail,
        },
        "order": {
            "Head": head,
            "Torso": torso,
            "Tail": tail,
        }
    }

    out_path = Path(args.out) if args.out else in_path.with_name(f"{blueprint}.anatomy.auto.json")
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"OK: {blueprint} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
