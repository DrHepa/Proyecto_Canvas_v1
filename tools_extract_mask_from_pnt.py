from __future__ import annotations

import argparse
from pathlib import Path

from MaskExtractor import save_mask_png


def main() -> int:
    ap = argparse.ArgumentParser(description="Extrae una máscara PNG (alpha) desde un .pnt.")
    ap.add_argument("pnt", type=str, help="Ruta .pnt")
    ap.add_argument("out", type=str, help="Ruta .png")
    ap.add_argument("--crop", action="store_true", help="Recorta a bbox de la máscara")
    ap.add_argument("--mode", type=str, default="auto", choices=["auto", "nonwhite", "black"],
                    help="auto: Mask_* => black; si no => nonwhite. nonwhite: byte!=0. black: solo byte de 'Black'.")
    args = ap.parse_args()

    info = save_mask_png(Path(args.pnt), Path(args.out), crop_to_bbox=args.crop, mode=args.mode)
    print(f"OK: {args.out}  ({info.width}x{info.height}) nonzero={info.nonzero} kind={info.kind} mode={info.mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
