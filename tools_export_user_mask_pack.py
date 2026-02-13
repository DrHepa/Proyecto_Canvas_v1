from __future__ import annotations

import argparse
from pathlib import Path

from MaskExtractor import save_user_mask_pack


def _parse_size(s: str):
    s = (s or "").strip().lower().replace(" ", "")
    if not s:
        return None, None
    if "x" not in s:
        raise ValueError("Size debe ser WxH, por ejemplo 256x256")
    a, b = s.split("x", 1)
    return int(a), int(b)


def _infer_blueprint_from_name(p: Path) -> str:
    stem = p.stem.strip()
    if stem.lower().startswith("mask_"):
        return stem[5:]
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description="Exporta un pack de máscara de usuario (PNG+meta+runs) desde un .pnt.")
    ap.add_argument("pnt", type=str, help="Ruta .pnt (MyPaintings o header20)")
    ap.add_argument("--blueprint", type=str, default="", help="Blueprint, ej. Doggo_Character_BP_C")
    ap.add_argument("--out_dir", type=str, default="", help="Carpeta destino (default: Templates/UserMasks)")
    ap.add_argument("--size", type=str, default="", help="Para ASA GUID-header: WxH, ej. 256x256")
    ap.add_argument("--crop", action="store_true", help="Recortar al bounding box no-cero")
    args = ap.parse_args()

    pnt = Path(args.pnt)

    blueprint = (args.blueprint or "").strip() or _infer_blueprint_from_name(pnt)
    if not blueprint:
        raise SystemExit("Falta blueprint. Usa --blueprint o nombra el archivo como Mask_<Blueprint>.pnt")

    out_dir = Path(args.out_dir) if args.out_dir else (Path(__file__).resolve().parent / "Templates" / "UserMasks")
    w, h = _parse_size(args.size)

    meta = save_user_mask_pack(
        pnt,
        blueprint=blueprint,
        out_dir=out_dir,
        width=w,
        height=h,
        crop_to_bbox=bool(args.crop),
    )
    print(f"OK: {blueprint} -> {out_dir} | {meta.get('png')} + {meta.get('runs')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
