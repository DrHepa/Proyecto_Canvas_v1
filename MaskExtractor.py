from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import json
import math
import hashlib
import numpy as np
from PIL import Image

from PntIO import looks_like_header20, parse_header20
from ExternalPntLibrary_v1 import try_parse_asa_guid_header_pnt


@dataclass(frozen=True)
class MaskResult:
    width: int
    height: int
    nonzero: int
    kind: str
    mode: str
    painted_byte: int


def _load_black_byte(project_root: Path) -> int:
    """Resolve 'Black' dye byte from TablaDyes_v1.json (canonical in this repo)."""
    tabla = project_root / "TablaDyes_v1.json"
    try:
        data = json.loads(tabla.read_text(encoding="utf-8"))
        for d in data.get("dyes", []):
            name = str(d.get("name", "")).lower()
            aliases = [str(a).lower() for a in d.get("aliases", [])]
            if "black" in name or "black" in aliases:
                b = d.get("observed_byte", None)
                if isinstance(b, int):
                    return int(b)
    except Exception:
        pass
    # fallback to observed value in current repo history
    return 100


def read_pnt_raster_any(
    pnt_path: Path,
    *,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Reads raster from either:
      - header20 ('raster20')
      - ASA GUID-header (MyPaintings/cache)
    Returns (raster2d uint8, meta dict with width/height/kind).
    """
    data = pnt_path.read_bytes()

    # header20
    if looks_like_header20(data):
        h = parse_header20(data[:20])
        off = 20
        raster = np.frombuffer(data[off:off + h.paint_data_size], dtype=np.uint8).reshape((h.height, h.width))
        return raster, {"kind": "raster20", "width": h.width, "height": h.height}

    # ASA GUID
    meta = try_parse_asa_guid_header_pnt(pnt_path)
    if meta:
        off = int(meta["raster_off"])
        raster_len = int(meta["raster_len"])
        w = int(meta.get("a1", 0) or 0)
        h = int(meta.get("a2", 0) or 0)

        # Explicit override (from scan hint / UI) wins if consistent.
        if isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0 and width * height == raster_len:
            raster = np.frombuffer(data[off:off + raster_len], dtype=np.uint8).reshape((height, width))
            return raster, {"kind": "asa_guid_override", "width": width, "height": height, "blueprint": meta.get("blueprint", ""), "guid": meta.get("guid", "")}
        if w > 0 and h > 0 and w * h == raster_len:
            raster = np.frombuffer(data[off:off + raster_len], dtype=np.uint8).reshape((h, w))
            return raster, {"kind": "asa_guid", "width": w, "height": h, "blueprint": meta.get("blueprint", ""), "guid": meta.get("guid", "")}

        # fallback: attempt square, then factorize (deterministic, bounded)
        sq = int(round(raster_len ** 0.5))
        if sq * sq == raster_len:
            raster = np.frombuffer(data[off:off + raster_len], dtype=np.uint8).reshape((sq, sq))
            return raster, {"kind": "asa_guid_guess", "width": sq, "height": sq, "blueprint": meta.get("blueprint", ""), "guid": meta.get("guid", "")}

        # factorize bounded (prefer near-square if ambiguous). If a single dimension is provided,
        # prefer matches that satisfy it.
        pairs = []
        for a in range(1, int(math.isqrt(raster_len)) + 1):
            if raster_len % a == 0:
                b = raster_len // a
                if 16 <= a <= 4096 and 16 <= b <= 4096:
                    pairs.append((a, b))
                    if a != b:
                        pairs.append((b, a))
        if not pairs:
            raise ValueError(f"No valid WxH factorization for raster_len={raster_len}")

        def _pair_score(wh):
            pw, ph = wh
            penalty = abs(pw - ph)
            # If height is known, prefer exact height matches strongly.
            if isinstance(height, int) and height > 0:
                penalty += 0 if ph == height else 10_000
            # If width is known, prefer exact width matches strongly.
            if isinstance(width, int) and width > 0:
                penalty += 0 if pw == width else 10_000
            return (penalty, -min(pw, ph))

        pairs.sort(key=_pair_score)
        w, h = pairs[0]
        raster = np.frombuffer(data[off:off + raster_len], dtype=np.uint8).reshape((h, w))
        return raster, {"kind": "asa_guid_factor", "width": w, "height": h, "blueprint": meta.get("blueprint", ""), "guid": meta.get("guid", "")}

    raise ValueError("Unknown .pnt format (not header20, not ASA GUID).")


def extract_mask_from_pnt(
    pnt_path: Path,
    *,
    mode: str = "auto",  # auto|nonwhite|black
    crop_to_bbox: bool = False,
) -> Tuple[Image.Image, MaskResult]:
    """Extrae una máscara RGBA desde un .pnt.

    Convención final (PNG):
      - Pintable => negro opaco (alpha 255)
      - No pintable => transparente (alpha 0)

    Modos:
      - auto: si el nombre empieza por 'Mask_' => usa 'black'; si no => 'nonwhite'
      - nonwhite: cualquier byte != 0 (White) => pintable
      - black: solo byte == Black (según TablaDyes_v1.json) => pintable
    """
    raster, meta = read_pnt_raster_any(pnt_path)
    h, w = raster.shape

    project_root = Path(__file__).resolve().parent
    black_byte = _load_black_byte(project_root)

    chosen_mode = mode.lower().strip()
    if chosen_mode == "auto":
        # Accept both 'Mask_' and 'Mask' prefixes (users often name the file simply as 'Mask').
        if pnt_path.stem.lower().startswith("mask"):
            chosen_mode = "black"
        else:
            chosen_mode = "nonwhite"

    if chosen_mode == "black":
        painted = (raster == black_byte)
        painted_byte = black_byte
    elif chosen_mode == "nonwhite":
        painted = (raster != 0)
        painted_byte = -1  # "any nonwhite"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    mask = painted.astype(np.uint8) * 255
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 3] = mask  # alpha
    # RGB=0 already (black)

    img = Image.fromarray(rgba, mode="RGBA")

    if crop_to_bbox:
        ys, xs = np.where(mask != 0)
        if len(xs) and len(ys):
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            img = img.crop((x0, y0, x1 + 1, y1 + 1))

    info = MaskResult(
        width=w,
        height=h,
        nonzero=int(mask.sum() // 255),
        kind=str(meta.get("kind", "unknown")),
        mode=chosen_mode,
        painted_byte=int(painted_byte),
    )
    return img, info


def save_mask_png(
    pnt_path: Path,
    out_png_path: Path,
    *,
    mode: str = "auto",
    crop_to_bbox: bool = False,
) -> MaskResult:
    img, info = extract_mask_from_pnt(pnt_path, mode=mode, crop_to_bbox=crop_to_bbox)
    out_png_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png_path)
    return info


def _alpha_runs(mask_alpha: np.ndarray) -> list[list[int]]:
    """Encode alpha>0 pixels into scanline runs: [y, x0, x1) (x1 exclusive)."""
    h, w = mask_alpha.shape
    out: list[list[int]] = []
    for y in range(h):
        row = mask_alpha[y]
        x = 0
        while x < w:
            if row[x] != 0:
                x0 = x
                x += 1
                while x < w and row[x] != 0:
                    x += 1
                out.append([int(y), int(x0), int(x)])
            else:
                x += 1
    return out
def _fill_holes_bool(mask: np.ndarray) -> np.ndarray:
    """Fill holes in a boolean mask deterministically.

    A 'hole' is a False-region fully enclosed by True pixels.
    Implementation: flood-fill background from edges on the inverted mask.
    """
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)
    h, w = mask.shape
    inv = ~mask
    visited = np.zeros((h, w), dtype=np.bool_)

    from collections import deque
    q = deque()

    # enqueue border pixels where inv is True (background)
    for x in range(w):
        if inv[0, x] and not visited[0, x]:
            visited[0, x] = True
            q.append((0, x))
        if inv[h - 1, x] and not visited[h - 1, x]:
            visited[h - 1, x] = True
            q.append((h - 1, x))
    for y in range(h):
        if inv[y, 0] and not visited[y, 0]:
            visited[y, 0] = True
            q.append((y, 0))
        if inv[y, w - 1] and not visited[y, w - 1]:
            visited[y, w - 1] = True
            q.append((y, w - 1))

    # BFS 4-neighborhood
    while q:
        y, x = q.popleft()
        if y > 0 and inv[y - 1, x] and not visited[y - 1, x]:
            visited[y - 1, x] = True
            q.append((y - 1, x))
        if y + 1 < h and inv[y + 1, x] and not visited[y + 1, x]:
            visited[y + 1, x] = True
            q.append((y + 1, x))
        if x > 0 and inv[y, x - 1] and not visited[y, x - 1]:
            visited[y, x - 1] = True
            q.append((y, x - 1))
        if x + 1 < w and inv[y, x + 1] and not visited[y, x + 1]:
            visited[y, x + 1] = True
            q.append((y, x + 1))

    # holes are inv True pixels not reachable from border
    holes = inv & (~visited)
    if not holes.any():
        return mask
    return mask | holes



def _load_pairs_alpha_bool(pairs_png_path: Path, w: int, h: int) -> tuple[np.ndarray | None, dict]:
    """Load *_mask_user_P.png as boolean mask (alpha>0), resizing nearest if needed."""
    info = {"pairs_png": str(pairs_png_path), "alpha_pixels": 0}
    try:
        img = Image.open(pairs_png_path).convert("RGBA")
        if img.size != (w, h):
            img = img.resize((w, h), Image.NEAREST)
        arr = np.array(img, dtype=np.uint8)
        a = arr[..., 3] > 0
        info["alpha_pixels"] = int(a.sum())
        return a, info
    except Exception:
        return None, info

def _refine_mask_with_pairs(
    base_mask: np.ndarray,
    *,
    pairs_png_path: Path,
) -> tuple[np.ndarray, dict]:
    """Refine base_mask (bool HxW) by snapping to colored regions in pairs_png_path.

    Contract:
    - pairs png may contain transparent pixels; only alpha>0 are considered part of a region.
    - Any RGB color touched by base_mask expands to include all pixels of that color.
    Returns (refined_mask_bool, debug_info).
    """
    info = {"pairs_png": str(pairs_png_path), "touched_colors": 0, "added_pixels": 0}
    try:
        img = Image.open(pairs_png_path).convert("RGBA")
    except Exception:
        return base_mask, info

    arr = np.array(img, dtype=np.uint8)
    h, w = base_mask.shape
    if arr.shape[0] != h or arr.shape[1] != w:
        # resize pairs map nearest, preserving regions
        img = img.resize((w, h), Image.NEAREST)
        arr = np.array(img, dtype=np.uint8)

    a = arr[..., 3] > 0
    rgb = arr[..., :3]
    # colors touched by base mask (where pairs alpha>0)
    touched = rgb[base_mask & a]
    if touched.size == 0:
        return base_mask, info

    # unique colors (as tuples)
    # Convert to bytes key for set
    touched_colors = set(map(bytes, touched.reshape(-1, 3).tolist()))
    info["touched_colors"] = int(len(touched_colors))

    # build mask: pixels with alpha>0 and color in touched set
    # Vectorized approach: create key per pixel
    flat_rgb = rgb.reshape(-1, 3)
    flat_a = a.reshape(-1)
    keys = [bytes(v.tolist()) for v in flat_rgb]  # small (<=256*256)
    sel = np.zeros((h * w,), dtype=np.bool_)
    for i, k in enumerate(keys):
        if flat_a[i] and k in touched_colors:
            sel[i] = True
    snapped = sel.reshape((h, w))

    refined = base_mask | snapped
    info["added_pixels"] = int((refined & (~base_mask)).sum())
    return refined, info





def save_user_mask_pack(
    pnt_path: Path,
    *,
    blueprint: str,
    out_dir: Path,
    width: Optional[int] = None,
    height: Optional[int] = None,
    mode: str = "auto",
    crop_to_bbox: bool = False,
) -> Dict[str, Any]:
    """Exporta un pack portable para máscaras de usuario.

    NO se integra en el pipeline principal: solo prepara artefactos reproducibles.
    Outputs en out_dir:
      - <Blueprint>_mask_user.png
      - <Blueprint>_mask_user.json
      - <Blueprint>_mask_user.runs.json
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read raster (supports header20 and ASA GUID-header). Allow explicit WxH override.
    raster, meta = read_pnt_raster_any(Path(pnt_path), width=width, height=height)
    h, w = raster.shape

    project_root = Path(__file__).resolve().parent
    black_byte = _load_black_byte(project_root)

    chosen_mode = (mode or "auto").lower().strip()
    if chosen_mode == "auto":
        if Path(pnt_path).stem.lower().startswith("mask"):
            chosen_mode = "black"
        else:
            chosen_mode = "nonwhite"

    if chosen_mode == "black":
        painted = (raster == black_byte)
        painted_byte = int(black_byte)
    elif chosen_mode == "nonwhite":
        painted = (raster != 0)
        painted_byte = -1
    else:
        raise ValueError(f"Unknown mode: {mode}")

    mask_alpha = (painted.astype(np.uint8) * 255)

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 3] = mask_alpha
    img = Image.fromarray(rgba, mode="RGBA")

    if crop_to_bbox:
        ys, xs = np.where(mask_alpha != 0)
        if len(xs) and len(ys):
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            img = img.crop((x0, y0, x1 + 1, y1 + 1))

    png_name = f"{blueprint}_mask_user.png"
    json_name = f"{blueprint}_mask_user.json"
    runs_name = f"{blueprint}_mask_user.runs.json"

    png_path = out_dir / png_name
    json_path = out_dir / json_name
    runs_path = out_dir / runs_name

    img.save(png_path)

    # --------------------------------------------------
    # Refined mask (deterministic): hole-fill + optional snap to *_mask_user_P.png
    # --------------------------------------------------
    base_bool = (mask_alpha != 0)
    refined_bool = _fill_holes_bool(base_bool)

    pair_info = None
    pairs_png = out_dir / f"{blueprint}_mask_user_P.png"
    if pairs_png.exists():
        pairs_bool, pair_info = _load_pairs_alpha_bool(pairs_png, w, h)
        if pairs_bool is not None and pairs_bool.any():
            refined_bool = refined_bool | pairs_bool
        refined_bool = _fill_holes_bool(refined_bool)

    refined_alpha = refined_bool.astype(np.uint8) * 255
    refined_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    refined_rgba[..., 3] = refined_alpha
    refined_img = Image.fromarray(refined_rgba, mode="RGBA")

    refined_png_name = f"{blueprint}_mask_user_refined.png"
    refined_runs_name = f"{blueprint}_mask_user_refined.runs.json"
    refined_png_path = out_dir / refined_png_name
    refined_runs_path = out_dir / refined_runs_name
    refined_img.save(refined_png_path)

    refined_runs_payload = {
        "blueprint": blueprint,
        "version": 1,
        "resolution": [int(w), int(h)],
        "runs": _alpha_runs(refined_alpha),
        "format": "runs_y_x0_x1_exclusive",
        "refined": True,
    }
    refined_runs_path.write_text(json.dumps(refined_runs_payload, indent=2), encoding="utf-8")

    runs_payload = {
        "blueprint": blueprint,
        "version": 1,
        "resolution": [int(w), int(h)],
        "runs": _alpha_runs(mask_alpha),
        "format": "runs_y_x0_x1_exclusive",
    }
    runs_path.write_text(json.dumps(runs_payload, indent=2), encoding="utf-8")

    sha = hashlib.sha256(raster.tobytes()).hexdigest()
    nonzero = int(mask_alpha.sum() // 255)
    total = int(w * h)
    coverage = float(nonzero) / float(total) if total else 0.0

    meta_payload: Dict[str, Any] = {
        "blueprint": blueprint,
        "version": 1,
        "source": {
            "pnt": str(Path(pnt_path)),
            "kind": str(meta.get("kind", "unknown")),
            "guid": str(meta.get("guid", "") or ""),
            "blueprint_in_file": str(meta.get("blueprint", "") or ""),
        },
        "resolution": [int(w), int(h)],
        "mode": chosen_mode,
        "painted_byte": int(painted_byte),
        "mask": {
            "nonzero": nonzero,
            "coverage": coverage,
        },

        "refined": {
            "png": refined_png_name,
            "runs": refined_runs_name,
            "nonzero": int(refined_alpha.sum() // 255),
            "coverage": float(int(refined_alpha.sum() // 255)) / float(total) if total else 0.0,
            "pair_refine": (pair_info or None),
            "hole_fill": True,
        },
        "sha256_raster": sha,
        "png": png_name,
        "runs": runs_name,
    }
    json_path.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")

    return meta_payload

def refine_user_mask_pack_existing(
    *,
    blueprint: str,
    out_dir: Path,
) -> Dict[str, Any]:
    """Recompute refined artifacts for an existing UserMask pack.

    Requires:
      - <BP>_mask_user.png
    Optional:
      - <BP>_mask_user_P.png
    Produces/updates:
      - <BP>_mask_user_refined.png
      - <BP>_mask_user_refined.runs.json
      - Updates <BP>_mask_user.json with 'refined' block.
    """
    out_dir = Path(out_dir)
    png_path = out_dir / f"{blueprint}_mask_user.png"
    meta_path = out_dir / f"{blueprint}_mask_user.json"
    if not png_path.exists():
        raise FileNotFoundError(f"Missing base user mask: {png_path}")

    img = Image.open(png_path).convert("RGBA")
    arr = np.array(img, dtype=np.uint8)
    h, w = arr.shape[0], arr.shape[1]
    base_alpha = arr[..., 3]
    base_bool = base_alpha > 0

    refined_bool = _fill_holes_bool(base_bool)
    pair_info = None
    pairs_png = out_dir / f"{blueprint}_mask_user_P.png"
    if pairs_png.exists():
        pairs_bool, pair_info = _load_pairs_alpha_bool(pairs_png, w, h)
        if pairs_bool is not None and pairs_bool.any():
            refined_bool = refined_bool | pairs_bool
        refined_bool = _fill_holes_bool(refined_bool)

    refined_alpha = refined_bool.astype(np.uint8) * 255
    refined_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    refined_rgba[..., 3] = refined_alpha
    refined_img = Image.fromarray(refined_rgba, mode="RGBA")

    refined_png_name = f"{blueprint}_mask_user_refined.png"
    refined_runs_name = f"{blueprint}_mask_user_refined.runs.json"
    refined_png_path = out_dir / refined_png_name
    refined_runs_path = out_dir / refined_runs_name

    refined_img.save(refined_png_path)

    refined_runs_payload = {
        "blueprint": blueprint,
        "version": 1,
        "resolution": [int(w), int(h)],
        "runs": _alpha_runs(refined_alpha),
        "format": "runs_y_x0_x1_exclusive",
        "refined": True,
    }
    refined_runs_path.write_text(json.dumps(refined_runs_payload, indent=2), encoding="utf-8")

    # update meta json if present
    meta = {}
    try:
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        meta = {}

    total = int(w * h)
    meta["refined"] = {
        "png": refined_png_name,
        "runs": refined_runs_name,
        "nonzero": int(refined_alpha.sum() // 255),
        "coverage": float(int(refined_alpha.sum() // 255)) / float(total) if total else 0.0,
        "pair_refine": (pair_info or None),
        "hole_fill": True,
    }
    meta.setdefault("blueprint", blueprint)
    meta.setdefault("version", 1)
    meta.setdefault("resolution", [int(w), int(h)])
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return meta
