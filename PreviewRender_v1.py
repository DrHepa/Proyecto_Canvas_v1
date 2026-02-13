# PreviewRender_v0.py
# Proyecto Canvas — Preview Renderer (clean)
#
# Goals:
# - No legacy dead branches / no unsafe passthrough
# - Safe palette-aware dithering in ark_simulation
# - Optional legacy dithering for visual mode
# - Thread-safe small cache for overlay/mask IO (PyInstaller-friendly)

from __future__ import annotations

from pathlib import Path
from collections import OrderedDict
import threading

import numpy as np
from PIL import Image

from FrameBorder import apply_frame_border
from Dithering import floyd_steinberg_dither, ordered_dither
from ErrorDiffusion_v1 import ed_quantize_to_bytes, ordered_quantize_to_bytes, nearest_bytes_batch_from_pack


# ==========================================================
# Small thread-safe LRU cache for overlay assets
# ==========================================================

_OVERLAY_CACHE_LOCK = threading.Lock()
_OVERLAY_RGBA_CACHE: OrderedDict[tuple, np.ndarray] = OrderedDict()
_OVERLAY_RGBA_CACHE_MAX = 12


def _load_rgba_resized_cached(path: Path, *, size: tuple[int, int], nearest: bool) -> np.ndarray | None:
    """
    Loads an RGBA image from disk and resizes it to `size`.
    Cached for performance (especially under PyInstaller).
    Returns uint8 RGBA numpy array, or None if file missing/unreadable.
    """
    key = (str(path), int(size[0]), int(size[1]), "N" if nearest else "B")

    with _OVERLAY_CACHE_LOCK:
        hit = _OVERLAY_RGBA_CACHE.get(key)
        if hit is not None:
            _OVERLAY_RGBA_CACHE.move_to_end(key)
            return hit

    try:
        img = Image.open(path).convert("RGBA")
    except Exception:
        return None

    resample = Image.NEAREST if nearest else Image.BILINEAR
    if img.size != size:
        img = img.resize(size, resample)

    arr = np.array(img, dtype=np.uint8)

    with _OVERLAY_CACHE_LOCK:
        _OVERLAY_RGBA_CACHE[key] = arr
        _OVERLAY_RGBA_CACHE.move_to_end(key)
        while len(_OVERLAY_RGBA_CACHE) > _OVERLAY_RGBA_CACHE_MAX:
            _OVERLAY_RGBA_CACHE.popitem(last=False)

    return arr


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _srgb01_to_linear01(arr: np.ndarray) -> np.ndarray:
    """Convert sRGB in [0,1] to linear RGB in [0,1]."""
    return np.where(arr <= 0.04045, arr / 12.92, ((arr + 0.055) / 1.055) ** 2.4)


# ==========================================================
# Preview Renderer
# ==========================================================

def render_preview(
    image: Image.Image,
    *,
    template_id: dict | None,  # kept for compatibility (not used here)
    target_width: int,
    target_height: int,
    border: dict | None = None,
    dithering: dict | None = None,
    palette: dict | None = None,
    preview_mode: str = "visual",  # "visual" | "ark_simulation"
    game_object_type=None,         # kept for compatibility (not used here)
    show_game_object: bool = False,  # kept for compatibility (not used here)
    overlay_def: dict | None = None,
) -> Image.Image:
    """
    Renderiza una preview visual del resultado final, sin generar .pnt.

    Contratos:
    - Border: uint8 RGBA -> uint8 RGBA
    - Dithering: float32 RGB [0–1] -> float32 RGB [0–1]
    - Alpha NO se ditheriza
    """

    if target_width <= 0 or target_height <= 0:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))

    # --------------------------------------------------
    # 1) Resize base (uint8 RGBA)
    # --------------------------------------------------
    img = image.convert("RGBA")
    if img.size != (target_width, target_height):
        img = img.resize((target_width, target_height), Image.BILINEAR)

    img_np = np.array(img, dtype=np.uint8)

    # --------------------------------------------------
    # 2) Border (uint8 RGBA -> uint8 RGBA)
    # --------------------------------------------------
    if border:
        style = border.get("style", "none")
        size = int(border.get("size", 0))
        noise = float(border.get("noise", 0.0))

        if style != "none" and size > 0:
            if style == "image":
                frame_img = border.get("frame_image")
                if frame_img is None:
                    pass  # no-op: aún no hay frame cargado
                else:
                    frame_np = np.array(frame_img.convert("RGBA"), dtype=np.uint8)
                    img_np = apply_frame_border(
                        img_np,
                        border_size=size,
                        style="image",
                        frame_image=frame_np,
                        noise_strength=float(border.get("noise", 0.0)),
                    )
    # --------------------------------------------------
    # 3) Split RGBA (uint8)
    # --------------------------------------------------
    rgb_u8_in = img_np[..., :3]
    a_u8_in = img_np[..., 3]

    # --------------------------------------------------
    # 4) Dithering + ARK simulation (bytes-first)
    # --------------------------------------------------
    d_mode = "none"
    d_strength = 0.5
    if dithering:
        d_mode = dithering.get("mode", "none")
        d_strength = _clamp01(float(dithering.get("strength", 0.5)))

    if preview_mode == "ark_simulation":
        if not (palette and palette.get("mode") == "palette"):
            out = np.zeros((target_height, target_width, 4), dtype=np.uint8)
        else:
            translator = palette.get("translator")
            enabled = palette.get("enabled_dyes", None)
            # NOTE: palette fields like byte_to_rgb_u8 are numpy arrays.
            # Never use `or` with numpy arrays (truth-value is ambiguous).
            pack = palette.get("pack")
            if pack is None and translator is not None:
                pack = translator.palette_pack(enabled_dyes=enabled)

            b2rgb = palette.get("byte_to_rgb_u8")
            if b2rgb is None and translator is not None:
                b2rgb = translator.byte_to_rgb_u8(enabled_dyes=enabled)

            if pack is None or b2rgb is None:
                out = np.zeros((target_height, target_width, 4), dtype=np.uint8)
            else:
                alpha_threshold = int(palette.get("alpha_threshold", 10))
                active = a_u8_in >= alpha_threshold

                # Quantization/dithering must use the same RGB space as the dye table.
                # In this project build, TablaDyes_v1.json stores dye "linear_rgb" values
                # already in *sRGB normalized* (0..1). Historical pipeline therefore treats
                # image_rgb/255 as that same space (no sRGB->linear transform).
                rgb_lin = rgb_u8_in.astype(np.float32) / 255.0

                if d_mode in ("palette_fs", "fs"):
                    bytes_map = ed_quantize_to_bytes(
                        rgb_lin,
                        active,
                        pack["palette_w"],
                        pack["palette_w_norm2"],
                        pack["palette_bytes"],
                        pack["palette_linear"],
                        kernel="floyd_steinberg",
                        strength=d_strength,
                        serpentine=True,
                        respect_mask=True,
                        clamp01=True,
                    )

                elif d_mode in ("palette_ordered", "ordered"):
                    bytes_map = ordered_quantize_to_bytes(
                        rgb_lin,
                        active,
                        palette_w=pack["palette_w"],
                        palette_w_norm2=pack["palette_w_norm2"],
                        palette_bytes=pack["palette_bytes"],
                        sqrt_w=pack.get("sqrt_w"),
                        strength=d_strength,
                    )

                else:
                    bytes_map = np.zeros((target_height, target_width), dtype=np.uint8)
                    if np.any(active):
                        flat = rgb_lin.reshape(-1, 3)
                        act = active.reshape(-1)
                        idx = np.nonzero(act)[0]
                        bytes_sel = nearest_bytes_batch_from_pack(
                            flat[idx],
                            palette_w=pack["palette_w"],
                            palette_w_norm2=pack["palette_w_norm2"],
                            palette_bytes=pack["palette_bytes"],
                            sqrt_w=pack.get("sqrt_w"),
                        )
                        bytes_map.reshape(-1)[idx] = bytes_sel

                rgb_out = b2rgb[bytes_map]
                a_out = a_u8_in.copy()
                a_out[~active] = 0
                out = np.dstack([rgb_out, a_out])

    else:
        # visual mode: optional legacy dithering on float RGB
        rgb = rgb_u8_in.astype(np.float32) / 255.0
        alpha = a_u8_in.astype(np.float32) / 255.0

        def _identity_quantize(rgb_arr):
            return rgb_arr

        if d_mode == "fs":
            rgb = floyd_steinberg_dither(rgb, quantize_fn=_identity_quantize, strength=d_strength)

        elif d_mode == "ordered":
            rgb = ordered_dither(rgb, quantize_fn=_identity_quantize, strength=d_strength)

        rgb_u8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        a_u8 = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
        out = np.dstack([rgb_u8, a_u8])


    # --------------------------------------------------
    # 7) Overlay / Mask (optional)
    # --------------------------------------------------
    if overlay_def and not (preview_mode == "ark_simulation" and (not palette or palette.get("mode") != "palette")):
        out = out.copy()
        size = (target_width, target_height)

        # 7.1 Mask as alpha (real cut)
        mask_alpha = overlay_def.get("mask_alpha")
        if isinstance(mask_alpha, np.ndarray):
            alpha2 = mask_alpha
            if alpha2.dtype == np.bool_:
                alpha2 = alpha2.astype(np.uint8) * 255
            elif alpha2.dtype != np.uint8:
                alpha2 = alpha2.astype(np.uint8)
            # Ensure shape (H,W) and resize if needed
            if alpha2.ndim == 3:
                alpha2 = alpha2[..., 0]
            if alpha2.shape != (target_height, target_width):
                try:
                    alpha_img = Image.fromarray(alpha2, mode="L")
                    alpha_img = alpha_img.resize((target_width, target_height), Image.NEAREST)
                    alpha2 = np.array(alpha_img, dtype=np.uint8)
                except Exception:
                    alpha2 = None
            if alpha2 is not None:
                out[..., 3] = alpha2
        else:
            mask_path = overlay_def.get("mask")
            if mask_path is not None:
                mask_np = _load_rgba_resized_cached(Path(mask_path), size=size, nearest=True)
                if mask_np is not None:
                    out[..., 3] = mask_np[..., 3]
        # 7.2 Overlay RGB (decorative, respects overlay alpha>0)
        overlay_path = overlay_def.get("image")
        if overlay_path is not None:
            overlay_np = _load_rgba_resized_cached(Path(overlay_path), size=size, nearest=True)
            if overlay_np is not None:
                mask_overlay = overlay_np[..., 3] > 0
                out[..., :3][mask_overlay] = overlay_np[..., :3][mask_overlay]

    return Image.fromarray(out, mode="RGBA")
