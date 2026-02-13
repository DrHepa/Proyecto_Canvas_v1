"""ErrorDiffusion_v1

Proyecto Canvas — bytes-first error diffusion and palette helpers.

Main entrypoints
----------------
- ed_quantize_to_bytes(): error diffusion (Floyd–Steinberg + extra kernels) -> uint8 bytes map
- ordered_quantize_to_bytes(): ordered dithering (Bayer 4×4) -> uint8 bytes map
- nearest_bytes_batch_from_pack(): nearest (no dither) -> uint8 bytes

Rationale
---------
The encoder and the preview should not call a per-pixel "match" that loops dyes in
Python. Instead, we quantize directly to bytes and reuse that result.

Notes
-----
- The project currently uses values called "linear_rgb" in TablaDyes_v1.json but the
  pipeline historically treats image_rgb/255 as "linear". This module follows that
  convention to stay coherent with the existing encoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np


# ==========================================================
# Kernels registry
# ==========================================================

@dataclass(frozen=True)
class EDKernel:
    name: str
    taps: Tuple[Tuple[int, int, float], ...]  # (dx, dy, weight) for L->R scan


KERNELS: Dict[str, EDKernel] = {
    "floyd_steinberg": EDKernel(
        "floyd_steinberg",
        (
            (1, 0, 7.0 / 16.0),
            (-1, 1, 3.0 / 16.0),
            (0, 1, 5.0 / 16.0),
            (1, 1, 1.0 / 16.0),
        ),
    ),
    # Extra kernel (good quality, low grain)
    "atkinson": EDKernel(
        "atkinson",
        (
            (1, 0, 1.0 / 8.0),
            (2, 0, 1.0 / 8.0),
            (-1, 1, 1.0 / 8.0),
            (0, 1, 1.0 / 8.0),
            (1, 1, 1.0 / 8.0),
            (0, 2, 1.0 / 8.0),
        ),
    ),
}


# ==========================================================
# Bayer 4×4 (canonical 0..15)
# ==========================================================

_BAYER_4x4 = np.array(
    [
        [0, 8, 2, 10],
        [12, 4, 14, 6],
        [3, 11, 1, 9],
        [15, 7, 13, 5],
    ],
    dtype=np.int32,
)


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


# ==========================================================
# Palette helpers (NumPy, chunked)
# ==========================================================

def nearest_bytes_batch_from_pack(
    rgb_linear: np.ndarray,
    *,
    palette_w: np.ndarray,
    palette_w_norm2: np.ndarray,
    palette_bytes: np.ndarray,
    sqrt_w: Optional[np.ndarray] = None,
    chunk_size: int = 65536,
) -> np.ndarray:
    """Nearest dye byte for each pixel in rgb_linear (N,3).

    This mirrors PntColorTranslatorV1.nearest_bytes_batch but receives a palette pack.
    """
    if rgb_linear.ndim != 2 or rgb_linear.shape[1] != 3:
        raise ValueError("rgb_linear debe ser (N,3)")

    rgb = np.asarray(rgb_linear, dtype=np.float32, order="C")
    n = int(rgb.shape[0])
    if n == 0:
        return np.empty((0,), dtype=np.uint8)

    pal_w = np.asarray(palette_w, dtype=np.float32, order="C")
    pal_norm2 = np.asarray(palette_w_norm2, dtype=np.float32, order="C")
    pal_bytes = np.asarray(palette_bytes, dtype=np.uint8, order="C")

    if sqrt_w is None:
        sqrt_w = np.array([np.sqrt(0.2126), np.sqrt(0.7152), np.sqrt(0.0722)], dtype=np.float32)
    else:
        sqrt_w = np.asarray(sqrt_w, dtype=np.float32)

    out = np.empty((n,), dtype=np.uint8)
    cs = int(chunk_size)

    for i0 in range(0, n, cs):
        i1 = min(i0 + cs, n)
        c = rgb[i0:i1]

        cw = c * sqrt_w
        cw_norm2 = np.sum(cw * cw, axis=1)

        dist = cw_norm2[:, None] + pal_norm2[None, :] - 2.0 * (cw @ pal_w.T)
        idx = np.argmin(dist, axis=1)
        out[i0:i1] = pal_bytes[idx]

    return out


def nearest2_bytes_dists_batch_from_pack(
    rgb_linear: np.ndarray,
    *,
    palette_w: np.ndarray,
    palette_w_norm2: np.ndarray,
    palette_bytes: np.ndarray,
    sqrt_w: Optional[np.ndarray] = None,
    chunk_size: int = 65536,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return nearest and second nearest (byte, dist) for each pixel.

    Returns:
        b1: (N,) uint8
        d1: (N,) float32
        b2: (N,) uint8
        d2: (N,) float32
    """
    if rgb_linear.ndim != 2 or rgb_linear.shape[1] != 3:
        raise ValueError("rgb_linear debe ser (N,3)")

    rgb = np.asarray(rgb_linear, dtype=np.float32, order="C")
    n = int(rgb.shape[0])
    if n == 0:
        z0 = np.empty((0,), dtype=np.uint8)
        zf = np.empty((0,), dtype=np.float32)
        return z0, zf, z0.copy(), zf.copy()

    pal_w = np.asarray(palette_w, dtype=np.float32, order="C")
    pal_norm2 = np.asarray(palette_w_norm2, dtype=np.float32, order="C")
    pal_bytes = np.asarray(palette_bytes, dtype=np.uint8, order="C")

    if sqrt_w is None:
        sqrt_w = np.array([np.sqrt(0.2126), np.sqrt(0.7152), np.sqrt(0.0722)], dtype=np.float32)
    else:
        sqrt_w = np.asarray(sqrt_w, dtype=np.float32)

    b1 = np.empty((n,), dtype=np.uint8)
    b2 = np.empty((n,), dtype=np.uint8)
    d1 = np.empty((n,), dtype=np.float32)
    d2 = np.empty((n,), dtype=np.float32)

    cs = int(chunk_size)
    for i0 in range(0, n, cs):
        i1 = min(i0 + cs, n)
        c = rgb[i0:i1]

        cw = c * sqrt_w
        cw_norm2 = np.sum(cw * cw, axis=1)

        dist = cw_norm2[:, None] + pal_norm2[None, :] - 2.0 * (cw @ pal_w.T)  # (C,K)

        # top-2 indices (unordered)
        idx2 = np.argpartition(dist, kth=1, axis=1)[:, :2]  # (C,2)
        d_a = dist[np.arange(dist.shape[0]), idx2[:, 0]]
        d_b = dist[np.arange(dist.shape[0]), idx2[:, 1]]

        # order by distance
        swap = d_b < d_a
        i_first = np.where(swap, idx2[:, 1], idx2[:, 0])
        i_second = np.where(swap, idx2[:, 0], idx2[:, 1])

        d_first = np.where(swap, d_b, d_a).astype(np.float32)
        d_second = np.where(swap, d_a, d_b).astype(np.float32)

        b1[i0:i1] = pal_bytes[i_first]
        b2[i0:i1] = pal_bytes[i_second]
        d1[i0:i1] = d_first
        d2[i0:i1] = d_second

    return b1, d1, b2, d2


def ordered_quantize_to_bytes(
    rgb_linear: np.ndarray,
    active_mask: np.ndarray,
    *,
    palette_w: np.ndarray,
    palette_w_norm2: np.ndarray,
    palette_bytes: np.ndarray,
    sqrt_w: Optional[np.ndarray] = None,
    strength: float = 1.0,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Ordered dithering (Bayer 4×4) choosing between nearest and 2nd nearest.

    strength:
        0 -> always choose nearest
        1 -> choose based on inverse-distance weights
    """
    strength = _clamp01(float(strength))

    if rgb_linear.ndim != 3 or rgb_linear.shape[2] != 3:
        raise ValueError("rgb_linear debe ser (H,W,3)")

    h, w, _ = rgb_linear.shape
    if active_mask.shape != (h, w):
        raise ValueError("active_mask debe ser (H,W)")

    if out is None:
        out = np.zeros((h, w), dtype=np.uint8)
    else:
        if out.shape != (h, w) or out.dtype != np.uint8:
            raise ValueError("out debe ser uint8 (H,W)")
        out.fill(0)

    # Flatten only active pixels for top2 search
    flat = rgb_linear.reshape(-1, 3)
    act = active_mask.reshape(-1)
    if not np.any(act):
        return out

    idx_act = np.nonzero(act)[0]
    pix = flat[idx_act]

    b1, d1, b2, d2 = nearest2_bytes_dists_batch_from_pack(
        pix,
        palette_w=palette_w,
        palette_w_norm2=palette_w_norm2,
        palette_bytes=palette_bytes,
        sqrt_w=sqrt_w,
    )

    eps = np.float32(1e-6)
    w1 = 1.0 / (eps + d1)
    w2 = 1.0 / (eps + d2)
    s = w1 + w2
    w1n = (w1 / s).astype(np.float32)

    # strength modulation: blend towards 1.0 (always choose first)
    if strength < 1.0:
        w1n = (1.0 - strength) * 1.0 + strength * w1n

    # Bayer threshold t in (0,1)
    ys = (idx_act // w).astype(np.int32)
    xs = (idx_act - ys * w).astype(np.int32)
    b = _BAYER_4x4[ys & 3, xs & 3]
    t = ((b.astype(np.float32) + 0.5) / 16.0)

    choose_second = t >= w1n
    chosen = np.where(choose_second, b2, b1).astype(np.uint8)

    out_flat = out.reshape(-1)
    out_flat[idx_act] = chosen
    return out


# ==========================================================
# Error diffusion core (Numba accelerated if available)
# ==========================================================

try:
    from numba import njit
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False
    njit = None


if _HAVE_NUMBA:
    @njit(cache=True, fastmath=True)
    def _nearest_idx_weighted(r: float, g: float, b: float, pal_lin: np.ndarray) -> int:
        # IMPORTANT (Proyecto Canvas compatibility)
        # ---------------------------------------
        # Historically, the Preview dithering path used an *unweighted* Euclidean
        # distance over the palette triples (via PreviewController._nearest_color).
        # The first bytes-first implementation switched to Rec.709 weighted distance,
        # which noticeably changes palette decisions (users reported "wrong" ARK
        # simulation colors compared to the legacy preview and in-game look).
        #
        # To restore visual compatibility, we keep the function name (to avoid
        # refactors) but compute the classic Euclidean squared distance.
        best_i = 0
        best_d = 1e30
        k = pal_lin.shape[0]
        for i in range(k):
            dr = r - pal_lin[i, 0]
            dg = g - pal_lin[i, 1]
            db = b - pal_lin[i, 2]
            d = dr * dr + dg * dg + db * db
            if d < best_d:
                best_d = d
                best_i = i
        return best_i


    @njit(cache=True, fastmath=True)
    def _ed_core_numba(
        work: np.ndarray,           # (H,W,3) float32
        active: np.ndarray,         # (H,W) uint8 0/1
        pal_lin: np.ndarray,        # (K,3) float32
        pal_bytes: np.ndarray,      # (K,) uint8
        dx: np.ndarray,             # (T,) int32
        dy: np.ndarray,             # (T,) int32
        wt: np.ndarray,             # (T,) float32
        strength: float,
        serpentine: int,
        respect_mask: int,
        clamp01: int,
        out: np.ndarray,            # (H,W) uint8
    ) -> None:
        h = work.shape[0]
        w = work.shape[1]
        tcount = dx.shape[0]

        for y in range(h):
            rev = 0
            if serpentine != 0 and (y & 1) == 1:
                rev = 1

            if rev == 0:
                x0 = 0
                x1 = w
                step = 1
            else:
                x0 = w - 1
                x1 = -1
                step = -1

            x = x0
            while x != x1:
                if active[y, x] == 0:
                    x += step
                    continue

                r = work[y, x, 0]
                g = work[y, x, 1]
                b = work[y, x, 2]

                if clamp01 != 0:
                    if r < 0.0:
                        r = 0.0
                    elif r > 1.0:
                        r = 1.0
                    if g < 0.0:
                        g = 0.0
                    elif g > 1.0:
                        g = 1.0
                    if b < 0.0:
                        b = 0.0
                    elif b > 1.0:
                        b = 1.0

                idx = _nearest_idx_weighted(r, g, b, pal_lin)
                out[y, x] = pal_bytes[idx]

                pr = pal_lin[idx, 0]
                pg = pal_lin[idx, 1]
                pb = pal_lin[idx, 2]

                work[y, x, 0] = pr
                work[y, x, 1] = pg
                work[y, x, 2] = pb

                er = (r - pr) * strength
                eg = (g - pg) * strength
                eb = (b - pb) * strength

                for ti in range(tcount):
                    ddx = dx[ti]
                    if rev == 1:
                        ddx = -ddx
                    nx = x + ddx
                    ny = y + dy[ti]

                    if ny < 0 or ny >= h or nx < 0 or nx >= w:
                        continue

                    if respect_mask != 0 and active[ny, nx] == 0:
                        continue

                    wgt = wt[ti]
                    work[ny, nx, 0] += er * wgt
                    work[ny, nx, 1] += eg * wgt
                    work[ny, nx, 2] += eb * wgt

                x += step


def ed_quantize_to_bytes(
    rgb_linear: np.ndarray,
    active_mask: np.ndarray,
    palette_w: np.ndarray,
    palette_w_norm2: np.ndarray,
    palette_bytes: np.ndarray,
    palette_linear: Optional[np.ndarray] = None,
    *,
    kernel: str = "floyd_steinberg",
    strength: float = 1.0,
    serpentine: bool = True,
    respect_mask: bool = True,
    clamp01: bool = True,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Error diffusion quantization to bytes.

    Parameters
    ----------
    rgb_linear:
        float32 (H,W,3) in [0,1]
    active_mask:
        bool (H,W) – only pixels where active_mask=True are quantized and receive error.
    palette_*:
        palette pack from PntColorTranslatorV1.palette_pack()
    palette_linear:
        optional float32 (K,3). If omitted, uses palette_w/sqrt_w is not enough for
        diffusion; so this should be provided.
    """
    strength = _clamp01(float(strength))

    if rgb_linear.ndim != 3 or rgb_linear.shape[2] != 3:
        raise ValueError("rgb_linear debe ser (H,W,3)")

    h, w, _ = rgb_linear.shape
    if active_mask.shape != (h, w):
        raise ValueError("active_mask debe ser (H,W)")

    if palette_linear is None:
        # Fallback: try to reconstruct from palette_w if possible (requires sqrt_w)
        raise ValueError("palette_linear es obligatorio para error diffusion")

    pal_lin = np.asarray(palette_linear, dtype=np.float32, order="C")
    pal_bytes = np.asarray(palette_bytes, dtype=np.uint8, order="C")

    if pal_lin.ndim != 2 or pal_lin.shape[1] != 3:
        raise ValueError("palette_linear debe ser (K,3)")

    if pal_lin.shape[0] == 0:
        if out is None:
            return np.zeros((h, w), dtype=np.uint8)
        out.fill(0)
        return out

    k = KERNELS.get(kernel)
    if k is None:
        raise ValueError(f"kernel no soportado: {kernel}")

    dx = np.array([t[0] for t in k.taps], dtype=np.int32)
    dy = np.array([t[1] for t in k.taps], dtype=np.int32)
    wt = np.array([t[2] for t in k.taps], dtype=np.float32)

    if out is None:
        out = np.zeros((h, w), dtype=np.uint8)
    else:
        if out.shape != (h, w) or out.dtype != np.uint8:
            raise ValueError("out debe ser uint8 (H,W)")
        out.fill(0)

    if not np.any(active_mask):
        return out

    # Work buffer (in-place diffusion)
    work = np.asarray(rgb_linear, dtype=np.float32, order="C").copy()

    if _HAVE_NUMBA:
        active_u8 = np.ascontiguousarray(active_mask.astype(np.uint8))
        _ed_core_numba(
            work,
            active_u8,
            pal_lin,
            pal_bytes,
            dx,
            dy,
            wt,
            float(strength),
            1 if serpentine else 0,
            1 if respect_mask else 0,
            1 if clamp01 else 0,
            out,
        )
        return out

    # Pure-Python fallback (slower, but correct)
    active = active_mask
    for y in range(h):
        rev = serpentine and (y & 1) == 1
        xs = range(w - 1, -1, -1) if rev else range(w)
        for x in xs:
            if not active[y, x]:
                continue

            r, g, b = work[y, x]
            if clamp01:
                r = 0.0 if r < 0.0 else (1.0 if r > 1.0 else r)
                g = 0.0 if g < 0.0 else (1.0 if g > 1.0 else g)
                b = 0.0 if b < 0.0 else (1.0 if b > 1.0 else b)

            # nearest
            best_i = 0
            best_d = 1e30
            for i in range(pal_lin.shape[0]):
                dr = r - pal_lin[i, 0]
                dg = g - pal_lin[i, 1]
                db = b - pal_lin[i, 2]
                # See note in _nearest_idx_weighted(): legacy preview used
                # unweighted Euclidean distance.
                d = dr * dr + dg * dg + db * db
                if d < best_d:
                    best_d = d
                    best_i = i

            out[y, x] = int(pal_bytes[best_i])
            pr, pg, pb = pal_lin[best_i]
            work[y, x] = (pr, pg, pb)

            er = (r - pr) * strength
            eg = (g - pg) * strength
            eb = (b - pb) * strength

            for (ddx, ddy, wgt) in k.taps:
                if rev:
                    ddx = -ddx
                nx = x + ddx
                ny = y + ddy
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue
                if respect_mask and not active[ny, nx]:
                    continue
                work[ny, nx, 0] += er * wgt
                work[ny, nx, 1] += eg * wgt
                work[ny, nx, 2] += eb * wgt

    return out
