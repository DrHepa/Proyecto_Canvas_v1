# PntColorTranslator_v0.py

import json
import math
from typing import Tuple, Optional, List, Dict

import numpy as np


class DyeEntry:
    def __init__(self, name: str, observed_byte: int, linear_rgb: Tuple[float, float, float]):
        self.name = name
        self.observed_byte = observed_byte
        self.linear_rgb = tuple(linear_rgb)


class PntColorTranslatorV1:
    """ARK dye palette helper.

    - match_linear_rgb(): slow per-pixel matching (kept for compatibility)
    - nearest_bytes_batch(): fast batch matching for the *active* palette
    - palette_pack(): exposes a numpy-friendly palette pack (optionally filtered)
    - byte_to_rgb_u8(): map bytes -> RGB uint8 for preview
    """

    def __init__(
        self,
        tabla_path: str,
        *,
        enabled_dyes: Optional[set[int]] = None,
    ):
        self._tabla_path = str(tabla_path)

        data = json.loads(open(tabla_path, "r", encoding="utf-8").read())

        if "dyes" not in data:
            raise RuntimeError("TablaDyes_v1.json no contiene la clave 'dyes'")

        all_dyes: List[DyeEntry] = []

        for d in data["dyes"]:
            if d.get("observed_byte") is None:
                continue

            all_dyes.append(
                DyeEntry(
                    name=d["name"],
                    observed_byte=d["observed_byte"],
                    linear_rgb=d["linear_rgb"],
                )
            )

        if not all_dyes:
            raise RuntimeError("No se han cargado dyes válidos desde TablaDyes_v1.json")

        # Keep a canonical list to build filtered packs without reloading JSON.
        self._all_dyes = list(all_dyes)

        # --------------------------------------------
        # Active dyes filter (used by nearest_bytes_batch)
        # --------------------------------------------
        if enabled_dyes is None:
            self.dyes = all_dyes
        else:
            if not enabled_dyes:
                raise RuntimeError("enabled_dyes está vacío: no hay dyes activos")

            self.dyes = [d for d in all_dyes if d.observed_byte in enabled_dyes]

            if not self.dyes:
                raise RuntimeError("enabled_dyes no coincide con ningún dye válido")

        # --------------------------------------------------
        # Cache numpy arrays for fast batch matching (ACTIVE palette)
        # --------------------------------------------------
        # Shape: (K, 3) float32
        self._palette_linear = np.array([d.linear_rgb for d in self.dyes], dtype=np.float32)
        # Shape: (K,) uint8
        self._palette_bytes = np.array([d.observed_byte for d in self.dyes], dtype=np.uint8)

        # Rec.709 weights for squared distance (same as match_linear_rgb)
        w = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        self._sqrt_w = np.sqrt(w)

        # Precompute weighted palette and norms: ||pal * sqrt(w)||^2
        self._palette_w = self._palette_linear * self._sqrt_w
        self._palette_w_norm2 = np.sum(self._palette_w * self._palette_w, axis=1)

    # --------------------------------------------------
    # NEW: palette pack for bytes-first render/encode
    # --------------------------------------------------

    def palette_pack(self, *, enabled_dyes: Optional[set[int]] = None) -> Dict[str, np.ndarray]:
        """Return a palette pack usable by ErrorDiffusion_v1.

        If enabled_dyes is None, uses the currently active palette (self.dyes).
        If enabled_dyes is provided, filters from the full palette loaded from JSON.

        Returns dict with:
        - palette_linear: float32 (K,3)
        - palette_bytes: uint8 (K,)
        - palette_w: float32 (K,3)
        - palette_w_norm2: float32 (K,)
        - sqrt_w: float32 (3,)
        """

        if enabled_dyes is None:
            dyes = self.dyes
        else:
            if not enabled_dyes:
                dyes = []
            else:
                dyes = [d for d in self._all_dyes if d.observed_byte in enabled_dyes]

        if not dyes:
            return {
                "palette_linear": np.zeros((0, 3), dtype=np.float32),
                "palette_bytes": np.zeros((0,), dtype=np.uint8),
                "palette_w": np.zeros((0, 3), dtype=np.float32),
                "palette_w_norm2": np.zeros((0,), dtype=np.float32),
                "sqrt_w": self._sqrt_w.astype(np.float32, copy=False),
            }

        palette_linear = np.array([d.linear_rgb for d in dyes], dtype=np.float32)
        palette_bytes = np.array([d.observed_byte for d in dyes], dtype=np.uint8)

        palette_w = palette_linear * self._sqrt_w
        palette_w_norm2 = np.sum(palette_w * palette_w, axis=1)

        return {
            "palette_linear": palette_linear,
            "palette_bytes": palette_bytes,
            "palette_w": palette_w,
            "palette_w_norm2": palette_w_norm2,
            "sqrt_w": self._sqrt_w.astype(np.float32, copy=False),
        }

    def byte_to_rgb_u8(self, *, enabled_dyes: Optional[set[int]] = None) -> np.ndarray:
        """Return a uint8 mapping array of shape (256,3) for preview.

        - Unassigned bytes map to (0,0,0).
        - IMPORTANT: In TablaDyes_v1.json the field is named `linear_rgb` but, in this
          project build, values are already stored as *sRGB normalized* (0..1).
          Therefore we map them directly to uint8 without gamma conversion.
        """

        if enabled_dyes is None:
            dyes = self.dyes
        else:
            if not enabled_dyes:
                dyes = []
            else:
                dyes = [d for d in self._all_dyes if d.observed_byte in enabled_dyes]

        out = np.zeros((256, 3), dtype=np.uint8)
        for d in dyes:
            b = int(d.observed_byte) & 0xFF
            srgb01 = np.asarray(d.linear_rgb, dtype=np.float32)
            rgb_u8 = np.clip(srgb01 * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
            out[b, :] = rgb_u8
        return out

    # --------------------------------------------------
    # RGB -> RGB (for legacy dithering)
    # --------------------------------------------------

    def quantize_linear_rgb(self, rgb: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Devuelve el RGB lineal del dye más cercano (compat)."""
        candidates = self.match_linear_rgb(rgb, max_candidates=1)
        return candidates[0]["linear_rgb"]

    # --------------------------------------------------
    # Slow matcher (compat)
    # --------------------------------------------------

    def match_linear_rgb(
        self,
        rgb: Tuple[float, float, float],
        *,
        max_candidates: int = 3,
    ) -> list[dict]:
        """Devuelve una lista de candidatos de dye ordenados por cercanía."""

        distances = []

        r, g, b = rgb

        for d in self.dyes:
            dr = r - d.linear_rgb[0]
            dg = g - d.linear_rgb[1]
            db = b - d.linear_rgb[2]

            # Distancia ponderada por luminancia (Rec.709)
            dist = 0.2126 * dr * dr + 0.7152 * dg * dg + 0.0722 * db * db

            distances.append((dist, d))

        distances.sort(key=lambda x: x[0])
        selected = distances[:max_candidates]

        eps = 1e-6
        inv = [1.0 / (eps + d[0]) for d in selected]
        inv_sum = sum(inv)

        candidates = []
        for (dist, dye), w in zip(selected, inv):
            candidates.append(
                {
                    "byte": dye.observed_byte,
                    "linear_rgb": dye.linear_rgb,
                    "distance": dist,
                    "weight": w / inv_sum,
                }
            )

        return candidates

    # --------------------------------------------------
    # FAST: batch nearest dye byte (encoder hot-path)
    # --------------------------------------------------

    def nearest_bytes_batch(
        self,
        rgb_linear: np.ndarray,
        *,
        chunk_size: int = 65536,
    ) -> np.ndarray:
        """Return nearest dye observed_byte for each pixel.

        Uses a weighted squared-distance in linear RGB (Rec.709 weights),
        implemented as a GEMM-friendly formula:

            d^2 = ||x||^2 + ||p||^2 - 2 x·p

        where x and p are RGB scaled by sqrt(weights).

        Parameters
        ----------
        rgb_linear: (N,3) float array in [0,1]
        chunk_size: pixels per chunk (controls memory)
        """
        if rgb_linear.ndim != 2 or rgb_linear.shape[1] != 3:
            raise ValueError("rgb_linear debe ser (N,3)")

        rgb = np.asarray(rgb_linear, dtype=np.float32, order="C")
        n = int(rgb.shape[0])
        if n == 0:
            return np.empty((0,), dtype=np.uint8)

        pal_w = self._palette_w  # (K,3)
        pal_norm2 = self._palette_w_norm2  # (K,)
        pal_bytes = self._palette_bytes

        out = np.empty((n,), dtype=np.uint8)
        sw = self._sqrt_w

        cs = int(chunk_size)
        for i0 in range(0, n, cs):
            i1 = min(i0 + cs, n)
            c = rgb[i0:i1]

            cw = c * sw
            cw_norm2 = np.sum(cw * cw, axis=1)

            dist = cw_norm2[:, None] + pal_norm2[None, :] - 2.0 * (cw @ pal_w.T)
            idx = np.argmin(dist, axis=1)
            out[i0:i1] = pal_bytes[idx]

        return out

    # --------------------------------------------------

    @staticmethod
    def _distance(a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)
