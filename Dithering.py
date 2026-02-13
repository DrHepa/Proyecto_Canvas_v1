import numpy as np

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


def floyd_steinberg_dither(
    img_rgb_linear: np.ndarray,
    quantize_fn,
    strength: float = 1.0,
):
    """
    Aplica dithering Floyd–Steinberg en RGB lineal.

    img_rgb_linear:
        np.ndarray (H, W, 3), float [0, 1]

    quantize_fn:
        función que recibe (r, g, b) y devuelve (r_q, g_q, b_q)

    strength:
        0.0–1.0, controla cuánto error se difunde
    """
    strength = _clamp01(float(strength))

    h, w, _ = img_rgb_linear.shape
    img = img_rgb_linear.copy()

    for y in range(h):
        for x in range(w):
            old = img[y, x]
            new = np.array(
                quantize_fn((float(old[0]), float(old[1]), float(old[2]))),
                dtype=np.float32,
            )
            img[y, x] = new

            error = (old - new) * strength

            if x + 1 < w:
                img[y, x + 1] += error * (7.0 / 16.0)
            if y + 1 < h:
                if x > 0:
                    img[y + 1, x - 1] += error * (3.0 / 16.0)
                img[y + 1, x] += error * (5.0 / 16.0)
                if x + 1 < w:
                    img[y + 1, x + 1] += error * (1.0 / 16.0)

    return np.clip(img, 0.0, 1.0)


def ordered_dither(
    img_rgb_linear: np.ndarray,
    quantize_fn,
    strength: float = 1.0,
):
    """
    Ordered dithering usando matriz Bayer 4×4.

    img_rgb_linear:
        np.ndarray (H, W, 3), float [0, 1]

    quantize_fn:
        función (r,g,b) -> rgb cuantizado (lineal)

    strength:
        controla cuánto se perturba el valor antes de cuantizar.
    """
    strength = _clamp01(float(strength))

    h, w, _ = img_rgb_linear.shape
    out = np.empty_like(img_rgb_linear)

    # Umbral canónico: ((b + 0.5)/16 - 0.5) en [-0.46875, +0.46875]
    # Escalado a una perturbación muy pequeña (1/255) para no “romper” colorimetría.
    for y in range(h):
        for x in range(w):
            b = _BAYER_4x4[y & 3, x & 3]
            threshold = ((b + 0.5) / 16.0) - 0.5
            perturb = threshold * strength * (1.0 / 255.0)

            rgb = img_rgb_linear[y, x] + perturb
            rgb = np.clip(rgb, 0.0, 1.0)

            out[y, x] = quantize_fn((float(rgb[0]), float(rgb[1]), float(rgb[2])))

    return out


def select_dye_ordered_dither(
    candidates: list[dict],
    *,
    x: int,
    y: int,
) -> int:
    """
    Selecciona un dye usando ordered dithering con matriz Bayer 4×4.

    candidates:
        Lista de DyeCandidate ordenada por cercanía,
        con pesos normalizados (sum ~= 1.0).

    x, y:
        Coordenadas absolutas del píxel en el raster.

    Retorna:
        observed_byte (int)
    """
    if not candidates:
        raise ValueError("candidates vacío")

    if len(candidates) == 1:
        return candidates[0]["byte"]

    b = _BAYER_4x4[y & 3, x & 3]
    t = (b + 0.5) / 16.0  # (0, 1)

    acc = 0.0
    for cand in candidates:
        acc += float(cand.get("weight", 0.0))
        if t < acc:
            return cand["byte"]

    return candidates[-1]["byte"]
