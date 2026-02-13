import numpy as np


def _lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a * (1.0 - t) + b * t


def _bilinear_sample(img: np.ndarray, x: float, y: float) -> np.ndarray:
    """
    Muestreo bilineal de una imagen (RGB o RGBA) en coordenadas float.
    """
    h, w, _ = img.shape
    x = np.clip(x, 0.0, w - 1.0)
    y = np.clip(y, 0.0, h - 1.0)

    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)

    dx = x - x0
    dy = y - y0

    c00 = img[y0, x0]
    c10 = img[y0, x1]
    c01 = img[y1, x0]
    c11 = img[y1, x1]

    c0 = c00 * (1.0 - dx) + c10 * dx
    c1 = c01 * (1.0 - dx) + c11 * dx
    return c0 * (1.0 - dy) + c1 * dy


def _normalize_rgb(color) -> np.ndarray:
    """
    Acepta (r,g,b) en [0..1] o [0..255]. Devuelve float32 [0..1].
    """
    c = np.array(color, dtype=np.float32)
    if np.max(c) > 1.0:
        c = c / 255.0
    return np.clip(c, 0.0, 1.0)


def apply_frame_border(
    image: np.ndarray,
    *,
    border_size: int,
    paint_area: dict | None = None,
    style: str = "solid",
    color_outer=(0.0, 0.0, 0.0),
    color_inner=None,
    noise_strength: float = 0.3,
    frame_image: np.ndarray | None = None,
):
    """
    Aplica un borde tipo cuadro a una imagen RGB o RGBA.

    style:
        'solid' | 'gradient' | 'rough' | 'image'

    paint_area (opcional):
        dict {offset_x, offset_y, width, height}
        Si se pasa, el borde se aplica solo alrededor de esa región.
    """
    B = int(border_size)

    # ---------------------------------------------
    # Validaciones y early exits
    # ---------------------------------------------
    if B <= 0:
        return image.copy()

    if style not in ("solid", "gradient", "rough", "image"):
        raise ValueError(f"Estilo de borde no soportado: {style}")

    if style in ("gradient", "rough") and color_inner is None:
        raise ValueError("color_inner es obligatorio para gradient y rough")

    if style == "image" and frame_image is None:
        raise ValueError("frame_image es obligatorio para style='image'")

    # Border <2 no permite interpolación/tiling estable (compat con tu lógica previa)
    if B < 2:
        return image.copy()

    original_dtype = image.dtype

    if image.dtype == np.uint8:
        img_f = image.astype(np.float32) / 255.0
    else:
        img_f = image.astype(np.float32)

    h, w, c = img_f.shape
    if c not in (3, 4):
        raise ValueError("image debe ser RGB o RGBA")

    # ---------------------------------------------
    # Resolve paint area (clamp seguro)
    # ---------------------------------------------
    if paint_area is None:
        x0, y0, x1, y1 = 0, 0, w, h
    else:
        x0 = int(paint_area["offset_x"])
        y0 = int(paint_area["offset_y"])
        x1 = x0 + int(paint_area["width"])
        y1 = y0 + int(paint_area["height"])

        x0 = max(0, min(x0, w))
        y0 = max(0, min(y0, h))
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))

    pa_w = x1 - x0
    pa_h = y1 - y0
    if pa_w <= 1 or pa_h <= 1:
        return image.copy()

    # Clamp del borde a la paint_area
    B = min(B, pa_w // 2, pa_h // 2)
    if B < 2:
        return image.copy()

    # ---------------------------------------------
    # Preparar frame image (si aplica)
    # ---------------------------------------------
    if style == "image":
        fi = frame_image
        if fi.dtype == np.uint8:
            fi = fi.astype(np.float32) / 255.0
        else:
            fi = fi.astype(np.float32)

        fh, fw, fc = fi.shape
        if fc not in (3, 4):
            raise ValueError("frame_image debe ser RGB o RGBA")
    else:
        fi = None
        fh = fw = fc = 0

    # ---------------------------------------------
    # Preparar colores base (float32 [0..1])
    # ---------------------------------------------
    outer_rgb = _normalize_rgb(color_outer)
    inner_rgb = _normalize_rgb(color_inner) if style in ("gradient", "rough") else None

    if c == 4:
        outer_px = np.array([outer_rgb[0], outer_rgb[1], outer_rgb[2], 1.0], dtype=np.float32)
        inner_px = (
            np.array([inner_rgb[0], inner_rgb[1], inner_rgb[2], 1.0], dtype=np.float32)
            if inner_rgb is not None
            else None
        )
    else:
        outer_px = outer_rgb.astype(np.float32)
        inner_px = inner_rgb.astype(np.float32) if inner_rgb is not None else None

    out = img_f.copy()

    # ---------------------------------------------
    # Fast path: SOLID (sin loops)
    # ---------------------------------------------
    if style == "solid":
        out[y0 : y0 + B, x0:x1] = outer_px
        out[y1 - B : y1, x0:x1] = outer_px
        out[y0:y1, x0 : x0 + B] = outer_px
        out[y0:y1, x1 - B : x1] = outer_px

        if original_dtype == np.uint8:
            return np.clip(out * 255.0, 0, 255).astype(np.uint8)
        return np.clip(out, 0.0, 1.0)

    # ---------------------------------------------
    # Helper para iterar solo en el borde (evita recorrer toda la imagen)
    # ---------------------------------------------
    def _iter_border_pixels():
        # Top
        for yy in range(y0, y0 + B):
            for xx in range(x0, x1):
                yield xx, yy
        # Bottom
        for yy in range(y1 - B, y1):
            for xx in range(x0, x1):
                yield xx, yy
        # Sides (sin repetir esquinas ya cubiertas por top/bottom)
        for yy in range(y0 + B, y1 - B):
            for xx in range(x0, x0 + B):
                yield xx, yy
            for xx in range(x1 - B, x1):
                yield xx, yy

    # Denominadores seguros
    denom_u_w = float(max(pa_w - 1, 1))
    denom_u_h = float(max(pa_h - 1, 1))
    denom_v = float(max(B - 1, 1))
    denom_t = float(max(B - 1, 1))

    # ---------------------------------------------
    # Apply border (gradient / rough / image)
    # ---------------------------------------------
    for x, y in _iter_border_pixels():
        if style != "image":
            # Distancia al borde de la PAINT_AREA (no al borde global)
            d = min(
                x - x0,
                (x1 - 1) - x,
                y - y0,
                (y1 - 1) - y,
            )
            t = np.clip(d / denom_t, 0.0, 1.0)
            base_color = _lerp(outer_px, inner_px, t)

            if style == "gradient":
                out[y, x] = base_color
            else:  # rough
                n = (np.random.rand() * 2.0) - 1.0
                n_eff = n * float(noise_strength) * (1.0 - t)
                rough = base_color * (1.0 + n_eff)
                out[y, x] = np.clip(rough, 0.0, 1.0)

        else:
            # Determinar lado + coords normalizadas dentro de paint_area
            if y < y0 + B:  # TOP
                u = (x - x0) / denom_u_w
                v = (y - y0) / denom_v
            elif y >= y1 - B:  # BOTTOM
                u = (x - x0) / denom_u_w
                v = ((y1 - 1) - y) / denom_v
            elif x < x0 + B:  # LEFT
                u = (y - y0) / denom_u_h
                v = (x - x0) / denom_v
            else:  # RIGHT
                u = (y - y0) / denom_u_h
                v = ((x1 - 1) - x) / denom_v

            # Tiling solo a lo largo del borde
            tile_density = 4.0
            tu = (u * tile_density) % 1.0
            tv = v

            fx_f = tu * (fw - 1)
            fy_f = tv * (fh - 1)

            color = _bilinear_sample(fi, fx_f, fy_f)

            if c == 4 and color.shape[0] == 3:
                color = np.array([color[0], color[1], color[2], 1.0], dtype=np.float32)

            # Gamma suave (solo RGB)
            gamma = 0.9
            color[:3] = np.clip(color[:3], 0.0, 1.0) ** gamma

            # Shading por profundidad
            shade_strength = 0.15
            shade = 1.0 - shade_strength * (1.0 - v)
            color[:3] *= shade

            # Ruido opcional
            ns = float(noise_strength)
            if ns > 0.0:
                n = (np.random.rand() * 2.0) - 1.0
                color[:3] *= (1.0 + n * ns)

            out[y, x] = np.clip(color, 0.0, 1.0)

    # ---------------------------------------------
    # Restore dtype original
    # ---------------------------------------------
    if original_dtype == np.uint8:
        return np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return np.clip(out, 0.0, 1.0)
