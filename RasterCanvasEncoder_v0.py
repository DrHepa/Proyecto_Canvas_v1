# ==========================================================
# RasterCanvasEncoder_dev.py
#
# Encoder raster puro para canvas ARK
# - Layout físico siempre extraído del template real
# - Soporte opcional de dithering (Floyd–Steinberg)
# ==========================================================

from pathlib import Path
import struct
import numpy as np
import os
from time import perf_counter

from PntColorTranslator_v0 import PntColorTranslatorV1
from RasterLayoutExtractor_v0 import RasterLayoutExtractor
from PntIO import peek_pnt_info
from ErrorDiffusion_v1 import ed_quantize_to_bytes, ordered_quantize_to_bytes


class RasterCanvasEncoder:
    """
    Encoder para canvas raster puros.

    Modelo:
        offset = header_size + y * stride + x
    """

    def __init__(
        self,
        header_size: int = 20,
        tabla_dyes_path: Path = None,
        alpha_threshold: int = 10,
        enabled_dyes: set[int] | None = None,
    ):
        self.header_size = header_size
        self.alpha_threshold = alpha_threshold

        if tabla_dyes_path is None:
            raise ValueError("tabla_dyes_path es obligatorio")

        self.color_translator = PntColorTranslatorV1(
            str(tabla_dyes_path),
            enabled_dyes=enabled_dyes,
        )
        self.layout_extractor = RasterLayoutExtractor(header_size=header_size)

    # ------------------------------------------------------

    def encode(
        self,
        base_pnt_path: Path | None,
        image_rgba: np.ndarray,
        output_pnt_path: Path,
        *,
        writer_mode: str = "legacy_copy",  # legacy_copy | raster20

        layout=None,          # se mantiene por compatibilidad
        width: int,
        height: int,
        dither_mode: str = "none",
        dither_strength: float = 1.0,
        dither_kernel: str = "floyd_steinberg",
        dither_serpentine: bool = True,
        encode_paint_area: dict | None = None,
        planks: list[dict] | None = None,
        encode_visible_rows: list[int] | None = None,
        encode_visibility_mask: np.ndarray | None = None,
    ):
        """
        Genera un .pnt raster a partir de una imagen RGBA.

        - Layout físico SIEMPRE extraído del template real
        - width / height definen el raster lógico
        - dithering opcional antes de cuantizar
        """

        # --------------------------------------------------
        # PERF (debug): activar con PC_PERF=1
        # --------------------------------------------------
        perf_enabled = os.environ.get("PC_PERF", "0") == "1"
        t0 = perf_counter() if perf_enabled else 0.0
        t_alloc = t_pre = t_dither = t_encode = t_write = 0.0

        # --------------------------------------------------
        # Resolver layout físico (legacy) o virtual (raster20)
        # --------------------------------------------------

        writer_mode = (writer_mode or "raster20").strip().lower()

        if writer_mode == "raster20":
            # Layout virtual: row-major contiguous
            if self.header_size != 20:
                raise ValueError(f"Raster20 requiere header_size=20, pero es {self.header_size}")

            stride = int(width)
            header_size = 20
            buffer_rows = int(height)

            # Construir archivo desde cero: header20 + raster
            px = stride * buffer_rows
            output = bytearray(struct.pack("<IIIII", 0, stride, buffer_rows, 0, px) + (b"\x00" * px))
            buffer_limit = len(output)

            if perf_enabled:
                t_alloc = perf_counter()

        elif writer_mode == "preserve_source":
            # Header20 + suffix: copiar el archivo base completo y parchear SOLO el raster.
            if base_pnt_path is None:
                raise ValueError("writer_mode=preserve_source requiere base_pnt_path")

            info = peek_pnt_info(base_pnt_path)
            if not info.get("is_header20"):
                raise ValueError("preserve_source requiere un .pnt header20 contiguo")

            bw = int(info.get("width", 0))
            bh = int(info.get("height", 0))
            if bw != int(width) or bh != int(height):
                raise ValueError(
                    f"preserve_source: dims del base ({bw}x{bh}) no coinciden con el encode ({width}x{height})"
                )

            stride = int(width)
            header_size = 20
            buffer_rows = int(height)

            base_bytes = bytearray(Path(base_pnt_path).read_bytes())
            output = bytearray(base_bytes)
            # Limitar escrituras al raster para no tocar suffix.
            buffer_limit = header_size + stride * buffer_rows

            if perf_enabled:
                t_alloc = perf_counter()

        elif writer_mode == "legacy_copy":
            # Legacy: extraer layout real del template y copiar bytes base
            if base_pnt_path is None:
                raise ValueError("writer_mode=legacy_copy requiere base_pnt_path")

            real_layout = self.layout_extractor.extract(base_pnt_path)

            stride = real_layout.row_length
            header_size = real_layout.header_size
            buffer_rows = real_layout.row_count

            base_bytes = bytearray(base_pnt_path.read_bytes())
            output = bytearray(base_bytes)
            buffer_limit = len(output)

            if perf_enabled:
                t_alloc = perf_counter()

        else:
            raise ValueError(f"writer_mode inválido: {writer_mode}")

        # --------------------------------------------------
        # Validaciones básicas (seguras)
        # --------------------------------------------------

        if image_rgba.shape[0] != height or image_rgba.shape[1] != width:
            raise ValueError(
                f"Imagen debe ser {width}x{height}, "
                f"pero es {image_rgba.shape[1]}x{image_rgba.shape[0]}"
            )

        if height > buffer_rows:
            raise ValueError(
                f"Height lógico ({height}) excede row_count físico ({buffer_rows})"
            )

        if width > stride:
            raise ValueError(
                f"Width lógico ({width}) excede stride físico ({stride})"
            )

        if encode_visibility_mask is not None:
            if encode_visibility_mask.shape != (height, width):
                raise ValueError(
                    "encode_visibility_mask debe tener el mismo tamaño que el raster lógico"
                )
        # --------------------------------------------------
        # Resolver encode_paint_area (si existe)
        # --------------------------------------------------

        if encode_paint_area is not None:
            off_x = int(encode_paint_area.get("offset_x", 0))
            off_y = int(encode_paint_area.get("offset_y", 0))
            enc_w = int(encode_paint_area.get("width", width))
            enc_h = int(encode_paint_area.get("height", height))
        else:
            off_x = 0
            off_y = 0
            enc_w = width
            enc_h = height

        # --- FIX: validar/normalizar contra raster LÓGICO (flags recortadas, etc.)
        if off_x < 0 or off_y < 0:
            raise ValueError("encode_paint_area offset negativo")

        # Si paint_area viene de un raster mayor (p.ej. 256) pero la imagen lógica ya está recortada
        if (off_x + enc_w > width) or (off_y + enc_h > height):
            if enc_w == width and off_x != 0:
                off_x = 0
            if enc_h == height and off_y != 0:
                off_y = 0

        if off_x + enc_w > width:
            enc_w = max(0, min(enc_w, width - off_x))
        if off_y + enc_h > height:
            enc_h = max(0, min(enc_h, height - off_y))

        if enc_w <= 0 or enc_h <= 0:
            raise ValueError("encode_paint_area inválido tras normalización")

        # Validación física (layout del template)
        if off_x + enc_w > stride:
            raise ValueError("encode_paint_area excede stride físico")

        if off_y + enc_h > buffer_rows:
            raise ValueError("encode_paint_area excede row_count físico")


        # --------------------------------------------------
        # Resolver encode_visible_rows (si existe)
        # --------------------------------------------------
        if encode_visible_rows is None:
            y_iter = range(enc_h)
            y_map = lambda y: y + off_y
        else:
            y_iter = range(min(enc_h, len(encode_visible_rows)))
            y_map = lambda y: encode_visible_rows[y]

        # --------------------------------------------------
        # Resolver planks (sub-paint-area)
        # --------------------------------------------------

        if not planks:
            # Caso legacy: un solo "plank" que cubre todo el encode_paint_area
            plank_iter = [{
                "y0": off_y,
                "y1": off_y + enc_h - 1,
                "x_offset": 0,
                "width": enc_w,
                "flip_x": False,
            }]
        else:
            plank_iter = []
            for p in planks:
                y0, y1 = p["y"]
                plank_iter.append({
                    "y0": y0,
                    "y1": y1,
                    "x_offset": int(p.get("x_offset", 0)),
                    "width": int(p.get("width", enc_w)),
                    "flip_x": bool(p.get("flip_x", False)),
                })

        # --------------------------------------------------
        # Preprocesado: RGBA -> RGB lineal
        # --------------------------------------------------

        # --------------------------------------------------
        # Fast-path (caso común):
        # - sin planks
        # - sin ordered/fs dithering
        # - sin encode_visible_rows / encode_visibility_mask
        # - paint_area como ROI
        #
        # Evita bucles Python por píxel: cuantización en batch + asignación vectorizada.
        # --------------------------------------------------
        fast_path = (
            (not planks) and
            (encode_visible_rows is None) and
            (encode_visibility_mask is None) and
            (dither_mode in ("none", ""))
        )

        if fast_path:
            out_raster = np.frombuffer(
                output,
                dtype=np.uint8,
                offset=header_size,
                count=stride * buffer_rows,
            ).reshape((buffer_rows, stride))

            src = image_rgba[off_y:off_y + enc_h, off_x:off_x + enc_w]
            if src.size:
                alpha = src[..., 3] >= self.alpha_threshold
                if perf_enabled:
                    _t = perf_counter()

                if np.any(alpha):
                    rgb_linear = src[..., :3].astype(np.float32) / 255.0
                    rgb_sel = rgb_linear[alpha]

                    if perf_enabled:
                        t_pre = perf_counter()

                    bytes_sel = self.color_translator.nearest_bytes_batch(rgb_sel)
                    out_raster[off_y:off_y + enc_h, off_x:off_x + enc_w][alpha] = bytes_sel
                else:
                    if perf_enabled:
                        t_pre = perf_counter()

            if perf_enabled:
                t_dither = t_pre
                t_encode = perf_counter()

            goto_write = True
        else:
            goto_write = False

            rgb_linear = image_rgba[..., :3].astype(np.float32) / 255.0

            if perf_enabled:
                t_pre = perf_counter()

        # --------------------------------------------------
        # Fast-path completed: write and return (do NOT fall through)
        # --------------------------------------------------
        if goto_write:
            output_pnt_path.parent.mkdir(parents=True, exist_ok=True)
            output_pnt_path.write_bytes(output)
            return

        # --------------------------------------------------
        # Bytes map precompute (non-fast path)
        # - evita match_linear_rgb por píxel
        # - aplica dither directamente a bytes
        # --------------------------------------------------

        rgb_roi = image_rgba[off_y:off_y + enc_h, off_x:off_x + enc_w, :3].astype(np.float32) / 255.0
        alpha_roi = image_rgba[off_y:off_y + enc_h, off_x:off_x + enc_w, 3] >= self.alpha_threshold

        if encode_visibility_mask is not None:
            vis_roi = encode_visibility_mask[off_y:off_y + enc_h, off_x:off_x + enc_w]
            active_roi = alpha_roi & vis_roi
        else:
            active_roi = alpha_roi

        pack = self.color_translator.palette_pack()

        if dither_mode in ("fs", "palette_fs"):
            bytes_roi = ed_quantize_to_bytes(
                rgb_roi,
                active_roi,
                pack["palette_w"],
                pack["palette_w_norm2"],
                pack["palette_bytes"],
                pack["palette_linear"],
                kernel=str(dither_kernel or "floyd_steinberg"),
                strength=dither_strength,
                serpentine=bool(dither_serpentine),
                respect_mask=True,
                clamp01=True,
            )

        elif dither_mode in ("ordered", "palette_ordered"):
            bytes_roi = ordered_quantize_to_bytes(
                rgb_roi,
                active_roi,
                palette_w=pack["palette_w"],
                palette_w_norm2=pack["palette_w_norm2"],
                palette_bytes=pack["palette_bytes"],
                sqrt_w=pack.get("sqrt_w"),
                strength=dither_strength,
            )

        else:
            bytes_roi = np.zeros((enc_h, enc_w), dtype=np.uint8)
            if np.any(active_roi):
                bytes_sel = self.color_translator.nearest_bytes_batch(rgb_roi[active_roi])
                bytes_roi[active_roi] = bytes_sel


        # --------------------------------------------------
        # Raster encoding (con soporte de planks y rotación)
        # --------------------------------------------------

        for plank in plank_iter:
            x0 = plank["x_offset"]
            w  = plank["width"]
            flip_x = plank["flip_x"]

            y0 = plank["y0"]
            y1 = plank["y1"]
            plank_h = y1 -y0 +1

            for y in y_iter:
                dst_y = y_map(y)

                local_y = dst_y -y0

                if flip_x:
                    src_y = (y0 - off_y) + (plank_h - 1 - local_y)
                else:
                    src_y = dst_y - off_y

                # Filtrar por rango Y del plank (Y global)
                if dst_y < plank["y0"] or dst_y > plank["y1"]:
                    continue

                # Seguridad: limitar al encode_paint_area global
                if dst_y < off_y or dst_y >= off_y + enc_h:
                    continue

                row_base = header_size + dst_y * stride

                for i in range(w):
                    # Resolver X lógico y X destino (flip real)
                    src_x = i
                    dst_x = x0 + (w - 1 - i if flip_x else i)
                    dst_x_canvas = dst_x + off_x

                    # Seguridad: limitar a encode_paint_area
                    if dst_x_canvas < off_x or dst_x_canvas >= off_x + enc_w:
                        continue

                    # Seguridad extra: src dentro de ROI
                    if src_y < 0 or src_y >= enc_h or src_x < 0 or src_x >= enc_w:
                        continue

                    # active_roi ya incluye alpha_threshold + encode_visibility_mask (ROI)
                    if not active_roi[src_y, src_x]:
                        continue

                    offset = row_base + dst_x_canvas
                    if offset >= buffer_limit:
                        raise RuntimeError(
                            f"Offset fuera de rango: {offset} >= {buffer_limit}"
                        )

                    output[offset] = int(bytes_roi[src_y, src_x])

        # --------------------------------------------------
        # Escritura
        # --------------------------------------------------

        output_pnt_path.parent.mkdir(parents=True, exist_ok=True)
        output_pnt_path.write_bytes(output)
