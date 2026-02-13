from __future__ import annotations

from pathlib import Path

from GenerationRequest import GenerationRequest
from RasterCanvasEncoder_v0 import RasterCanvasEncoder
from PntValidator import validate_quick


class GenerationService:
    """Ejecuta GenerationRequest usando el encoder raster ARK."""

    @staticmethod
    def _map_dither_mode(mode: str | None) -> str:
        if not mode:
            return "none"
        if mode == "palette_fs":
            return "fs"
        if mode == "palette_ordered":
            return "ordered"
        return mode

    @staticmethod
    def _resolve_writer_mode(req: GenerationRequest) -> str:
        """Resolve writer_mode for the encoder.

        Nuevo contrato (simplificado):
        - raster20 es el flujo principal.
        - legacy_copy queda como compat/debug.
        - auto se trata como raster20.
        """
        mode = (req.writer_mode or "raster20").strip().lower()
        if mode == "auto":
            mode = "raster20"
        if mode not in ("legacy_copy", "raster20", "preserve_source"):
            mode = "raster20"
        return mode


    @staticmethod
    def run(request: GenerationRequest, *, tabla_dyes_path: Path, header_size: int = 20):
        """Genera el .pnt.

        Nota: el header_size aquí controla el tamaño del prefijo cuando writer_mode='raster20'.
        En legacy_copy, el encoder copia la cabecera del template.
        """

        encoder = RasterCanvasEncoder(
            header_size=header_size,
            tabla_dyes_path=tabla_dyes_path,
            alpha_threshold=request.alpha_threshold,
            enabled_dyes=request.enabled_dyes,
        )

        dcfg = request.dithering or {}
        d_mode = GenerationService._map_dither_mode(dcfg.get("mode", "none"))
        d_strength = float(dcfg.get("strength", 1.0))
        d_strength = max(0.0, min(1.0, d_strength))

        writer_mode = GenerationService._resolve_writer_mode(request)

        encoder.encode(
            base_pnt_path=request.base_pnt_path,
            image_rgba=request.image_rgba,
            output_pnt_path=request.output_path,
            width=request.width,
            height=request.height,
            dither_mode=d_mode,
            dither_strength=d_strength,
            encode_paint_area=request.encode_paint_area,
            encode_visibility_mask=request.encode_visibility_mask,
            planks=request.planks,
            encode_visible_rows=request.encode_visible_rows,
            writer_mode=writer_mode,
        )

        r = validate_quick(request.output_path)
        if not r.ok:
            print(f"[WARN] .pnt inválido: {r.kind} | {r.message}")
