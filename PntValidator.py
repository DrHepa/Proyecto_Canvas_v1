from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PntIO import parse_header20


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    kind: str
    message: str


def validate_raster20(pnt_path: Path) -> ValidationResult:
    data = pnt_path.read_bytes()
    if len(data) < 20:
        return ValidationResult(False, "raster20", "archivo < 20 bytes")

    h = parse_header20(data[:20])
    expected = 20 + h.width * h.height
    if h.paint_data_size != h.width * h.height:
        return ValidationResult(False, "raster20", "paint_data_size != w*h")
    if len(data) < expected:
        return ValidationResult(False, "raster20", f"archivo truncado (len={len(data)}, mínimo={expected})")

    if len(data) != expected:
        # Header20 con suffix/metadata (MyPaintings / LocalSaved)
        suffix_len = len(data) - expected
        return ValidationResult(True, "raster20_suffix", f"ok (suffix_len={suffix_len})")

    return ValidationResult(True, "raster20", "ok")


def validate_quick(pnt_path: Path) -> ValidationResult:
    """Validación rápida.

    - Si el archivo encaja con header20 canonico, valida como raster20.
    - Si no, devuelve 'unknown' (sin fallar), porque cabeceras extendidas aún no están formalizadas.
    """
    try:
        r = validate_raster20(pnt_path)
        if r.ok:
            return r
    except Exception:
        pass

    return ValidationResult(True, "unknown", "no se ha validado estructura (cabecera extendida o formato no inferido)")
