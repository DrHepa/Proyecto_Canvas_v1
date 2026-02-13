from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

from RasterLayoutExtractor_v0 import RasterLayoutExtractor, RasterLayoutInfoV2


@dataclass(frozen=True)
class Header20:
    unknown0: int
    width: int
    height: int
    unknown12: int
    paint_data_size: int


def parse_header20(raw: bytes) -> Header20:
    if len(raw) < 20:
        raise ValueError("raw demasiado pequeño para header20")
    u0, w, h, u12, p = struct.unpack_from("<IIIII", raw, 0)
    return Header20(u0, w, h, u12, p)


def looks_like_header20(pnt_bytes: bytes) -> bool:
    if len(pnt_bytes) < 20:
        return False
    h = parse_header20(pnt_bytes[:20])
    if h.width == 0 or h.height == 0:
        return False
    if h.width > 4096 or h.height > 4096:
        return False
    # En header20 canónico, paint_data_size == w*h
    if h.paint_data_size != h.width * h.height:
        return False
    # Header20 puede tener suffix/metadata (MyPaintings / LocalSaved).
    # Aceptamos len >= 20 + paint_data_size.
    if len(pnt_bytes) < 20 + h.paint_data_size:
        return False
    return True


def read_header20(pnt_path: Path) -> Header20:
    data = pnt_path.read_bytes()
    return parse_header20(data[:20])


def read_raster20(pnt_path: Path) -> np.ndarray:
    """Lee un .pnt con header20 y raster contiguo.

    Devuelve un array uint8 de forma (height, width).
    """
    data = pnt_path.read_bytes()
    if not looks_like_header20(data):
        raise ValueError("El archivo no parece ser un raster20 (header20 contiguo)")
    h = parse_header20(data[:20])
    raster = np.frombuffer(data[20:20 + h.paint_data_size], dtype=np.uint8).reshape((h.height, h.width))
    return raster


def peek_pnt_info(pnt_path: Path) -> dict:
    """Inspección rápida del layout sin leer el archivo entero.

    Soporta header20 con o sin suffix.
    Retorna un dict con:
      - is_header20: bool
      - width,height
      - raster_offset (normalmente 20)
      - raster_len (w*h)
      - has_suffix, suffix_len
      - file_size
    """
    p = Path(pnt_path)
    st = p.stat()
    size = int(st.st_size)
    if size < 20:
        return {"is_header20": False, "file_size": size}

    with p.open("rb") as f:
        head = f.read(20)

    try:
        h = parse_header20(head)
    except Exception:
        return {"is_header20": False, "file_size": size}

    # Basic sanity
    if h.width <= 0 or h.height <= 0 or h.width > 4096 or h.height > 4096:
        return {"is_header20": False, "file_size": size}
    if h.paint_data_size != h.width * h.height:
        return {"is_header20": False, "file_size": size}

    expected = 20 + int(h.paint_data_size)
    if size < expected:
        return {"is_header20": False, "file_size": size}

    return {
        "is_header20": True,
        "width": int(h.width),
        "height": int(h.height),
        "raster_offset": 20,
        "raster_len": int(h.paint_data_size),
        "has_suffix": size > expected,
        "suffix_len": max(0, size - expected),
        "file_size": size,
        "header": h,
    }


def read_legacy_raster(pnt_path: Path, *, header_size: int = 20) -> Tuple[np.ndarray, RasterLayoutInfoV2]:
    """Lee un .pnt que usa layout extraído por RasterLayoutExtractor (header_size fijo).

    Nota: esto NO soporta cabeceras extendidas desconocidas.

    Devuelve:
      - raster uint8 shape(row_count,row_length)
      - layout info
    """
    ext = RasterLayoutExtractor(header_size=header_size)
    layout = ext.extract(pnt_path)
    data = pnt_path.read_bytes()
    raster = np.frombuffer(data[layout.header_size:], dtype=np.uint8).reshape((layout.row_count, layout.row_length))
    return raster, layout


def read_best_effort(pnt_path: Path, *, header_size: int = 20) -> Tuple[np.ndarray, dict]:
    """Intenta leer un .pnt en el mejor modo posible.

    Soporta:
      - raster20 (contiguo)
      - legacy con header_size fijo (default 20)

    Retorna:
      (raster2d, meta)
    """
    data = pnt_path.read_bytes()
    if looks_like_header20(data):
        h = parse_header20(data[:20])
        raster = np.frombuffer(data[20:20 + h.paint_data_size], dtype=np.uint8).reshape((h.height, h.width))
        expected = 20 + int(h.paint_data_size)
        return raster, {
            "kind": "raster20",
            "header_size": 20,
            "row_length": h.width,
            "row_count": h.height,
            "width": h.width,
            "height": h.height,
            "has_suffix": len(data) > expected,
            "suffix_len": max(0, len(data) - expected),
        }

    raster, layout = read_legacy_raster(pnt_path, header_size=header_size)
    return raster, {
        "kind": "legacy_guess",
        "header_size": layout.header_size,
        "row_length": layout.row_length,
        "row_count": layout.row_count,
    }
