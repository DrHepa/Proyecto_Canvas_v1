from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class GenerationRequest:
    """Request inmutable para generar un .pnt.

    - writer_mode='legacy_copy': requiere base_pnt_path (template físico).
    - writer_mode='raster20': base_pnt_path debe ser None; se escribe header20 + raster (width*height).
    - writer_mode='preserve_source': requiere base_pnt_path header20; copia bytes base y parchea solo el raster (conserva suffix).
    - writer_mode='auto': (compat) se trata como raster20.
    """

    # Imagen preparada (uint8 RGBA) con tamaño final del raster lógico
    image_rgba: np.ndarray
    image_is_final: bool

    # Template base (solo legacy_copy; raster20 admite None)
    base_pnt_path: Optional[Path]

    # Raster lógico final
    width: int
    height: int

    # Configuración
    dithering: Dict[str, Any]
    border: Dict[str, Any]
    alpha_threshold: int

    # Output
    output_path: Path

    # Extras
    template_id: Optional[str] = None  # e.g. Doggo_Character_BP_C
    writer_mode: str = "raster20"  # auto | legacy_copy | raster20 | preserve_source

    # Encode paint area (subrect); por defecto None (=full raster lógico)
    encode_paint_area: Optional[dict] = None

    # Encode visibility mask (bool array h×w)
    encode_visibility_mask: Optional[np.ndarray] = None

    # Planks (sub-paint area)
    planks: Optional[list[dict]] = None

    # Visible rows (mapping)
    encode_visible_rows: Optional[list[int]] = None

    # Enabled dyes (color palette)
    enabled_dyes: Optional[set[int]] = None
