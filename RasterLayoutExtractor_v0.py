# ==========================================================
# RasterLayoutExtractor.py
#
# Extracción automática de layout raster desde un .pnt
# - header_size fijo (conocido)
# - buffer raster lineal
# - detección robusta de row_length y row_count
# ==========================================================

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import math
import numpy as np


# ----------------------------------------------------------
# Estructura de salida
# ----------------------------------------------------------

@dataclass(frozen=True)
class RasterLayoutInfoV2:
    header_size: int
    row_length: int
    row_count: int
    paint_data_size: int


# ----------------------------------------------------------
# Extractor
# ----------------------------------------------------------

class RasterLayoutExtractor:
    """
    Extrae automáticamente el layout raster (row_length, row_count)
    a partir de un archivo .pnt, asumiendo:

    - Cabecera fija y común
    - Buffer raster lineal
    - Sin padding entre filas
    """

    def __init__(
        self,
        header_size: int,
        min_row_length: int = 8,
        max_row_length: int = 4096,
    ):
        if header_size <= 0:
            raise ValueError("header_size debe ser > 0")

        self.header_size = header_size
        self.min_row_length = min_row_length
        self.max_row_length = max_row_length

    # ------------------------------------------------------

    def extract(self, pnt_path: Path) -> RasterLayoutInfoV2:
        """
        Extrae el layout raster del .pnt.

        Retorna:
            RasterLayoutInfoV2
        """

        data = pnt_path.read_bytes()
        file_size = len(data)

        if file_size <= self.header_size:
            raise RuntimeError("Archivo demasiado pequeño para contener raster")

        paint_data = np.frombuffer(
            data[self.header_size:], dtype=np.uint8
        )

        N = paint_data.size

        # --------------------------------------------------
        # 1) Candidatos de row_length
        # --------------------------------------------------

        candidates = self._candidate_row_lengths(N)
        if not candidates:
            raise RuntimeError("No se encontraron candidatos de row_length")

        # --------------------------------------------------
        # 2) Evaluación raster
        # --------------------------------------------------

        scored: List[Tuple[float, int, int]] = []

        for row_length in candidates:
            row_count = N // row_length
            score = self._raster_continuity_score(
                paint_data, row_length, row_count
            )
            scored.append((score, row_length, row_count))

        # --------------------------------------------------
        # 3) Selección inequívoca
        # --------------------------------------------------

        scored.sort(key=lambda x: x[0])  # menor score = mejor

        best_score, best_row_length, best_row_count = scored[0]

        return RasterLayoutInfoV2(
            header_size=self.header_size,
            row_length=best_row_length,
            row_count=best_row_count,
            paint_data_size=N
        )

    # ------------------------------------------------------
    # Helpers
    # ------------------------------------------------------

    def _candidate_row_lengths(self, N: int) -> List[int]:
        """
        Devuelve divisores razonables de N como candidatos
        a row_length.
        """
        candidates = []

        for row_length in range(
            self.min_row_length,
            min(self.max_row_length, N) + 1
        ):
            if N % row_length != 0:
                continue

            # filtro adicional: row_count razonable
            row_count = N // row_length
            if row_count < 2:
                continue

            candidates.append(row_length)

        return candidates

    # ------------------------------------------------------

    def _raster_continuity_score(
        self,
        data: np.ndarray,
        row_length: int,
        row_count: int
    ) -> float:
        """
        Calcula un score de continuidad raster.

        Menor score = mejor raster.
        """

        img = data.reshape((row_count, row_length))

        # Diferencias horizontales
        dh = np.abs(img[:, :-1].astype(np.int16) - img[:, 1:].astype(np.int16))
        mean_dh = float(dh.mean())

        # Diferencias verticales
        dv = np.abs(img[:-1, :].astype(np.int16) - img[1:, :].astype(np.int16))
        mean_dv = float(dv.mean())

        # Penalización por discontinuidad vertical excesiva
        # (bandas)
        score = mean_dv / (mean_dh + 1e-6)

        # Penalización suave por imágenes demasiado "estrechas"
        aspect_penalty = abs(
            math.log2(row_length / max(row_count, 1))
        ) * 0.01

        return score + aspect_penalty
