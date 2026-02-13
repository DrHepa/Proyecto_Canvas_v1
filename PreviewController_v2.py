# PreviewController_v1.py
# Proyecto Canvas — Preview + Generation controller
# Cleaned: no legacy dead code, no hidden side-effects, thread-safe snapshot rendering

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import copy
import threading
from collections import OrderedDict

import numpy as np
from PIL import Image

from FrameBorder import apply_frame_border
from GenerationRequest import GenerationRequest
from GenerationService import GenerationService
from PntColorTranslator_v0 import PntColorTranslatorV1
from PntIO import peek_pnt_info
from TemplateDescriptorLoader import TemplateDescriptorLoader
from paths import get_app_root

import os

# Debug-only: allow APC paint_area profile toggling if PC_ENABLE_APC_DEBUG=1
PC_ENABLE_APC_DEBUG = os.environ.get('PC_ENABLE_APC_DEBUG', '0') == '1'


# ======================================================
# AppState (explicit, no dynamic attributes)
# ======================================================

@dataclass
class AppState:
    # Input
    image_original: Optional[Image.Image] = None

    # Template selection / resolved
    template: Optional[dict] = None                 # resolved physical template (or abstract if selected)
    preview_descriptor: Optional[dict] = None       # descriptor used to render preview (can be abstract)
    selected_template_id: Optional[str] = None      # helps cache keys & UI cohesion

    # External mode (advanced): when set, the current descriptor represents an on-disk .pnt
    # from MyPaintings / LocalSaved.
    external_pnt_path: Optional[Path] = None

    # Canvas resolution (physical raster)
    canvas_resolved: Optional[dict] = None          # {"width","height","paint_area", "planks", "meta"}

    # Requests (dynamic / multi-canvas)
    canvas_request: Optional[dict] = None           # dynamic: rows/blocks + paint_area; multi: rows/cols
    canvas_is_dynamic: bool = False

    # Fixed templates: choose which paint_area profile to use
    # - "project": descriptor.encode.paint_area (validated in-game)
    # - "apc": optional descriptor.meta.apc.paint_area (ArkPaintingConverter reference)
    paint_area_profile: str = "project"

    # Preview
    preview_ready: bool = False
    preview_mode: str = "visual"                    # "visual" | "ark_simulation"
    palette: Optional[dict] = None                  # {"mode","quantize_fn","translator"} or None

    # Options
    enabled_dyes: Optional[set[int]] = None
    dithering_config: dict = field(default_factory=lambda: {"mode": "none", "strength": 0.5})
    border_config: dict = field(default_factory=lambda: {"style": "none", "size": 0, "frame_image": None})
    game_object_type: Optional[str] = None
    show_game_object: bool = False

    # Writer
    writer_mode: str = "raster20"  # auto | legacy_copy | raster20 | preserve_source

    # External .pnt source (advanced)
    external_pnt_path: Optional[Path] = None


# ======================================================
# PreviewController
# ======================================================

class PreviewController:
    """
    Controller: holds the current state, renders previews, and builds generation requests.

    Design constraints:
    - NO hidden side-effects (e.g., generation must not mutate current template selection).
    - Thread-safe snapshot-based preview rendering for background worker.
    - Multi-canvas preview cache (LRU) keyed by (image_rev + parameters).
    """

    def __init__(self, *, templates_root: Path):
        self.state = AppState()

        self.template_loader = TemplateDescriptorLoader(templates_root=templates_root)
        self.template_assets_root = templates_root

        # --------------------------------------------------
        # Preview performance: cap preview render resolution.
        # This only affects preview rendering (background worker). Export (.pnt)
        # continues to use the full raster resolution.
        # Override with env var PC_PREVIEW_MAX_DIM (e.g., 384, 512, 768).
        # --------------------------------------------------
        import os
        try:
            self.preview_max_dim = int(os.getenv("PC_PREVIEW_MAX_DIM", "512"))
        except Exception:
            self.preview_max_dim = 512
        self.preview_max_dim = max(64, min(self.preview_max_dim, 2048))

        # Generation-time extras (kept for future; do not auto-derive here)
        self._planks_override: Optional[list[dict]] = None

        # Prepared image cache for generation (depends on image+canvas+border)
        self._prepared_image_rgba: Optional[np.ndarray] = None

        # Translator (ARK dyes)
        tabla_path = get_app_root() / "TablaDyes_v1.json"
        self._ark_translator = PntColorTranslatorV1(str(tabla_path)) if tabla_path.exists() else None

        # --------------------------------------------------
        # Best dyes ranking cache (per image revision)
        # --------------------------------------------------
        self._best_dyes_lock = threading.Lock()
        self._best_dyes_rank_rev: Optional[int] = None
        self._best_dyes_rank: Optional[list[int]] = None

        # --------------------------------------------------
        # Multi-canvas preview cache (LRU) + image revision
        # --------------------------------------------------
        self._image_rev = 0
        self._mc_cache_lock = threading.Lock()
        self._mc_preview_cache: OrderedDict[tuple, Image.Image] = OrderedDict()
        self._mc_cache_max_items = 8
        self._mc_cache_hits = 0
        self._mc_cache_misses = 0

        # Last generation target (optional, used by GUI)
        self._last_generated_path: Optional[Path] = None

    # ==================================================
    # Template helpers
    # ==================================================

    @staticmethod
    def resolve_dynamic_physical_template_id(descriptor: dict, canvas_request: dict) -> str:
        """
        Resolve a physical template id for a dynamic template descriptor (schema 1.1).
        Chooses the smallest supported square size >= required max(rows_y, blocks_x).
        """
        dynamic = descriptor["dynamic"]
        required = max(int(canvas_request["blocks_x"]), int(canvas_request["rows_y"]))

        values = dynamic["values"]
        chosen = None
        for v in values:
            if v >= required:
                chosen = v
                break
        if chosen is None:
            chosen = values[-1]

        naming_pattern = descriptor["naming"]["pattern"]
        pnt_name = naming_pattern.format(size=chosen)
        if not pnt_name.endswith(".pnt"):
            raise RuntimeError("naming.pattern no genera un .pnt")

        return pnt_name[:-4]  # strip ".pnt"

    # ==================================================
    # Setters (state mutation is explicit and minimal)
    # ==================================================

    def set_image(self, image: Optional[Image.Image]) -> None:
        self.state.image_original = image
        self._prepared_image_rgba = None

        self._image_rev += 1
        self._invalidate_multicanvas_cache()

        with self._best_dyes_lock:
            self._best_dyes_rank_rev = None
            self._best_dyes_rank = None

        self._refresh()

    def set_writer_mode(self, mode: str) -> None:
        mode = (mode or 'raster20').strip().lower()
        if mode not in ('auto', 'legacy_copy', 'raster20', 'preserve_source'):
            raise ValueError(f'writer_mode inválido: {mode}')
        self.state.writer_mode = mode
        # IMPORTANTE: la imagen preparada depende del writer (legacy suele usar padding cuadrado)
        self._prepared_image_rgba = None
        self._refresh()

    def get_effective_writer_mode(self, descriptor: dict | None = None) -> str:
        """Resuelve el writer_mode final.

        Nuevo contrato (simplificado):
        - raster20 es el flujo principal (por defecto).
        - legacy_copy queda como compat/debug y requiere template físico.
        - auto se trata como raster20.
        """
        mode = (self.state.writer_mode or 'raster20').strip().lower()
        if mode == 'auto':
            mode = 'raster20'

        d = descriptor or self.state.preview_descriptor or self.state.template

        # Virtual templates (e.g. UserMasks) have no physical base .pnt.
        # Force raster20 regardless of any transient preserve_source/legacy_copy flags.
        try:
            if isinstance(d, dict) and ((d.get('resolved') or {}).get('virtual') is True):
                return 'raster20'
        except Exception:
            pass
        if mode == 'legacy_copy':
            # Si no hay template físico, no se puede legacy_copy
            base_pnt = None
            if isinstance(d, dict):
                base_pnt = (d.get('resolved', {}) or {}).get('base_pnt_path')
                if base_pnt is None:
                    # Template físico: resolved.asset_dir + resolved.pnt
                    r = d.get('resolved', {}) or {}
                    if r.get('asset_dir') and r.get('pnt'):
                        base_pnt = 'template'
            if not base_pnt:
                return 'raster20'
            return 'legacy_copy'

        if mode == 'preserve_source':
            base_pnt = None
            if isinstance(d, dict):
                base_pnt = (d.get('resolved', {}) or {}).get('base_pnt_path')
                if base_pnt is None:
                    r = d.get('resolved', {}) or {}
                    if r.get('asset_dir') and r.get('pnt'):
                        base_pnt = 'template'
            if not base_pnt:
                return 'raster20'
            return 'preserve_source'

        return 'raster20'


    def set_template(self, template_id: str, params: Optional[dict] = None) -> None:
        """
        Resolve template descriptor and derive canvas_resolved if raster exists.
        - For fixed templates: sets state.template and state.preview_descriptor.
        - For dynamic templates: sets state.template but preserves state.preview_descriptor (the abstract dynamic).
        """
        self._prepared_image_rgba = None

        resolved = self.template_loader.resolve(template_id, params or {})
        self.state.template = resolved

        if not self.state.canvas_is_dynamic:
            self.state.preview_descriptor = resolved

        raster = resolved["layout"]["raster"]
        if raster is not None:
            self.state.canvas_resolved = {
                "width": int(raster["width"]),
                "height": int(raster["height"]),
                "paint_area": resolved["encode"]["paint_area"],
                "planks": resolved["encode"]["planks"],
                "meta": resolved["resolved"],
            }
        else:
            self.state.canvas_resolved = None

        self._evaluate_preview_ready()
        self._refresh()

    # ==================================================
    # External .pnt (advanced)
    # ==================================================

    def set_external_pnt(self, pnt_path: Path) -> None:
        """Activates an external header20 .pnt as the current descriptor.

        Intended for advanced mode (MyPaintings / LocalSaved).
        The external file is treated as a fixed single-canvas template with full raster paint_area.
        """
        p = Path(pnt_path)
        info = peek_pnt_info(p)
        if not info.get("is_header20"):
            raise ValueError("External .pnt debe ser header20 contiguo (raster20)")

        w = int(info.get("width", 0))
        h = int(info.get("height", 0))
        if w <= 0 or h <= 0:
            raise ValueError("External .pnt: dimensiones inválidas")

        # Descriptor mínimo compatible con el contrato schema 1.1.
        tid = f"external::{p.name}"
        desc = {
            "schema_version": "1.1",
            "identity": {
                "id": tid,
                "label": p.name,
                "category": "External",
                "type": "external_pnt",
                "base_template": None,
            },
            "layout": {
                "raster": {"width": w, "height": h},
                "dynamic": None,
            },
            "dynamic": None,
            "encode": {
                # None => full raster
                "paint_area": None,
                "planks": None,
            },
            "preview": {
                "mode": None,
                "overlay_dir": None,
                "base_name": None,
                "world_scale": None,
                "show_indices": None,
            },
            "constraints": {
                "max_colors": None,
                "alpha_mode": None,
                "palette_required": None,
            },
            "capabilities": {
                "supports_border": True,
                "supports_dithering": True,
                "supports_dynamic_resize": False,
            },
            "multi_canvas": None,
            "naming": {"pattern": None},
            "resolved": {
                "asset_dir": None,
                "pnt": None,
                "base_pnt_path": str(p),
            },
            "meta": {
                "external": {
                    "has_suffix": bool(info.get("has_suffix", False)),
                    "suffix_len": int(info.get("suffix_len", 0)),
                }
            },
        }

        # Reset dynamic/multi state
        self.state.canvas_is_dynamic = False
        self.state.canvas_request = None
        self.state.template = desc
        self.state.preview_descriptor = desc
        self.state.selected_template_id = tid
        self.state.external_pnt_path = p
        self.state.canvas_resolved = {
            "width": w,
            "height": h,
            "paint_area": None,
            "planks": None,
            "meta": desc.get("resolved"),
        }

        self._prepared_image_rgba = None
        self._invalidate_multicanvas_cache()
        self._refresh()

    def clear_external_pnt(self) -> None:
        """Clears external mode selection.

        Note: the GUI is expected to re-select a regular template after this.
        """
        if self.state.external_pnt_path is None:
            return

        self.state.external_pnt_path = None
        self.state.template = None
        self.state.preview_descriptor = None
        self.state.selected_template_id = None
        self.state.canvas_resolved = None
        self.state.canvas_request = None
        self.state.canvas_is_dynamic = False

        self._prepared_image_rgba = None
        self._invalidate_multicanvas_cache()
        self._refresh()

    def set_dynamic_canvas_request(self, *, rows_y: Optional[int], blocks_x: Optional[int], mode: str) -> None:
        """
        Set dynamic canvas logical request.
        This does NOT resolve the physical template; UI should call resolve_dynamic_physical_template_id
        and then set_template() with that id.
        """
        if not self.state.canvas_is_dynamic:
            self.state.canvas_request = None
            self.state.canvas_resolved = None
            self._prepared_image_rgba = None
            self._refresh()
            return

        if rows_y is None or blocks_x is None:
            self.state.canvas_request = None
            self.state.canvas_resolved = None
        else:
            self.state.canvas_request = {
                "mode": mode,
                "rows_y": int(rows_y),
                "blocks_x": int(blocks_x),
                "paint_area": {"offset_x": 0, "offset_y": 0, "width": int(blocks_x), "height": int(rows_y)},
            }

        self._prepared_image_rgba = None
        self._refresh()

    def set_multicanvas_request(self, *, rows: Optional[int] = None, cols: Optional[int] = None) -> None:
        """
        Stores a logical multi-canvas request. Defaults are taken from descriptor if missing.
        """
        req = {"mode": "multi_canvas"}
        if rows is not None:
            req["rows"] = int(rows)
        if cols is not None:
            req["cols"] = int(cols)

        self.state.canvas_request = req
        self._invalidate_multicanvas_cache()
        self._refresh()

    def set_paint_area_profile(self, profile: str) -> None:
        """Select which paint_area profile to use for fixed templates.

        - project: descriptor.encode.paint_area (validated in-game)
        - apc: descriptor.meta.apc.paint_area (ArkPaintingConverter reference)

        Manual by default (project).
        """
        p = (profile or "").strip().lower()
        if p not in ("project", "apc"):
            return
        if self.state.paint_area_profile == p:
            return

        if not PC_ENABLE_APC_DEBUG:
            return

        self.state.paint_area_profile = p
        self._prepared_image_rgba = None
        self._refresh()

    # ==================================================
    # Fixed paint_area helper
    # ==================================================

    def _get_fixed_paint_area(self, *, template: Optional[dict], descriptor: Optional[dict]) -> Optional[dict]:
        """Returns paint_area dict for fixed templates according to the selected profile.

        Returns None to indicate "full raster".
        """
        src = template or descriptor
        if not isinstance(src, dict):
            return None

        # Project (canonical in our pipeline)
        project_pa = (src.get("encode") or {}).get("paint_area")
        if project_pa == "full_raster":
            project_pa = None

        # APC (optional, non-canonical)
        apc_pa = (((src.get("meta") or {}).get("apc")) or {}).get("paint_area")

        chosen = None
        if PC_ENABLE_APC_DEBUG and self.state.paint_area_profile == "apc" and isinstance(apc_pa, dict):
            chosen = apc_pa
        elif isinstance(project_pa, dict):
            chosen = project_pa

        if not isinstance(chosen, dict):
            return None

        try:
            off_x = int(chosen.get("offset_x", 0))
            off_y = int(chosen.get("offset_y", 0))
            w = int(chosen.get("width", 0))
            h = int(chosen.get("height", 0))
        except Exception:
            return None

        if w <= 0 or h <= 0:
            return None

        return {"offset_x": off_x, "offset_y": off_y, "width": w, "height": h}

    def set_preview_mode(self, mode: str) -> None:
        if mode not in ("visual", "ark_simulation"):
            return

        self.state.preview_mode = mode

        if mode == "ark_simulation":
            self.state.palette = self._build_palette_object()
        else:
            self.state.palette = None

        self._invalidate_multicanvas_cache()
        self._refresh()

    def set_dithering_config(self, *, mode: str, strength: float) -> None:
        self.state.dithering_config = {"mode": mode, "strength": float(strength)}
        self._invalidate_multicanvas_cache()
        self._refresh()

    def set_enabled_dyes(self, enabled: Optional[set[int]]) -> None:
        self.state.enabled_dyes = enabled
        self._prepared_image_rgba = None

        if self.state.preview_mode == "ark_simulation":
            self.state.palette = self._build_palette_object()

        self._invalidate_multicanvas_cache()
        self._refresh()

    def set_border_style(self, style: str) -> None:
        self.state.border_config["style"] = style
        self._prepared_image_rgba = None
        self._refresh()

    def set_border_size(self, size: int) -> None:
        self.state.border_config["size"] = int(size)
        self._prepared_image_rgba = None
        self._refresh()

    def set_border_frame_image(self, image: Optional[Image.Image]) -> None:
        self.state.border_config["frame_image"] = image
        self._prepared_image_rgba = None
        self._refresh()

    def set_planks(self, planks: Optional[list[dict]]) -> None:
        """
        Optional planks override (advanced feature). Not used by default.
        """
        self._planks_override = planks

    # ==================================================
    # Refresh / Preview readiness
    # ==================================================

    def _evaluate_preview_ready(self) -> None:
        self.state.preview_ready = self.state.image_original is not None and self.state.canvas_resolved is not None

    def _refresh(self) -> None:
        self._evaluate_preview_ready()

    # ==================================================
    # Preview (sync)
    # ==================================================

    def set_dynamic_preview_canvas(self, *, width: int, height: int) -> None:
        """
        Preview-only canvas override (does not affect generation).
        """
        self.state.canvas_resolved = {"width": int(width), "height": int(height), "paint_area": "full_raster", "meta": {}}
        self._evaluate_preview_ready()

    def render_preview_if_possible(self) -> Optional[Image.Image]:
        """
        Render a preview for physical (single) canvas.
        For multi-canvas, use render_preview_multicanvas_cached().
        """
        if not self.state.preview_ready:
            return None

        canvas = self.state.canvas_resolved
        descriptor = self.state.preview_descriptor
        if canvas is None or descriptor is None:
            return None

        from PreviewRender_v1 import render_preview

        preview = descriptor.get("preview", {})
        overlay_def = None

        if preview.get("mode") == "mask" and bool(self.state.show_game_object):
            overlay_def = {}
            try:
                cw = int(canvas.get("width", 0) or 0)
                ch = int(canvas.get("height", 0) or 0)
            except Exception:
                cw, ch = 0, 0

            # Decorative overlay image (canonical) if available
            overlay_dir = (preview.get("overlay_dir") or "Dinos_Overlay").strip()
            base_name = (preview.get("base_name") or "").strip()
            if overlay_dir and base_name:
                img_path = self.template_assets_root / overlay_dir / f"{base_name}.png"
                if img_path.exists():
                    overlay_def["image"] = img_path

            # Effective mask for preview = same as encode visibility mask
            if cw > 0 and ch > 0:
                mbool = self._build_encode_visibility_mask(
                    template=self.state.template,
                    descriptor=descriptor,
                    width=cw,
                    height=ch,
                )
                if mbool is not None:
                    overlay_def["mask_alpha"] = (mbool.astype(np.uint8) * 255)
            if not overlay_def:
                overlay_def = None

        # IMPORTANT:
        # Use the same prepared RGBA used by Generation so paint_area (offsets) is respected.
        # This keeps preview <-> encode coherent for fixed templates like flags.
        descriptor_or_template = descriptor or self.state.template
        eff_writer = self.get_effective_writer_mode(descriptor_or_template)

        if self._prepared_image_rgba is None:
            img_np = self._prepare_base_image_rgba(eff_writer=eff_writer)
            self._prepared_image_rgba = img_np.copy()
        else:
            img_np = self._prepared_image_rgba.copy()

        prepared_img = Image.fromarray(img_np, mode="RGBA")

        # Border is already applied inside _prepare_base_image_rgba() only within paint_area.
        return render_preview(
            prepared_img,
            template_id=None,
            target_width=int(prepared_img.size[0]),
            target_height=int(prepared_img.size[1]),
            border=None,
            dithering=self.state.dithering_config,
            palette=self.state.palette if self.state.preview_mode == "ark_simulation" else None,
            preview_mode=self.state.preview_mode,
            game_object_type=self.state.game_object_type,
            overlay_def=overlay_def,
        )

    # ==================================================
    # Multi-canvas preview cache
    # ==================================================

    def _invalidate_multicanvas_cache(self) -> None:
        with self._mc_cache_lock:
            self._mc_preview_cache.clear()
            self._mc_cache_hits = 0
            self._mc_cache_misses = 0

    def _enabled_dyes_signature(self) -> tuple:
        ed = self.state.enabled_dyes
        if ed is None:
            return ("ALL",)
        return ("SET", tuple(sorted(ed)))

    def _mc_cache_key(
        self,
        *,
        template_id: str,
        rows: int,
        cols: int,
        preview_max_dim: int,
        preview_mode: str,
        dithering: dict,
        palette: Optional[dict],
        image_rev: int,
    ) -> tuple:
        d_mode = (dithering or {}).get("mode", "none")
        d_strength = round(float((dithering or {}).get("strength", 0.5)), 4)
        pal_sig = self._enabled_dyes_signature() if (preview_mode == "ark_simulation" and palette is not None) else None
        return (
            "mc",
            image_rev,
            template_id,
            int(rows),
            int(cols),
            int(preview_max_dim),
            str(preview_mode),
            str(d_mode),
            d_strength,
            pal_sig,
        )

    def render_preview_multicanvas_cached(self) -> Optional[Image.Image]:
        """
        Render a virtual preview of a multi-canvas, with LRU caching.
        """
        img_src = self.state.image_original
        descriptor = self.state.preview_descriptor
        if img_src is None or descriptor is None:
            return None

        identity = descriptor.get("identity", {})
        if identity.get("type") != "multi_canvas":
            return None

        multi = descriptor.get("multi_canvas")
        if not multi:
            return None

        rows = int(multi["rows"]["default"])
        cols = int(multi["cols"]["default"])

        req = self.state.canvas_request
        if req and req.get("mode") == "multi_canvas":
            rows = int(req.get("rows", rows))
            cols = int(req.get("cols", cols))

        template_id = self.state.selected_template_id or identity.get("id") or "multi_canvas"

        key = self._mc_cache_key(
            template_id=template_id,
            rows=rows,
            cols=cols,
            preview_max_dim=int(getattr(self, "preview_max_dim", 512)),
            preview_mode=self.state.preview_mode,
            dithering=self.state.dithering_config,
            palette=self.state.palette if self.state.preview_mode == "ark_simulation" else None,
            image_rev=self._image_rev,
        )

        with self._mc_cache_lock:
            hit = self._mc_preview_cache.get(key)
            if hit is not None:
                self._mc_cache_hits += 1
                self._mc_preview_cache.move_to_end(key)
                return hit

        img = self._render_multicanvas_core(
            img_src=img_src,
            descriptor=descriptor,
            rows=rows,
            cols=cols,
            preview_mode=self.state.preview_mode,
            dithering=self.state.dithering_config,
            palette=self.state.palette if self.state.preview_mode == "ark_simulation" else None,
        )

        if img is None:
            return None

        with self._mc_cache_lock:
            self._mc_cache_misses += 1
            self._mc_preview_cache[key] = img
            self._mc_preview_cache.move_to_end(key)
            while len(self._mc_preview_cache) > self._mc_cache_max_items:
                self._mc_preview_cache.popitem(last=False)

        return img

    # Backwards-compatible alias (kept because GUI may still call it)
    def render_preview_multicanvas(self) -> Optional[Image.Image]:
        return self.render_preview_multicanvas_cached()

    def _render_multicanvas_core(
        self,
        *,
        img_src: Image.Image,
        descriptor: dict,
        rows: int,
        cols: int,
        preview_mode: str,
        dithering: dict,
        palette: Optional[dict],
    ) -> Optional[Image.Image]:
        """
        Core multi-canvas preview renderer (stateless).
        """
        from PreviewRender_v1 import render_preview

        identity = descriptor.get("identity", {})
        if identity.get("type") != "multi_canvas":
            return None

        base_template_id = identity["base_template"]
        base_descriptor = self.template_loader.load(base_template_id)
        raster = base_descriptor["layout"]["raster"]
        if raster is None:
            return None

        tile_w = int(raster["width"])
        tile_h = int(raster["height"])

        preview_w = tile_w * int(cols)
        preview_h = tile_h * int(rows)

        if preview_mode == "ark_simulation":
            img = render_preview(
                img_src,
                template_id=descriptor,
                target_width=preview_w,
                target_height=preview_h,
                border=None,
                dithering=dithering,
                palette=palette,
                preview_mode="ark_simulation",
                overlay_def=None,
            )
        else:
            img = img_src.convert("RGBA")
            if img.size != (preview_w, preview_h):
                img = img.resize((preview_w, preview_h), Image.BILINEAR)

        img_np = np.array(img, dtype=np.uint8)
        grid_color = np.array([255, 255, 255, 120], dtype=np.uint8)

        for c in range(1, int(cols)):
            x = c * tile_w
            img_np[:, x - 1:x + 1, :] = grid_color

        for r in range(1, int(rows)):
            y = r * tile_h
            img_np[y - 1:y + 1, :, :] = grid_color

        return Image.fromarray(img_np, mode="RGBA")

    # ==================================================
    # Async preview: snapshot + render from snapshot
    # ==================================================

    def build_preview_snapshot(self) -> dict:
        """
        Captures an immutable snapshot for rendering preview in a background thread.
        Snapshot contains only pure data objects and callables that are safe to use
        (quantize_fn closure has its own local cache).
        """
        img = self.state.image_original
        descriptor = self.state.preview_descriptor

        if img is None or descriptor is None:
            return {"kind": "none"}

        preview_mode = self.state.preview_mode
        dithering = dict(self.state.dithering_config or {})
        show_obj = bool(self.state.show_game_object)
        game_object_type = self.state.game_object_type

        # Build a thread-safe palette object (quantize_fn + pack + byte->rgb)
        palette = None
        if preview_mode == "ark_simulation":
            palette = self._build_palette_object()

        identity = descriptor.get("identity", {})

        # Multi-canvas
        if identity.get("type") == "multi_canvas":
            multi = descriptor.get("multi_canvas")
            if not multi:
                return {"kind": "none"}

            rows = int(multi["rows"]["default"])
            cols = int(multi["cols"]["default"])

            req = self.state.canvas_request
            if req and req.get("mode") == "multi_canvas":
                rows = int(req.get("rows", rows))
                cols = int(req.get("cols", cols))

            base_template_id = identity["base_template"]
            base_descriptor = self.template_loader.load(base_template_id)
            raster = base_descriptor["layout"]["raster"]
            if raster is None:
                return {"kind": "none"}

            tile_w = int(raster["width"])
            tile_h = int(raster["height"])

            # Cap multi-canvas preview resolution (entire grid) to keep UI responsive.
            preview_w_full = tile_w * cols
            preview_h_full = tile_h * rows
            max_dim = int(getattr(self, "preview_max_dim", 512))
            m = max(preview_w_full, preview_h_full)
            if m > max_dim and m > 0:
                s = max_dim / float(m)
            else:
                s = 1.0
            preview_w = max(1, int(round(preview_w_full * s)))
            preview_h = max(1, int(round(preview_h_full * s)))
            tile_draw_w = max(1, int(round(tile_w * s)))
            tile_draw_h = max(1, int(round(tile_h * s)))

            template_id = self.state.selected_template_id or identity.get("id") or "multi_canvas"
            cache_key = self._mc_cache_key(
                template_id=template_id,
                rows=rows,
                cols=cols,
                preview_max_dim=int(getattr(self, "preview_max_dim", 512)),
                preview_mode=preview_mode,
                dithering=dithering,
                palette=palette,
                image_rev=self._image_rev,
            )

            return {
                "kind": "multi_canvas",
                "image_rev": self._image_rev,
                "img": img,
                "descriptor": copy.deepcopy(descriptor),
                "rows": rows,
                "cols": cols,
                "tile_w": tile_w,
                "tile_h": tile_h,
                "preview_w": preview_w,
                "preview_h": preview_h,
                "tile_draw_w": tile_draw_w,
                "tile_draw_h": tile_draw_h,
                "preview_mode": preview_mode,
                "dithering": dithering,
                "palette": palette,
                "cache_key": cache_key,
            }

        # Single canvas (requires physical raster)
        if not self.state.preview_ready or self.state.canvas_resolved is None:
            return {"kind": "none"}

        # IMPORTANT:
        # For fixed templates (e.g., flags) the preview must respect paint_area offsets.
        # The async preview path therefore uses the same prepared RGBA as generation.
        descriptor_or_template = descriptor or self.state.template
        eff_writer = self.get_effective_writer_mode(descriptor_or_template)

        if self._prepared_image_rgba is None:
            img_np = self._prepare_base_image_rgba(eff_writer=eff_writer)
            self._prepared_image_rgba = img_np.copy()
        else:
            img_np = self._prepared_image_rgba

        prepared_img = Image.fromarray(np.array(img_np, copy=True), mode="RGBA")

        # Target size for preview is based on the prepared raster size (may be square padded).
        target_w, target_h = prepared_img.size

        # Cap preview resolution to keep UI responsive for very large canvases.
        max_dim = int(getattr(self, "preview_max_dim", 512))
        m = max(int(target_w), int(target_h))
        if m > max_dim and m > 0:
            s = max_dim / float(m)
            target_w = max(1, int(round(target_w * s)))
            target_h = max(1, int(round(target_h * s)))

        overlay_def = None
        preview = descriptor.get("preview", {})
        if preview.get("mode") == "mask" and show_obj:
            overlay_dir = preview["overlay_dir"]
            base_name = preview["base_name"]
            overlay_def = {
                "image": (self.template_assets_root / overlay_dir / f"{base_name}.png"),
                "mask": (self.template_assets_root / overlay_dir / f"{base_name}_mask.png"),
            }

        return {
            "kind": "single",
            "image_rev": self._image_rev,
            "img": prepared_img,
            "descriptor": copy.deepcopy(descriptor),
            "target_w": target_w,
            "target_h": target_h,
            # Border is already applied inside _prepare_base_image_rgba() within paint_area.
            "border": None,
            "dithering": dithering,
            "palette": palette if preview_mode == "ark_simulation" else None,
            "preview_mode": preview_mode,
            "game_object_type": game_object_type,
            "overlay_def": overlay_def,
        }

    def render_preview_from_snapshot(self, snapshot: dict) -> Optional[Image.Image]:
        kind = snapshot.get("kind")
        if kind == "none":
            return None
        if kind == "multi_canvas":
            return self._render_multicanvas_from_snapshot(snapshot)

        from PreviewRender_v1 import render_preview

        return render_preview(
            snapshot["img"],
            template_id=None,
            target_width=int(snapshot["target_w"]),
            target_height=int(snapshot["target_h"]),
            border=snapshot["border"],
            dithering=snapshot["dithering"],
            palette=snapshot.get("palette"),
            preview_mode=snapshot["preview_mode"],
            game_object_type=snapshot.get("game_object_type"),
            overlay_def=snapshot.get("overlay_def"),
        )

    def _render_multicanvas_from_snapshot(self, snap: dict) -> Optional[Image.Image]:
        key = snap.get("cache_key")

        if key is not None:
            with self._mc_cache_lock:
                hit = self._mc_preview_cache.get(key)
                if hit is not None:
                    self._mc_cache_hits += 1
                    self._mc_preview_cache.move_to_end(key)
                    return hit

        from PreviewRender_v1 import render_preview

        rows = int(snap["rows"])
        cols = int(snap["cols"])
        tile_w = int(snap.get("tile_draw_w", snap["tile_w"]))
        tile_h = int(snap.get("tile_draw_h", snap["tile_h"]))

        preview_w = int(snap.get("preview_w", tile_w * cols))
        preview_h = int(snap.get("preview_h", tile_h * rows))

        if snap["preview_mode"] == "ark_simulation":
            img = render_preview(
                snap["img"],
                template_id=snap["descriptor"],
                target_width=preview_w,
                target_height=preview_h,
                border=None,
                dithering=snap["dithering"],
                palette=snap.get("palette"),
                preview_mode="ark_simulation",
                overlay_def=None,
            )
        else:
            img = snap["img"].convert("RGBA")
            if img.size != (preview_w, preview_h):
                img = img.resize((preview_w, preview_h), Image.BILINEAR)

        img_np = np.array(img, dtype=np.uint8)
        grid_color = np.array([255, 255, 255, 120], dtype=np.uint8)

        for c in range(1, cols):
            x = c * tile_w
            img_np[:, x - 1:x + 1, :] = grid_color

        for r in range(1, rows):
            y = r * tile_h
            img_np[y - 1:y + 1, :, :] = grid_color

        out = Image.fromarray(img_np, mode="RGBA")

        if key is not None:
            with self._mc_cache_lock:
                self._mc_cache_misses += 1
                self._mc_preview_cache[key] = out
                self._mc_preview_cache.move_to_end(key)
                while len(self._mc_preview_cache) > self._mc_cache_max_items:
                    self._mc_preview_cache.popitem(last=False)

        return out

    # ==================================================
    # Palette quantize_fn builder (for preview)
    # ==================================================

    @staticmethod
    def _nearest_color(rgb: np.ndarray, palette: list[np.ndarray]) -> np.ndarray:
        best = None
        best_dist = None
        for p in palette:
            dr = rgb[0] - p[0]
            dg = rgb[1] - p[1]
            db = rgb[2] - p[2]
            d = dr * dr + dg * dg + db * db
            if best is None or d < best_dist:
                best = p
                best_dist = d
        return best if best is not None else rgb

    def _make_quantize_fn(self, palette_linear_rgb: list[np.ndarray]):
        if not palette_linear_rgb:
            return None

        cache: dict[tuple, np.ndarray] = {}

        def quantize_fn(rgb):
            key = (int(rgb[0] * 255) >> 2, int(rgb[1] * 255) >> 2, int(rgb[2] * 255) >> 2)
            hit = cache.get(key)
            if hit is not None:
                return hit

            rgb_arr = np.asarray(rgb, dtype=np.float32)
            nearest = self._nearest_color(rgb_arr, palette_linear_rgb)
            cache[key] = nearest
            return nearest

        return quantize_fn

    def _build_palette_quantize_fn(self):
        translator = self._ark_translator
        if translator is None:
            return None

        if self.state.enabled_dyes is None:
            dyes = translator.dyes
        else:
            enabled = self.state.enabled_dyes
            dyes = [d for d in translator.dyes if d.observed_byte in enabled]

        if not dyes:
            return None

        palette = [np.asarray(d.linear_rgb, dtype=np.float32) for d in dyes]
        return self._make_quantize_fn(palette)

    def _build_palette_object(self):
        """Build a palette dict for bytes-first preview/encode.

        Includes:
        - quantize_fn (compat)
        - pack (numpy arrays)
        - byte_to_rgb_u8 (256x3)
        """
        translator = self._ark_translator
        if translator is None:
            return None

        qfn = self._build_palette_quantize_fn()
        if qfn is None:
            return None

        enabled = self.state.enabled_dyes
        pack = translator.palette_pack(enabled_dyes=enabled)
        b2rgb = translator.byte_to_rgb_u8(enabled_dyes=enabled)

        return {
            'mode': 'palette',
            'quantize_fn': qfn,
            'translator': translator,
            'pack': pack,
            'byte_to_rgb_u8': b2rgb,
            'enabled_dyes': enabled,
            'alpha_threshold': 10,
        }

    # ==================================================
    # Best dyes (fast, size-independent)
    # ==================================================

    @staticmethod
    def _srgb01_to_linear01(arr: np.ndarray) -> np.ndarray:
        """
        arr: float32 in [0,1]
        returns: float32 linear in [0,1]
        """
        return np.where(arr <= 0.04045, arr / 12.92, ((arr + 0.055) / 1.055) ** 2.4)

    def _compute_best_dyes_ranking_cached(
        self,
        *,
        sample_side: int = 256,
        max_pixels: int = 65536,
    ) -> list[int]:
        translator = self._ark_translator
        img0 = self.state.image_original
        if translator is None or img0 is None:
            return []

        with self._best_dyes_lock:
            if self._best_dyes_rank_rev == self._image_rev and self._best_dyes_rank is not None:
                return list(self._best_dyes_rank)

        img = img0.convert("RGB")
        w, h = img.size
        if max(w, h) > sample_side:
            scale = sample_side / float(max(w, h))
            nw = max(1, int(w * scale))
            nh = max(1, int(h * scale))
            img = img.resize((nw, nh), Image.BILINEAR)

        # IMPORTANT:
        # TablaDyes_v1.json 'linear_rgb' values in this build are historically
        # treated in the same space as image_rgb/255 (see ErrorDiffusion_v1 notes).
        # The earlier patch converted the image to true linear here, which skewed
        # the "best dyes" ranking and could lead to obviously wrong palettes.
        rgb = np.asarray(img, dtype=np.float32) / 255.0
        pixels = rgb.reshape(-1, 3)

        n = pixels.shape[0]
        if n > max_pixels:
            stride = max(1, n // max_pixels)
            pixels = pixels[::stride]

        palette = np.array([d.linear_rgb for d in translator.dyes], dtype=np.float32)  # Kx3
        bytes_arr = np.array([d.observed_byte for d in translator.dyes], dtype=np.int32)

        K = palette.shape[0]
        counts = np.zeros(K, dtype=np.int64)

        chunk = 8192
        for i in range(0, pixels.shape[0], chunk):
            p = pixels[i : i + chunk]
            d2 = np.sum((p[:, None, :] - palette[None, :, :]) ** 2, axis=2)  # chunk x K
            idx = np.argmin(d2, axis=1)
            counts += np.bincount(idx, minlength=K)

        order = np.argsort(-counts)
        ranking = bytes_arr[order].tolist()

        with self._best_dyes_lock:
            self._best_dyes_rank_rev = self._image_rev
            self._best_dyes_rank = ranking

        return ranking

    def calculate_best_dyes(
        self,
        X: int,
        *,
        sample_side: int = 256,
        max_pixels: int = 65536,
    ) -> list[int]:
        """
        Returns the selected dye bytes (top-X) and updates enabled_dyes accordingly.
        Designed to be fast and independent of the original image size.
        """
        ranking = self._compute_best_dyes_ranking_cached(sample_side=sample_side, max_pixels=max_pixels)
        if not ranking:
            return []

        X = max(1, int(X))
        selected = ranking[:X]
        self.set_enabled_dyes(set(selected))
        return selected

    # ==================================================
    # Generation
    # ==================================================

    def can_generate(self) -> bool:
        if self.state.image_original is None:
            return False

        descriptor = self.state.preview_descriptor
        if descriptor and descriptor.get("identity", {}).get("type") == "multi_canvas":
            return True

        if self.state.canvas_resolved is None:
            return False

        eff_writer = self.get_effective_writer_mode(descriptor or self.state.template)

        if self.state.canvas_is_dynamic:
            if self.state.canvas_request is None:
                return False
            if eff_writer in ("legacy_copy", "preserve_source"):
                return self.state.template is not None
            return True

        # Fixed single
        if eff_writer in ("legacy_copy", "preserve_source"):
            return self.state.template is not None
        return True

    # --------------------------------------------------
    # Visibility mask (encode)
    # --------------------------------------------------

    def _build_encode_visibility_mask(self, *, template: dict | None, descriptor: dict | None, width: int, height: int):
        """Builds a boolean mask (H,W) that limits encoding to the paintable area.

        Sources (deterministic, no fuzzy):
        - Canonical mask: Templates/<overlay_dir>/<base_name>_mask.png when preview.mode == 'mask' and base_name exists.
        - User mask: Templates/UserMasks/<Blueprint>_mask_user_refined.png (preferred) or <Blueprint>_mask_user.png

        If both canonical and user mask exist, returns (canonical AND user).
        """
        try:
            src = template or descriptor
            if not isinstance(src, dict):
                return None

            preview = src.get("preview") or {}
            if not isinstance(preview, dict):
                return None

            if (preview.get("mode") or "").strip().lower() != "mask":
                return None

            # Blueprint id (prefer selected_template_id -> identity.id)
            bp = (self.state.selected_template_id or (src.get("identity") or {}).get("id") or (src.get("identity") or {}).get("label") or "").strip()

            def _load_mask_bool(mask_path: Path) -> np.ndarray | None:
                if not mask_path.exists():
                    return None
                try:
                    img = Image.open(mask_path).convert("RGBA")
                    if img.size != (width, height):
                        img = img.resize((width, height), Image.NEAREST)
                    arr = np.array(img, dtype=np.uint8)
                    return arr[..., 3] > 0
                except Exception:
                    return None

            # User masks (portable)
            user_bool = None
            if bp:
                um_dir = self.template_assets_root / "UserMasks"
                p_ref = um_dir / f"{bp}_mask_user_refined.png"
                p_base = um_dir / f"{bp}_mask_user.png"
                user_bool = _load_mask_bool(p_ref) or _load_mask_bool(p_base)

            # Canonical mask (optional)
            can_bool = None
            overlay_dir = (preview.get("overlay_dir") or "").strip()
            base_name = (preview.get("base_name") or "").strip()
            if overlay_dir and base_name:
                can_path = self.template_assets_root / overlay_dir / f"{base_name}_mask.png"
                can_bool = _load_mask_bool(can_path)

            # Prefer user mask when present (portable override). Canonical mask is fallback only.
            if user_bool is not None:
                return user_bool
            if can_bool is not None:
                return can_bool
            return None
        except Exception:
            return None

            preview = src.get("preview") or {}
            if not isinstance(preview, dict):
                return None

            if (preview.get("mode") or "").strip().lower() != "mask":
                return None

            overlay_dir = (preview.get("overlay_dir") or "").strip()
            base_name = (preview.get("base_name") or "").strip()
            if not overlay_dir or not base_name:
                return None

            mask_path = self.template_assets_root / overlay_dir / f"{base_name}_mask.png"
            if not mask_path.exists():
                return None

            img = Image.open(mask_path).convert("RGBA")
            if img.size != (width, height):
                img = img.resize((width, height), Image.NEAREST)

            arr = np.array(img, dtype=np.uint8)
            alpha = arr[..., 3]
            return alpha > 0
        except Exception:
            return None

    def build_generation_request(self, *, output_path: Path) -> GenerationRequest:
        if not self.can_generate():
            raise RuntimeError("Estado no válido para generar .pnt")

        template = self.state.template
        canvas = self.state.canvas_resolved
        descriptor = self.state.preview_descriptor
        if canvas is None:
            raise RuntimeError("build_generation_request requiere canvas_resolved")

        eff_writer = self.get_effective_writer_mode(descriptor or template)

        # --------------------------------------------------
        # Template base (legacy_copy / preserve_source)
        # --------------------------------------------------
        base_pnt_path = None
        if eff_writer in ("legacy_copy", "preserve_source"):
            if template is None and descriptor is None:
                raise RuntimeError(f"{eff_writer} requiere base_pnt_path")

            src = template or descriptor
            r = (src.get("resolved") or {}) if isinstance(src, dict) else {}
            # External mode can supply a direct base path.
            if r.get("base_pnt_path"):
                base_pnt_path = Path(r["base_pnt_path"])
            else:
                # Physical template stored under Templates assets.
                if r.get("asset_dir") and r.get("pnt"):
                    base_pnt_path = self.template_assets_root / r["asset_dir"] / r["pnt"]

            if base_pnt_path is None:
                raise RuntimeError(f"{eff_writer} requiere base_pnt_path")
            if not base_pnt_path.exists():
                raise RuntimeError(f"No existe el .pnt base: {base_pnt_path}")

        # --------------------------------------------------
        # Imagen preparada
        # --------------------------------------------------
        if self._prepared_image_rgba is None:
            img_np = self._prepare_base_image_rgba(eff_writer=eff_writer)
            self._prepared_image_rgba = img_np.copy()
        else:
            img_np = self._prepared_image_rgba.copy()

        req = self.state.canvas_request

        # --------------------------------------------------
        # paint_area / planks
        # --------------------------------------------------
        if self.state.canvas_is_dynamic:
            if req is None:
                raise RuntimeError("canvas_request requerido para canvas dinámico")
            encode_paint_area = req["paint_area"]
        else:
            pa = self._get_fixed_paint_area(template=template, descriptor=descriptor)
            encode_paint_area = pa  # None means full raster

        planks = self._planks_override if self._planks_override is not None else canvas.get("planks")

        # template_id usable para naming / debugging
        template_id = None
        if template is not None:
            template_id = (template.get("identity") or {}).get("id")
        elif descriptor is not None:
            template_id = (descriptor.get("identity") or {}).get("id")

        encode_visibility_mask = self._build_encode_visibility_mask(template=template, descriptor=descriptor, width=int(canvas["width"]), height=int(canvas["height"]))

        return GenerationRequest(
            image_rgba=img_np,
            image_is_final=True,
            base_pnt_path=base_pnt_path,
            width=int(canvas["width"]),
            height=int(canvas["height"]),
            dithering=self.state.dithering_config,
            border=self.state.border_config,
            alpha_threshold=10,
            output_path=output_path,
            template_id=template_id,
            writer_mode=eff_writer,
            encode_paint_area=encode_paint_area,
            planks=planks,
            encode_visible_rows=None,
            enabled_dyes=self.state.enabled_dyes,
            encode_visibility_mask=encode_visibility_mask,
        )
    def build_generation_requests_multi(self, *, output_path: Path) -> list[GenerationRequest]:
        """
        Build a list of GenerationRequest for a multi-canvas descriptor.
        IMPORTANT: This function has NO side-effects (does not mutate controller state).
        """
        descriptor = self.state.preview_descriptor
        if descriptor is None:
            raise RuntimeError("No hay descriptor seleccionado")

        identity = descriptor.get("identity", {})
        if identity.get("type") != "multi_canvas":
            raise RuntimeError("build_generation_requests_multi llamado sin multi-canvas")

        multi = descriptor.get("multi_canvas")
        if not multi:
            raise RuntimeError("Descriptor multi_canvas inválido")

        cols = int(multi["cols"]["default"])
        rows = int(multi["rows"]["default"])

        req = self.state.canvas_request or {}
        cols = int(req.get("cols", cols))
        rows = int(req.get("rows", rows))

        if not (int(multi["cols"]["min"]) <= cols <= int(multi["cols"]["max"])):
            raise RuntimeError(f"cols fuera de rango: {cols}")
        if not (int(multi["rows"]["min"]) <= rows <= int(multi["rows"]["max"])):
            raise RuntimeError(f"rows fuera de rango: {rows}")

        # Resolve base physical template locally (NO state mutation)
        base_template_id = identity["base_template"]
        base_template = self.template_loader.resolve(base_template_id, {})
        raster = base_template["layout"]["raster"]
        if raster is None:
            raise RuntimeError("El template base del multi-canvas no tiene raster físico")

        canvas_w = int(raster["width"])
        canvas_h = int(raster["height"])

        base_pnt_path = self.template_assets_root / base_template["resolved"]["asset_dir"] / base_template["resolved"]["pnt"]
        if not base_pnt_path.exists():
            raise RuntimeError(f"No existe el .pnt base: {base_pnt_path}")

        if self.state.image_original is None:
            raise RuntimeError("No hay imagen cargada para multi-canvas")

        full_img = np.array(self.state.image_original.convert("RGBA"), dtype=np.uint8)

        expected_w = canvas_w * cols
        expected_h = canvas_h * rows

        full_h, full_w, _ = full_img.shape
        if full_w != expected_w or full_h != expected_h:
            full_img = np.array(
                Image.fromarray(full_img, mode="RGBA").resize((expected_w, expected_h), Image.BILINEAR),
                dtype=np.uint8,
            )

        # Output directory handling
        out_dir = output_path
        if out_dir.suffix.lower() == ".pnt":
            out_dir = out_dir.with_suffix("")
        out_dir.mkdir(parents=True, exist_ok=True)

        pattern = descriptor.get("naming", {}).get("pattern", "tile_r{row}_c{col}.pnt")

        requests: list[GenerationRequest] = []
        encode_visibility_mask = self._build_encode_visibility_mask(
            template=base_template,
            descriptor=descriptor,
            width=canvas_w,
            height=canvas_h,
        )
        for row in range(rows):
            for col in range(cols):
                y0 = row * canvas_h
                x0 = col * canvas_w
                sub_img = full_img[y0 : y0 + canvas_h, x0 : x0 + canvas_w, :].copy()

                filename = pattern.format(row=row, col=col)
                if not filename.lower().endswith(".pnt"):
                    filename += ".pnt"

                requests.append(
                    GenerationRequest(
                        image_rgba=sub_img,
                        image_is_final=True,
                        base_pnt_path=base_pnt_path,
                        width=canvas_w,
                        height=canvas_h,
                        dithering=self.state.dithering_config,
                        border=self.state.border_config,
                        alpha_threshold=10,
                        encode_paint_area=None,
                        planks=None,
                        encode_visible_rows=None,
                        enabled_dyes=self.state.enabled_dyes,
                        encode_visibility_mask=encode_visibility_mask,
                        output_path=out_dir / filename,
                    )
                )

        return requests

    def request_generation(self, *, output_path: Path, tabla_dyes_path: Path) -> None:
        """
        Entry point used by GUI.
        Routes to single-canvas or multi-canvas generation.
        """
        descriptor = self.state.preview_descriptor
        tpl_type = descriptor.get("identity", {}).get("type") if descriptor else None

        if tpl_type == "multi_canvas":
            self.requests_generation(output_path=output_path, tabla_dyes_path=tabla_dyes_path)
            return

        req = self.build_generation_request(output_path=output_path)
        GenerationService.run(req, tabla_dyes_path=tabla_dyes_path)
        self._last_generated_path = output_path

    def requests_generation(self, *, output_path: Path, tabla_dyes_path: Path) -> None:
        """
        Multi-canvas generation entry: builds N requests and executes them.
        """
        descriptor = self.state.preview_descriptor
        identity = descriptor.get("identity", {}) if descriptor else {}

        if identity.get("type") != "multi_canvas":
            raise RuntimeError("requests_generation llamado sin multi-canvas")

        requests = self.build_generation_requests_multi(output_path=output_path)
        for req in requests:
            GenerationService.run(req, tabla_dyes_path=tabla_dyes_path)

        # For multi-canvas, output is a directory
        out_dir = output_path.with_suffix("") if output_path.suffix.lower() == ".pnt" else output_path
        self._last_generated_path = out_dir

    # ==================================================
    # Image preparation (generation)
    # ==================================================

    def _prepare_base_image_rgba(self, *, eff_writer: str = 'legacy_copy') -> np.ndarray:
        """
        Prepares RGBA image for generation:
        - Resizes original image to the visible area.
        - Inserts into a square raster (template size).
        - Applies border only on visible area.
        """
        if self.state.canvas_resolved is None or self.state.image_original is None:
            raise RuntimeError("_prepare_base_image_rgba requiere canvas_resolved e image_original")

        eff_writer = (eff_writer or 'legacy_copy').strip().lower()
        if eff_writer not in ('legacy_copy', 'raster20', 'preserve_source'):
            eff_writer = 'legacy_copy'

        # -----------------------------------------------
        # Resolver dimensiones visibles y del raster
        # -----------------------------------------------
        if self.state.canvas_is_dynamic:
            if self.state.canvas_request is None:
                raise RuntimeError("canvas_request requerido para canvas dinámico")

            pa = self.state.canvas_request["paint_area"]
            visible_w = int(pa["width"])
            visible_h = int(pa["height"])
            off_x = int(pa.get('offset_x', 0))
            off_y = int(pa.get('offset_y', 0))

            raster_w = int(self.state.canvas_resolved["width"])
            raster_h = int(self.state.canvas_resolved["height"])
        else:
            # Fijos: raster del template + paint_area perfil (project/apc)
            raster_w = int(self.state.canvas_resolved["width"])
            raster_h = int(self.state.canvas_resolved["height"])

            pa = self._get_fixed_paint_area(template=self.state.template, descriptor=self.state.preview_descriptor)
            if pa is None:
                visible_w = raster_w
                visible_h = raster_h
                off_x = 0
                off_y = 0
            else:
                visible_w = int(pa["width"])
                visible_h = int(pa["height"])
                off_x = int(pa.get("offset_x", 0))
                off_y = int(pa.get("offset_y", 0))

        # -----------------------------------------------
        # Lienzo destino
        # - legacy_copy: mantiene padding cuadrado (compat)
        # - raster20: tamaño exacto (permite rectángulos)
        # -----------------------------------------------
        if eff_writer == 'legacy_copy':
            template_size = max(raster_w, raster_h)
            canvas = np.zeros((template_size, template_size, 4), dtype=np.uint8)
        else:
            canvas = np.zeros((raster_h, raster_w, 4), dtype=np.uint8)

        # -----------------------------------------------
        # Insertar imagen reescalada en el área visible
        # -----------------------------------------------
        img = self.state.image_original.convert("RGBA").resize((visible_w, visible_h), Image.BILINEAR)
        img_np = np.array(img, dtype=np.uint8)

        y1 = off_y + visible_h
        x1 = off_x + visible_w
        canvas[off_y:y1, off_x:x1] = img_np

        # -----------------------------------------------
        # Border SOLO sobre el área visible
        # -----------------------------------------------
        border = self.state.border_config or {}
        style = border.get("style", "none")
        size = int(border.get("size", 0))

        if style != "none" and size > 0:
            canvas_visible = canvas[off_y:y1, off_x:x1]

            if style == "image":
                frame_img = border.get("frame_image")
                if frame_img is not None:
                    canvas_visible = apply_frame_border(
                        canvas_visible,
                        border_size=size,
                        style="image",
                        frame_image=np.array(frame_img.convert("RGBA"), dtype=np.uint8),
                    )
            else:
                canvas_visible = apply_frame_border(canvas_visible, border_size=size, style=style)

            canvas[off_y:y1, off_x:x1] = canvas_visible

        return canvas
