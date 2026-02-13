from __future__ import annotations

import copy
import os
import json
from pathlib import Path
from typing import Optional


# ==================================================
# Normalización schema 1.1
# ==================================================

SCHEMA_1_1 = {
    "schema_version": "1.1",

    "identity": {
        "id": None,
        "label": None,
        "category": None,
        "type": None,
        "base_template": None,
    },

    "layout": {
        "raster": None,
        "dynamic": None,
    },

    "dynamic": None,

    "encode": {
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
        "supports_border": False,
        "supports_dithering": False,
        "supports_dynamic_resize": False,
    },

    "multi_canvas": None,

    "naming": {
        "pattern": None,
    },

    "resolved": {
        "asset_dir": None,
    },

    "meta": {},
}


def deep_merge(target: dict, source: dict):
    """Rellena target con claves faltantes de source sin sobrescribir valores existentes."""
    for key, value in source.items():
        if key not in target:
            target[key] = copy.deepcopy(value)
        else:
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                deep_merge(target[key], value)
    return target


def normalize_descriptor(data: dict) -> dict:
    data = copy.deepcopy(data)

    # Forzar schema_version
    data["schema_version"] = "1.1"

    # Estructura base
    deep_merge(data, SCHEMA_1_1)

    # Sub-bloques obligatorios
    deep_merge(data["layout"], SCHEMA_1_1["layout"])
    deep_merge(data["encode"], SCHEMA_1_1["encode"])
    deep_merge(data["preview"], SCHEMA_1_1["preview"])
    deep_merge(data["constraints"], SCHEMA_1_1["constraints"])
    deep_merge(data["capabilities"], SCHEMA_1_1["capabilities"])
    deep_merge(data["naming"], SCHEMA_1_1["naming"])
    deep_merge(data["resolved"], SCHEMA_1_1["resolved"])

    return data


# ==================================================
# Loader principal
# ==================================================


class TemplateDescriptorLoader:
    """Carga y resuelve descriptores .template.json.

    Mejoras respecto al loader original:
    - Índice de paths (evita rglob por cada load)
    - Templates virtuales (user_virtual_templates.json) para añadir nuevos dinos/humans
    - Escaneo acotado a TemplateDescriptors_* para evitar loops/junctions en Windows
    """

    VIRTUAL_CFG_NAME = "user_virtual_templates.json"

    # ArkPaintingConverter metadata (non-canonical, used only as optional reference)
    APC_CFG_NAME = "ArkPaintingConverter_PntPaintingConfiguration.json"

    # Debug-only: load/attach APC metadata if PC_ENABLE_APC_DEBUG=1
    PC_ENABLE_APC_DEBUG = os.environ.get("PC_ENABLE_APC_DEBUG", "0") == "1"

    def __init__(self, templates_root: Path):
        self.templates_root = Path(templates_root)

        # Cache de descriptores normalizados
        self._cache: dict[str, dict] = {}

        # Index: template_id -> file path
        self._index: dict[str, Path] = {}
        self._build_index()

        # ArkPaintingConverter metadata (optional reference)
        # template_id -> {"width","height","category","visible_area","paint_area"}
        self._apc_meta: dict[str, dict] = {}
        # APC meta is debug-only (env PC_ENABLE_APC_DEBUG=1)
        if self.PC_ENABLE_APC_DEBUG:
            self._load_apc_meta()

        # Virtual templates
        self._virtual_templates: dict[str, dict] = {}
        self._load_virtual_templates()

        # Explicit blueprint->overlay base mapping (no fuzzy)
        self._blueprint_to_base = self._load_blueprint_to_base_map()


    # --------------------------------------------------
    # ArkPaintingConverter meta
    # --------------------------------------------------

    def _apc_cfg_path(self) -> Path:
        return self.templates_root / self.APC_CFG_NAME

    def _load_apc_meta(self) -> None:
        """Loads ArkPaintingConverter_PntPaintingConfiguration.json if present.

        This data is NOT treated as canonical (the game is). We only surface it
        as optional meta for debugging/alternative visible areas.
        """
        self._apc_meta.clear()
        cfg_path = self._apc_cfg_path()
        if not cfg_path.exists():
            return

        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            return

        # Expected shape: {"PntPaintings": [...]} or legacy list
        entries = None
        if isinstance(cfg, dict):
            entries = cfg.get("PntPaintings") or cfg.get("paintings") or cfg.get("items")
        if entries is None and isinstance(cfg, list):
            entries = cfg
        if not isinstance(entries, list):
            return

        for e in entries:
            if not isinstance(e, dict):
                continue
            tid = e.get("Id") or e.get("id")
            if not isinstance(tid, str) or not tid.strip():
                continue
            tid = tid.strip()

            meta = {
                "width": e.get("Width"),
                "height": e.get("Height"),
                "category": e.get("Category"),
                "file_extension": e.get("FileExtension"),
            }

            va = e.get("VisibleArea")
            if isinstance(va, dict):
                sx = va.get("StartX")
                sy = va.get("StartY")
                ex = va.get("EndX")
                ey = va.get("EndY")
                if all(isinstance(v, int) for v in (sx, sy, ex, ey)):
                    # ArkPaintingConverter appears to use end-exclusive coordinates.
                    meta["visible_area"] = {"start_x": sx, "start_y": sy, "end_x": ex, "end_y": ey}
                    meta["paint_area"] = {
                        "offset_x": int(sx),
                        "offset_y": int(sy),
                        "width": int(ex) - int(sx),
                        "height": int(ey) - int(sy),
                    }

            self._apc_meta[tid] = meta

    def _attach_apc_meta(self, descriptor: dict, template_id: str) -> None:
        meta = self._apc_meta.get(template_id)
        if not meta:
            return
        # Preserve existing keys under descriptor["meta"]
        descriptor.setdefault("meta", {})
        descriptor["meta"].setdefault("apc", {})
        # Shallow merge (do not overwrite if already present)
        for k, v in meta.items():
            if k not in descriptor["meta"]["apc"]:
                descriptor["meta"]["apc"][k] = copy.deepcopy(v)

    # --------------------------------------------------
    # Index
    # --------------------------------------------------

    def _build_index(self) -> None:
        """Construye un índice de template_id -> Path.

        IMPORTANTE: se limita a subdirectorios TemplateDescriptors_* para evitar:
        - Escanear Assets enormes
        - Loops por junctions/symlinks
        """
        self._index.clear()

        # Si existe una estructura estándar, usarla.
        roots = [p for p in self.templates_root.glob("TemplateDescriptors_*") if p.is_dir()]

        # Fallback: si no existe, degradar a templates_root
        scan_roots = roots if roots else [self.templates_root]

        for r in scan_roots:
            for path in r.rglob("*.template.json"):
                # path.stem = "X.template" -> quitar sufijo ".template"
                template_id = path.stem.replace(".template", "")
                # Si hay colisiones, preferimos el primero encontrado (determinista por orden de scan_roots)
                self._index.setdefault(template_id, path)

    # --------------------------------------------------
    # Virtual templates
    # --------------------------------------------------

    def _virtual_cfg_path(self) -> Path:
        return self.templates_root / self.VIRTUAL_CFG_NAME

    def _load_virtual_templates(self) -> None:
        """Carga templates virtuales desde Templates/user_virtual_templates.json."""
        cfg_path = self._virtual_cfg_path()
        if not cfg_path.exists():
            # Create default config but continue (we also auto-load Templates/UserMasks below).
            cfg_path.write_text(json.dumps({"dinos": [], "humans": []}, indent=2), encoding="utf-8")
            cfg = {"dinos": [], "humans": []}
        else:
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            except Exception:
                cfg = {"dinos": [], "humans": []}

        self._virtual_templates.clear()

        dinos = cfg.get("dinos") or []
        humans = cfg.get("humans") or []

        for tid in dinos:
            if isinstance(tid, str) and tid.strip():
                self._virtual_templates[tid.strip()] = self._mk_virtual(tid.strip(), "Dinos", 256, 256)

        for tid in humans:
            if isinstance(tid, str) and tid.strip():
                self._virtual_templates[tid.strip()] = self._mk_virtual(tid.strip(), "Humans", 512, 512)

        # Auto virtual templates from Templates/UserMasks/*_mask_user.json (non-persistent)
        try:
            um_dir = self.templates_root / "UserMasks"
            if um_dir.exists():
                for p in um_dir.glob("*_mask_user.json"):
                    try:
                        meta = json.loads(p.read_text(encoding="utf-8"))
                    except Exception:
                        continue
                    bp = (meta.get("blueprint") or "").strip()
                    if not bp:
                        continue
                    # Resolution from meta, fallback by convention
                    res = meta.get("resolution") or []
                    w = int(res[0]) if isinstance(res, list) and len(res) == 2 and isinstance(res[0], int) else (256 if bp.endswith("_Character_BP_C") else 512)
                    h = int(res[1]) if isinstance(res, list) and len(res) == 2 and isinstance(res[1], int) else (256 if bp.endswith("_Character_BP_C") else 512)

                    asset_dir = "Dinos" if bp.endswith("_Character_BP_C") else ("Humans" if bp.endswith("_Human_BP_C") else "Canvas")
                    # NOTE: Virtual templates created from UserMasks have NO physical base .pnt.
                    # Do NOT set resolved.pnt here, otherwise preserve_source/legacy_copy could
                    # incorrectly assume a physical template exists and fail at generate time.
                    d = normalize_descriptor({
                        "identity": {
                            "id": bp,
                            "label": bp,
                            "category": "Canvas",
                            "type": "single",
                            "base_template": None,
                        },
                        "layout": {
                            "raster": {"width": int(w), "height": int(h), "paint_data_size": int(w) * int(h)},
                            "dynamic": None,
                        },
                        "encode": {"paint_area": "full_raster", "planks": None},
                        "preview": {"mode": "mask" if asset_dir == "Dinos" else None, "overlay_dir": "Dinos_Overlay" if asset_dir == "Dinos" else None, "base_name": None},
                        "constraints": {"max_colors": 127, "alpha_mode": "binary", "palette_required": True},
                        "capabilities": {"supports_border": True, "supports_dithering": True, "supports_dynamic_resize": False},
                        "naming": {"pattern": f"{bp}.pnt"},
                        "resolved": {"asset_dir": asset_dir, "virtual": True},
                        "meta": {
                            "notes": "Virtual template from Templates/UserMasks (no .template.json). Recomendado: writer raster20.",
                            "usermask": True,
                            "usermask_dir": "UserMasks",
                            "usermask_meta": str(p.name),
                        },
                    })
                    self._virtual_templates[bp] = d
        except Exception:
            pass


    def reload_virtual_templates(self) -> None:
        """Reload virtual templates (persistent + auto UserMasks)."""
        try:
            self._cache.clear()
        except Exception:
            pass
        self._load_virtual_templates()

    def register_virtual_template(self, *, template_id: str, kind: str) -> None:
        """Registra un template virtual persistente.

        kind: 'dino' | 'human'
        """
        tid = (template_id or "").strip()
        if not tid:
            raise ValueError("template_id vacío")

        kind = (kind or "").strip().lower()
        if kind not in ("dino", "human"):
            raise ValueError("kind inválido")

        cfg_path = self._virtual_cfg_path()
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            except Exception:
                cfg = {"dinos": [], "humans": []}
        else:
            cfg = {"dinos": [], "humans": []}

        key = "dinos" if kind == "dino" else "humans"
        arr = cfg.get(key) or []

        if tid not in arr:
            arr.append(tid)
        cfg[key] = sorted(set(arr))

        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        self._load_virtual_templates()

    def _mk_virtual(self, template_id: str, asset_dir: str, w: int, h: int) -> dict:
        return normalize_descriptor({
            "identity": {
                "id": template_id,
                "label": template_id,
                "category": "Canvas",
                "type": "single",
                "base_template": None,
            },
            "layout": {
                "raster": {
                    "width": int(w),
                    "height": int(h),
                    "paint_data_size": int(w) * int(h),
                },
                "dynamic": None,
            },
            "encode": {
                "paint_area": "full_raster",
                "planks": None,
            },
            "preview": {
                "mode": None,
                "overlay_dir": None,
                "base_name": template_id,
                "world_scale": None,
                "show_indices": None,
            },
            "constraints": {
                "max_colors": 127,
                "alpha_mode": "binary",
                "palette_required": True,
            },
            "capabilities": {
                "supports_border": True,
                "supports_dithering": True,
                "supports_dynamic_resize": False,
            },
            "naming": {
                "pattern": f"{template_id}.pnt",
            },
            "resolved": {
                "asset_dir": asset_dir,
                "virtual": True,
            },
            "meta": {
                "notes": "Virtual template (sin .pnt físico). Recomendado: writer raster20.",
            },
        })

    # --------------------------------------------------
    # Carga base (normalizada, NO resuelta)
    # --------------------------------------------------

    def load(self, template_id: str) -> dict:
        if template_id in self._cache:
            return copy.deepcopy(self._cache[template_id])

        if template_id in self._virtual_templates:
            d = copy.deepcopy(self._virtual_templates[template_id])
            self._attach_apc_meta(d, template_id)
            self._postprocess_descriptor(d, template_id)
            return d

        path = self._find_template_file(template_id)
        if not path:
            raise FileNotFoundError(f"Template no encontrado: {template_id}")

        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        descriptor = normalize_descriptor(raw)
        # Attach optional ArkPaintingConverter metadata (if present)
        self._attach_apc_meta(descriptor, template_id)
        self._postprocess_descriptor(descriptor, template_id)

        self._cache[template_id] = descriptor
        return copy.deepcopy(descriptor)


    # --------------------------------------------------
    # Postprocess: Dino overlays (preview + mask)
    # --------------------------------------------------

    def _postprocess_descriptor(self, descriptor: dict, template_id: str) -> None:
        """Attach derived preview overlay info for dinos.

        - Dinos get preview.mode='mask' so the UI can show silhouette+mask.
        - base_name is inferred from Templates/Dinos_Overlay assets (and CreatureIDs catalog if present).
        """
        try:
            asset_dir = (descriptor.get("resolved") or {}).get("asset_dir")
            if asset_dir != "Dinos":
                return
            preview = descriptor.get("preview") or {}
            if not isinstance(preview, dict):
                preview = {}
                descriptor["preview"] = preview

            # Ensure mode is 'mask' so PreviewController can build overlay_def
            if not preview.get("mode"):
                preview["mode"] = "mask"

            if not preview.get("overlay_dir"):
                preview["overlay_dir"] = "Dinos_Overlay"

            # Resolve base_name deterministically (mapping only). If current base_name is missing
            # or does not correspond to an existing mask asset, attempt mapping.
            base_name = (preview.get("base_name") or "").strip()
            overlay_dir = (preview.get("overlay_dir") or "Dinos_Overlay").strip()

            def _has_mask(bn: str) -> bool:
                try:
                    p = self.templates_root / overlay_dir / f"{bn}_mask.png"
                    return p.exists()
                except Exception:
                    return False

            if (not base_name) or (base_name and not _has_mask(base_name)):
                label = ((descriptor.get("identity") or {}).get("label") or "").strip()
                base = self._infer_dino_overlay_base(label)
                if base and _has_mask(base):
                    preview["base_name"] = base

        except Exception:
            # Never hard-fail load() due to overlays
            return


    def _load_blueprint_to_base_map(self) -> dict[str, str]:
        """Load explicit blueprint->overlay base mapping (no fuzzy matching)."""
        try:
            p = self.templates_root / "Dinos_Overlay" / "blueprint_to_base.json"
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    out = {}
                    for k, v in data.items():
                        if isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip():
                            out[k.strip()] = v.strip()
                    return out
        except Exception:
            pass
        return {}

    def _ensure_dino_overlay_cache(self) -> None:
        if hasattr(self, "_dino_overlay_bases") and hasattr(self, "_dino_asset_to_name"):
            return

        # Overlay bases from Templates/Dinos_Overlay/*.png (excluding *_mask*.png)
        bases = []
        try:
            ov_dir = self.templates_root / "Dinos_Overlay"
            if ov_dir.exists():
                for p in ov_dir.glob("*.png"):
                    name = p.stem
                    if name.endswith("_mask") or name.endswith("_mask_P") or name.endswith("_mask_S"):
                        continue
                    bases.append(name)
        except Exception:
            bases = []

        # Creature catalog (optional): blueprint_asset -> creature name
        asset_to_name = {}
        try:
            cat_path = self.templates_root / "CreatureIDs_Catalog_Normalized.json"
            if cat_path.exists():
                data = json.loads(cat_path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    for e in data:
                        if not isinstance(e, dict):
                            continue
                        a = e.get("blueprint_asset")
                        nm = e.get("name")
                        if isinstance(a, str) and isinstance(nm, str) and a and nm:
                            asset_to_name[a] = nm
        except Exception:
            asset_to_name = {}

        self._dino_overlay_bases = bases
        self._dino_asset_to_name = asset_to_name

    @staticmethod
    def _norm_key(s: str) -> str:
        import re
        return re.sub(r"[^A-Za-z0-9]", "", s or "").lower()

    @staticmethod
    def _levenshtein(a: str, b: str, limit: int = 12) -> int:
        # small DP with early stop; good enough for ~100 candidates
        if a == b:
            return 0
        la, lb = len(a), len(b)
        if abs(la - lb) > limit:
            return limit + 1
        prev = list(range(lb + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            minrow = cur[0]
            for j, cb in enumerate(b, 1):
                ins = cur[j - 1] + 1
                dele = prev[j] + 1
                sub = prev[j - 1] + (ca != cb)
                v = ins if ins < dele else dele
                v = sub if sub < v else v
                cur.append(v)
                if v < minrow:
                    minrow = v
            prev = cur
            if minrow > limit:
                return limit + 1
        return prev[-1]

    def _infer_dino_overlay_base(self, label: str) -> str | None:
        """Map blueprint id -> overlay base using explicit mapping only.

        Zero fuzzy matching: no substring, no levenshtein.
        """
        self._ensure_dino_overlay_cache()
        bases = set(getattr(self, "_dino_overlay_bases", []) or [])
        if not bases:
            return None

        label = (label or "").strip()
        if not label:
            return None

        # Prefer explicit mapping (keys are blueprint ids with trailing _C)
        mp = getattr(self, "_blueprint_to_base", {}) or {}
        base = mp.get(label)
        if base is None and label.endswith("_C"):
            base = mp.get(label[:-2])
        if base is None and (not label.endswith("_C")):
            base = mp.get(label + "_C")

        if isinstance(base, str) and base in bases:
            return base

        # No mapping -> no inference (strict)
        return None

    # --------------------------------------------------
    # Resolución con parámetros (dinámicos)
    # --------------------------------------------------

    def resolve(self, template_id: str, params: Optional[dict] = None) -> dict:
        base = self.load(template_id)
        resolved = copy.deepcopy(base)

        params = params or {}

        # Resolver layout
        resolved["layout"] = self._resolve_layout(
            base_layout=base["layout"],
            params=params,
        )

        # Resolver naming (si aplica)
        resolved = self._resolve_naming(resolved, params)

        return resolved

    def _resolve_layout(self, *, base_layout: dict, params: dict) -> dict:
        layout = copy.deepcopy(base_layout)
        raster = layout.get("raster")
        if raster is None:
            return layout

        size = params.get("size")
        if size is None:
            return layout

        # Dinámico por size (cuadrado)
        layout["raster"]["width"] = size
        layout["raster"]["height"] = size
        return layout

    def _resolve_naming(self, descriptor: dict, params: dict) -> dict:
        pattern = descriptor["naming"].get("pattern")
        if not pattern:
            return descriptor

        try:
            pnt_name = pattern.format(**params)
        except KeyError:
            return descriptor

        descriptor["resolved"]["pnt"] = pnt_name
        return descriptor

    # --------------------------------------------------
    # Localizar archivo .template.json
    # --------------------------------------------------

    def _find_template_file(self, template_id: str) -> Optional[Path]:
        if template_id in self._index:
            return self._index[template_id]

        # Fallback ultra-conservador (si hay cambios en estructura)
        filename = f"{template_id}.template.json"
        for path in self.templates_root.rglob("*.template.json"):
            if path.name == filename:
                self._index[template_id] = path
                return path

        return None

    def list_templates(self, *, include_abstract: bool = False, include_virtual: bool = True) -> list[str]:
        """Lista template_ids disponibles."""
        template_ids = list(self._index.keys())

        if include_virtual:
            template_ids.extend(self._virtual_templates.keys())

        # Eliminar duplicados
        template_ids = sorted(set(template_ids))

        if not include_abstract:
            filtered = []
            for tid in template_ids:
                tpl = self.load(tid)
                if not tpl.get("identity", {}).get("abstract", False):
                    filtered.append(tid)
            template_ids = filtered

        return sorted(template_ids)
