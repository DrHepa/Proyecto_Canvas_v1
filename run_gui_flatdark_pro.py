import tkinter as tk
from tkinter import ttk
from pathlib import Path
import math
import sys
import ctypes

from PIL import Image, ImageDraw, ImageTk

from PreviewGUI_v1 import PreviewGUI as BasePreviewGUI

import os
import json
import time

# ----------------------------
# settings.json (robust + validation)
# ----------------------------

_SETTINGS_DEFAULT = {
    "version": 1,
    "language": "auto",  # auto = do not override BasePreviewGUI/user_cache
    "user_translation_overrides": None,  # path to JSON overrides
    "defaults": {
        "preview_mode": None,  # visual | ark_simulation | null
        "show_game_object": None,
        "use_all_dyes": None,
        "best_colors": None,
        "border_style": None,  # none | image | null
        "dither_mode": None,  # none | palette_fs | palette_ordered | null
        "show_advanced": None,
    },
    "advanced_defaults": {
        "external_enabled": None,
        "external_recursive": None,
        "external_detect_guid": None,
        "external_max_files": None,
        "external_preserve_metadata": None,
        "show_dino_tools": None,
    },
    "window": {
        "remember_geometry": False,
        "geometry": None,  # "1100x800+100+80"
    },
}


def _deep_merge(a, b):
    if not isinstance(a, dict) or not isinstance(b, dict):
        return b
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out.get(k), dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _is_writable_dir(p: Path) -> bool:
    try:
        p.mkdir(parents=True, exist_ok=True)
        test = p / ".__wtest__"
        test.write_text("ok", encoding="utf-8")
        test.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def _settings_dir_candidates(base_dir: Path):
    # 1) beside script/exe
    yield base_dir

    # 2) user config
    if os.name == "nt":
        appdata = os.environ.get("APPDATA") or os.environ.get("LOCALAPPDATA")
        if appdata:
            yield Path(appdata) / "ProyectoCanvas"
    else:
        yield Path.home() / ".proyecto_canvas"


def _resolve_settings_path(base_dir: Path) -> Path:
    for d in _settings_dir_candidates(base_dir):
        if _is_writable_dir(d):
            return d / "settings.json"
    # last resort: current dir
    return base_dir / "settings.json"


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json_atomic(path: Path, data: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        os.replace(tmp, path)
    except Exception:
        # fallback if replace fails
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def _validate_enum(v, allowed, default=None):
    if v is None:
        return None
    if isinstance(v, str) and v in allowed:
        return v
    return default


def _validate_bool(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    return None


def _validate_int(v, lo=None, hi=None):
    if v is None:
        return None
    try:
        iv = int(v)
    except Exception:
        return None
    if lo is not None and iv < lo:
        return None
    if hi is not None and iv > hi:
        return None
    return iv


def _validate_geometry(s: str):
    if s is None:
        return None
    if not isinstance(s, str):
        return None
    # Basic Tk geometry: WxH+X+Y (X/Y can be negative)
    import re
    if re.match(r"^\d+x\d+\+[+-]?\d+\+[+-]?\d+$", s):
        return s
    return None


def _validate_settings(raw: dict) -> dict:
    s = _deep_merge(_SETTINGS_DEFAULT, raw if isinstance(raw, dict) else {})

    s["version"] = 1
    s["language"] = _validate_enum(s.get("language"), {"auto", "es", "en", "zh", "ru"}, default="auto")

    uto = s.get("user_translation_overrides")
    if uto is not None and not isinstance(uto, str):
        uto = None
    s["user_translation_overrides"] = uto

    d = s.get("defaults") or {}
    d["preview_mode"] = _validate_enum(d.get("preview_mode"), {"visual", "ark_simulation"})
    d["show_game_object"] = _validate_bool(d.get("show_game_object"))
    d["use_all_dyes"] = _validate_bool(d.get("use_all_dyes"))
    d["best_colors"] = _validate_int(d.get("best_colors"), 1, 255)
    d["border_style"] = _validate_enum(d.get("border_style"), {"none", "image"})
    d["dither_mode"] = _validate_enum(d.get("dither_mode"), {"none", "palette_fs", "palette_ordered"})
    d["show_advanced"] = _validate_bool(d.get("show_advanced"))
    s["defaults"] = d

    a = s.get("advanced_defaults") or {}
    a["external_enabled"] = _validate_bool(a.get("external_enabled"))
    a["external_recursive"] = _validate_bool(a.get("external_recursive"))
    a["external_detect_guid"] = _validate_bool(a.get("external_detect_guid"))
    a["external_max_files"] = _validate_int(a.get("external_max_files"), 50, 20000)
    a["external_preserve_metadata"] = _validate_bool(a.get("external_preserve_metadata"))
    a["show_dino_tools"] = _validate_bool(a.get("show_dino_tools"))
    s["advanced_defaults"] = a

    w = s.get("window") or {}
    rg = w.get("remember_geometry")
    w["remember_geometry"] = bool(rg) if isinstance(rg, bool) else False
    w["geometry"] = _validate_geometry(w.get("geometry"))
    s["window"] = w

    return s


def _load_settings_bundle(base_dir: Path):
    path = _resolve_settings_path(base_dir)
    raw = _read_json(path)

    if raw is None:
        # if file exists but unreadable, backup
        if path.exists():
            try:
                stamp = time.strftime("%Y%m%d_%H%M%S")
                bad = path.with_name(f"settings.bad_{stamp}.json")
                path.replace(bad)
            except Exception:
                pass
        s = _validate_settings({})
        try:
            _write_json_atomic(path, s)
        except Exception:
            pass
        return s, path, path.parent

    s = _validate_settings(raw)
    # if file had invalid entries, normalize in-place (best effort)
    try:
        _write_json_atomic(path, s)
    except Exception:
        pass

    return s, path, path.parent


OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path("assets_flatdark_pro")


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


def _ensure_assets():
    """
    Ensures the FlatDark Pro asset pack exists on disk.

    Notes:
    - We generate PNGs at runtime so the theme remains portable (no external designer dependency).
    - Existing files are NOT overwritten (unless force=True for icons that we intentionally refresh).
    """
    ASSETS_PATH.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Helpers
    # ----------------------------
    def rounded_rect(size, radius, fill, outline=None, outline_w=1):
        w, h = size
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle(
            (0, 0, w - 1, h - 1),
            radius=radius,
            fill=fill,
            outline=outline,
            width=outline_w,
        )
        return img

    def draw_checkmark(d: ImageDraw.ImageDraw, box, color, width=3):
        # box = (x0, y0, x1, y1)
        x0, y0, x1, y1 = box
        # Nice-ish proportions for 18..22px
        p1 = (x0 + (x1 - x0) * 0.20, y0 + (y1 - y0) * 0.55)
        p2 = (x0 + (x1 - x0) * 0.42, y0 + (y1 - y0) * 0.75)
        p3 = (x0 + (x1 - x0) * 0.80, y0 + (y1 - y0) * 0.28)
        d.line([p1, p2, p3], fill=color, width=width, joint="round")

    def checkbox_indicator(size, *, selected=False, disabled=False,
                           fill=(17, 19, 22, 255),
                           border=(43, 47, 54, 255),
                           tick=(76, 139, 245, 255),
                           radius=5):
        w, h = size
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)

        # Disabled overrides
        if disabled:
            border = (109, 117, 130, 180)
            tick = (109, 117, 130, 180)
            fill = (21, 23, 26, 255)

        d.rounded_rectangle((1, 1, w - 2, h - 2), radius=radius, fill=fill, outline=border, width=2)

        if selected:
            # Subtle inner glow for selected
            d.rounded_rectangle((3, 3, w - 4, h - 4), radius=max(2, radius - 2),
                                outline=tick, width=1)
            draw_checkmark(d, (2, 2, w - 3, h - 3), tick, width=3 if w >= 18 else 2)

        return img

    def save_png(img, name, force=False, expect_size=None):
        p = relative_to_assets(name)
        if p.exists() and (not force) and expect_size is not None:
            try:
                with Image.open(p) as cur:
                    if tuple(cur.size) != tuple(expect_size):
                        force = True
            except Exception:
                force = True
        if force or (not p.exists()):
            img.save(p)
            return
        # If the on-disk asset exists but differs (e.g. older pack), refresh it.
        try:
            with Image.open(p) as old:
                if old.size != img.size:
                    img.save(p)
        except Exception:
            img.save(p)

    # ----------------------------
    # Palette (RGBA)
    # ----------------------------
    C_BG     = (21, 23, 26, 255)      # #15171a
    C_PANEL  = (27, 30, 34, 255)      # #1b1e22
    C_FIELD  = (17, 19, 22, 255)      # #111316
    C_BORDER = (43, 47, 54, 255)      # #2b2f36
    C_MUTED  = (210, 214, 220, 220)
    C_ACCENT = (76, 139, 245, 255)   # #4c8bf5
    C_HOVER  = (39, 48, 68, 255)      # #273044
    C_PRESS  = (32, 40, 58, 255)      # #20283a
    C_DISAB  = (109, 117, 130, 255)  # #6d7582

    C_GREEN  = (56, 203, 137, 255)
    C_PURPLE = (180, 136, 255, 255)
    C_ORANGE = (255, 176, 90, 255)
    C_YELLOW = (255, 214, 102, 255)

    # ----------------------------
    # Icons (vector-ish with Pillow)
    # ----------------------------
    def icon_open(size, stroke=C_ACCENT):
        w, h = size
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((3, 7, w - 4, h - 4), radius=3, outline=stroke, width=2)
        d.rectangle((6, 5, w // 2, 10), outline=stroke, width=2)
        return img

    def icon_download(size, stroke=C_ACCENT):
        w, h = size
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.line((w // 2, 4, w // 2, h - 10), fill=stroke, width=2)
        d.polygon(
            [(w // 2 - 6, h - 12), (w // 2 + 6, h - 12), (w // 2, h - 4)],
            outline=stroke,
            fill=None,
        )
        d.rounded_rectangle((4, h - 9, w - 5, h - 4), radius=2, outline=stroke, width=2)
        return img

    def icon_moon(size, stroke=C_MUTED):
        w, h = size
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.arc((4, 4, w - 4, h - 4), start=40, end=320, fill=stroke, width=2)
        d.arc((w // 2, 4, w + 2, h - 4), start=60, end=300, fill=(0, 0, 0, 0), width=2)
        return img

    def icon_calc(size):
        w, h = size
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((4, 3, w - 5, h - 4), radius=3, outline=C_PURPLE, width=2)
        d.rectangle((6, 5, w - 7, 9), outline=C_PURPLE, width=2)
        for ry in [11, 15]:
            for cx in [7, 11, 15]:
                d.rectangle((cx, ry, cx + 2, ry + 2), outline=C_PURPLE, width=1)
        return img

    def icon_scan(size):
        w, h = size
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        # brackets + mid line
        d.line((5, 6, 9, 6), fill=C_GREEN, width=2)
        d.line((5, 6, 5, 10), fill=C_GREEN, width=2)
        d.line((w - 6, 6, w - 10, 6), fill=C_GREEN, width=2)
        d.line((w - 6, 6, w - 6, 10), fill=C_GREEN, width=2)
        d.line((5, h - 7, 9, h - 7), fill=C_GREEN, width=2)
        d.line((5, h - 7, 5, h - 11), fill=C_GREEN, width=2)
        d.line((w - 6, h - 7, w - 10, h - 7), fill=C_GREEN, width=2)
        d.line((w - 6, h - 7, w - 6, h - 11), fill=C_GREEN, width=2)
        d.line((8, h // 2, w - 9, h // 2), fill=C_GREEN, width=2)
        return img

    def icon_copy(size):
        w, h = size
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((7, 5, w - 4, h - 6), radius=3, outline=C_MUTED, width=2)
        d.rounded_rectangle((4, 8, w - 7, h - 3), radius=3, outline=C_MUTED, width=2)
        return img

    def icon_eye(size, off=False):
        w, h = size
        col = C_YELLOW if not off else C_MUTED
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.ellipse((3, 7, w - 4, h - 6), outline=col, width=2)
        d.ellipse((w // 2 - 2, h // 2 - 1, w // 2 + 2, h // 2 + 3), outline=col, width=2)
        if off:
            d.line((4, h - 6, w - 5, 6), fill=col, width=2)
        return img

    def icon_apply(size):
        w, h = size
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.line((5, h // 2, w // 2 - 1, h - 6), fill=C_GREEN, width=3)
        d.line((w // 2 - 1, h - 6, w - 5, 6), fill=C_GREEN, width=3)
        return img

    def icon_use(size):
        w, h = size
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((4, 9, w - 5, h - 4), radius=2, outline=C_ACCENT, width=2)
        d.line((w // 2, 4, w // 2, 12), fill=C_ACCENT, width=2)
        d.polygon([(w // 2 - 5, 8), (w // 2 + 5, 8), (w // 2, 3)], outline=C_ACCENT, fill=None)
        return img

    def icon_load(size):
        w, h = size
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((4, 4, w - 5, h - 5), radius=3, outline=C_ORANGE, width=2)
        d.line((4, 9, w - 5, 9), fill=C_ORANGE, width=2)
        return img

    # ----------------------------
    # Backplates (existing)
    # ----------------------------
    save_png(rounded_rect((34, 34), 10, fill=C_PANEL, outline=C_BORDER), "btn_icon_normal.png")
    save_png(rounded_rect((34, 34), 10, fill=C_HOVER, outline=(65, 76, 110, 255)), "btn_icon_hover.png")
    save_png(rounded_rect((34, 34), 10, fill=C_PRESS, outline=(65, 76, 110, 255)), "btn_icon_pressed.png")

    save_png(rounded_rect((190, 36), 11, fill=C_ACCENT, outline=C_ACCENT), "btn_primary_wide_normal.png", expect_size=(190,36))
    save_png(rounded_rect((190, 36), 11, fill=(63, 121, 223, 255), outline=(63, 121, 223, 255)), "btn_primary_wide_hover.png", expect_size=(190,36))
    save_png(rounded_rect((190, 36), 11, fill=(53, 102, 191, 255), outline=(53, 102, 191, 255)), "btn_primary_wide_pressed.png", expect_size=(190,36))

    # ----------------------------
    # NEW: scalable ttk button backplates (9-slice friendly)
    # ----------------------------
    # Base button
    save_png(rounded_rect((140, 32), 10, fill=C_PANEL, outline=C_BORDER), "btn_flat_normal.png", expect_size=(140,32))
    save_png(rounded_rect((140, 32), 10, fill=C_HOVER, outline=(65, 76, 110, 255)), "btn_flat_hover.png", expect_size=(140,32))
    save_png(rounded_rect((140, 32), 10, fill=C_PRESS, outline=(65, 76, 110, 255)), "btn_flat_pressed.png", expect_size=(140,32))
    save_png(rounded_rect((140, 32), 10, fill=C_BG, outline=C_BORDER), "btn_flat_disabled.png", expect_size=(140,32))

    # Primary button
    save_png(rounded_rect((140, 32), 10, fill=C_ACCENT, outline=C_ACCENT), "btn_primary_normal.png", expect_size=(140,32))
    save_png(rounded_rect((140, 32), 10, fill=(63, 121, 223, 255), outline=(63, 121, 223, 255)), "btn_primary_hover.png", expect_size=(140,32))
    save_png(rounded_rect((140, 32), 10, fill=(53, 102, 191, 255), outline=(53, 102, 191, 255)), "btn_primary_pressed.png", expect_size=(140,32))
    save_png(rounded_rect((140, 32), 10, fill=C_BORDER, outline=C_BORDER), "btn_primary_disabled.png", expect_size=(140,32))

    # Menubutton (slightly more "field-like")
    save_png(rounded_rect((140, 32), 10, fill=C_FIELD, outline=C_BORDER), "btn_menu_normal.png", expect_size=(140,32))
    save_png(rounded_rect((140, 32), 10, fill=C_HOVER, outline=(65, 76, 110, 255)), "btn_menu_hover.png", expect_size=(140,32))
    save_png(rounded_rect((140, 32), 10, fill=C_PRESS, outline=(65, 76, 110, 255)), "btn_menu_pressed.png", expect_size=(140,32))
    save_png(rounded_rect((140, 32), 10, fill=C_BG, outline=C_BORDER), "btn_menu_disabled.png", expect_size=(140,32))

    # Checkbuttons
    save_png(checkbox_indicator((18, 18), selected=False, disabled=False, fill=C_FIELD, border=C_BORDER, tick=C_ACCENT), "cb_off.png")
    save_png(checkbox_indicator((18, 18), selected=True,  disabled=False, fill=C_FIELD, border=C_BORDER, tick=C_ACCENT), "cb_on.png")
    save_png(checkbox_indicator((18, 18), selected=False, disabled=True,  fill=C_FIELD, border=C_BORDER, tick=C_ACCENT), "cb_off_disabled.png")
    save_png(checkbox_indicator((18, 18), selected=True,  disabled=True,  fill=C_FIELD, border=C_BORDER, tick=C_ACCENT), "cb_on_disabled.png")

    # ----------------------------
    # Icons (force refresh)
    # ----------------------------
    save_png(icon_open((22, 22), stroke=C_ACCENT), "ico_open.png", force=True)
    save_png(icon_download((22, 22), stroke=C_ACCENT), "ico_generate.png", force=True)
    save_png(icon_moon((22, 22), stroke=C_MUTED), "ico_theme.png", force=True)
    save_png(icon_calc((22, 22)), "ico_calc.png", force=True)
    save_png(icon_eye((22, 22), off=False), "ico_eye.png", force=True)
    save_png(icon_eye((22, 22), off=True), "ico_eye_off.png", force=True)
    save_png(icon_open((22, 22), stroke=C_ACCENT), "ico_folder.png", force=True)
    save_png(icon_scan((22, 22)), "ico_scan.png", force=True)
    save_png(icon_copy((22, 22)), "ico_copy.png", force=True)
    save_png(icon_use((22, 22)), "ico_use.png", force=True)
    save_png(icon_apply((22, 22)), "ico_apply.png", force=True)
    # App icon (for window + packaging). We generate an .ico for Windows and keep a PNG fallback.
    try:
        icon_src = OUTPUT_PATH / "Icon_512x512_RaptorBrush_fix1.png"
        if icon_src.exists():
            with Image.open(icon_src).convert("RGBA") as ic:
                # Store a PNG copy in assets (useful for iconphoto and non-Windows platforms)
                save_png(ic, "app_icon.png", force=True)
                # Generate ICO (Windows titlebar/taskbar + PyInstaller-friendly)
                ico_path = relative_to_assets("app_icon.ico")
                # Always refresh to avoid stale/partial ICOs
                ic.save(
                    ico_path,
                    format="ICO",
                    sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
                )
    except Exception:
        pass


class CanvasButton:
    def __init__(self, canvas: tk.Canvas, x: int, y: int, img_normal, img_hover, img_pressed, command):
        self.canvas = canvas
        self.command = command
        self._img_n = img_normal
        self._img_h = img_hover
        self._img_p = img_pressed
        self._state = "normal"

        self.item = canvas.create_image(x, y, image=self._img_n, anchor="nw")
        canvas.tag_bind(self.item, "<Enter>", self._on_enter)
        canvas.tag_bind(self.item, "<Leave>", self._on_leave)
        canvas.tag_bind(self.item, "<ButtonPress-1>", self._on_press)
        canvas.tag_bind(self.item, "<ButtonRelease-1>", self._on_release)

    def _set_state(self, state: str):
        self._state = state
        img = self._img_n if state == "normal" else self._img_h if state == "hover" else self._img_p
        self.canvas.itemconfigure(self.item, image=img)

    def _on_enter(self, _):
        self._set_state("hover")

    def _on_leave(self, _):
        self._set_state("normal")

    def _on_press(self, _):
        self._set_state("pressed")

    def _on_release(self, _):
        self._set_state("hover")
        if callable(self.command):
            self.command()


class PreviewGUI(BasePreviewGUI):
    def __init__(self):
        # Guard for BasePreviewGUI callbacks during super().__init__ (notably _apply_language).
        # Wrapper post-processing (icons/shortcuts/sizing) should only run after our init completes.
        self._wrapper_ready = False
        super().__init__()

        # settings.json (validated)
        try:
            self._settings, self._settings_path, self._settings_dir = _load_settings_bundle(OUTPUT_PATH)
        except Exception:
            self._settings, self._settings_path, self._settings_dir = {}, None, OUTPUT_PATH
        self._apply_settings_early()

        _ensure_assets()
        self._images = {}
        self._ico_cache = {}

        self._apply_flatdark_theme()
        self._apply_window_chrome_and_icon()
        self._install_custom_ttk_styles()
        self._apply_custom_styles_to_widgets()

        self._tune_dark_surfaces()
        self._install_dyes_grid_fit()
        self._hook_popdown_styling()


        self._install_wheel_router()
        self._install_long_optionmenu_dropdowns()
        self._promote_primary_buttons()
        self._attach_button_icons()
        self._update_shortcut_labels()
        self._tune_action_button_sizes()

        # Re-apply sizing after icons/translations and late widget creation (e.g. AdvancedPanel)
        self.after_idle(self._tune_action_button_sizes)
        self.after(250, self._tune_action_button_sizes)
        self.after(900, self._tune_action_button_sizes)

        # settings persistence + late application (after styles/icons)
        self._install_settings_persistence()
        try:
            self.after_idle(self._apply_settings_late)
        except Exception:
            pass

        self._bind_shortcuts()
        self._wrap_actions_for_status()

        # Wrapper is now fully ready.
        self._wrapper_ready = True

    # -------------------------------------------------
    # i18n hook: BasePreviewGUI rewrites button labels on language apply.
    # We re-append shortcut hints and ensure sizing after every language refresh.
    # -------------------------------------------------
    def _apply_language(self):
        # Always let Base apply its bindings first.
        try:
            super()._apply_language()
        except Exception:
            # If Base fails, don't crash; keep wrapper alive.
            pass

        if not getattr(self, "_wrapper_ready", False):
            return

        # Re-apply our shortcut labels (Ctrl+O / Ctrl+G) after Base overwrote the text.
        try:
            self._update_shortcut_labels()
        except Exception:
            pass

        # Ensure widths are recalculated for current language (ES is longer than EN).
        try:
            self._tune_action_button_sizes()
        except Exception:
            pass

    # ----------------------------
    # Theme (colors + baseline)
    # ----------------------------
    def _apply_flatdark_theme(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        self._PAL = {
            "BG": "#15171a",
            "PANEL": "#1b1e22",
            "FIELD": "#111316",
            "BORDER": "#2b2f36",
            "FG": "#e6e6e6",
            "MUTED": "#a9b0bb",
            "ACCENT": "#4c8bf5",
            "HOVER": "#273044",
            "PRESSED": "#20283a",
            "DISABLED": "#6d7582",
        }
        p = self._PAL

        style.configure(".", background=p["BG"], foreground=p["FG"])
        style.configure("TFrame", background=p["BG"])
        style.configure("TLabelframe", background=p["BG"], bordercolor=p["BORDER"])
        style.configure("TLabelframe.Label", background=p["BG"], foreground=p["FG"], font=("Segoe UI Semibold", 10))

        style.configure("TLabel", background=p["BG"], foreground=p["FG"], font=("Segoe UI", 10))
        style.configure("Hint.TLabel", foreground=p["MUTED"])

        # Baseline styles (fallbacks)
        style.configure(
            "TButton",
            background=p["PANEL"],
            foreground=p["FG"],
            borderwidth=1,
            relief="flat",
            padding=(10, 6),
            font=("Segoe UI Semibold", 10),
        )
        style.map(
            "TButton",
            background=[("active", p["HOVER"]), ("pressed", p["PRESSED"]), ("disabled", p["BG"])],
            foreground=[("active", p["FG"]), ("pressed", p["FG"]), ("disabled", p["DISABLED"])],
        )

        style.configure(
            "TEntry",
            fieldbackground=p["FIELD"],
            background=p["FIELD"],
            foreground=p["FG"],
            bordercolor=p["BORDER"],
            padding=(8, 6),
        )

        # Keep disabled/readonly entries dark (Windows/Tk sometimes falls back to light system colors)
        try:
            style.map(
                "TEntry",
                fieldbackground=[("disabled", p["BG"]), ("readonly", p["FIELD"]), ("!disabled", p["FIELD"])],
                foreground=[("disabled", p["DISABLED"]), ("!disabled", p["FG"])],
            )
        except Exception:
            pass

        # Combobox/Spinbox (avoid white fields on Windows)
        try:
            style.configure(
                "TCombobox",
                fieldbackground=p["FIELD"],
                background=p["FIELD"],
                foreground=p["FG"],
                bordercolor=p["BORDER"],
                arrowcolor=p["FG"],
                padding=(8, 6),
            )
            style.map(
                "TCombobox",
                fieldbackground=[("readonly", p["FIELD"]), ("disabled", p["BG"]), ("!disabled", p["FIELD"])],
                background=[("active", p["HOVER"]), ("readonly", p["FIELD"]), ("disabled", p["BG"])],
                foreground=[("disabled", p["DISABLED"]), ("readonly", p["FG"]), ("!disabled", p["FG"])],
                arrowcolor=[("disabled", p["DISABLED"]), ("!disabled", p["FG"])],
            )
        except Exception:
            pass

        try:
            style.configure(
                "TSpinbox",
                fieldbackground=p["FIELD"],
                background=p["FIELD"],
                foreground=p["FG"],
                bordercolor=p["BORDER"],
                arrowcolor=p["FG"],
                padding=(8, 6),
            )
            style.map(
                "TSpinbox",
                fieldbackground=[("disabled", p["BG"]), ("!disabled", p["FIELD"])],
                background=[("active", p["HOVER"]), ("disabled", p["BG"])],
                foreground=[("disabled", p["DISABLED"]), ("!disabled", p["FG"])],
                arrowcolor=[("disabled", p["DISABLED"]), ("!disabled", p["FG"])],
            )
        except Exception:
            pass

        style.configure(
            "TMenubutton",
            background=p["FIELD"],
            foreground=p["FG"],
            bordercolor=p["BORDER"],
            padding=(10, 6),
        )
        style.map(
            "TMenubutton",
            background=[("active", p["HOVER"]), ("pressed", p["PRESSED"])],
            foreground=[("active", p["FG"]), ("pressed", p["FG"])],
        )

        style.configure("TCheckbutton", background=p["BG"], foreground=p["FG"], font=("Segoe UI", 10))
        style.map(
            "TCheckbutton",
            foreground=[("disabled", p["DISABLED"]), ("!disabled", p["FG"])],
            background=[("disabled", p["BG"]), ("active", p["BG"]), ("selected", p["BG"]), ("!disabled", p["BG"])],
        )

        style.configure("TScrollbar", background=p["BG"], troughcolor=p["FIELD"], bordercolor=p["BORDER"])

        style.configure(
            "Treeview",
            background=p["FIELD"],
            fieldbackground=p["FIELD"],
            foreground=p["FG"],
            bordercolor=p["BORDER"],
            rowheight=24,
            font=("Segoe UI", 9),
        )
        style.configure(
            "Treeview.Heading",
            background=p["PANEL"],
            foreground=p["FG"],
            relief="flat",
            font=("Segoe UI Semibold", 9),
        )
        style.map("Treeview", background=[("selected", "#2b3b5f")], foreground=[("selected", "#ffffff")])

        # Tk option database defaults (menus / listboxes / etc.).
        # These matter for classic tk widgets (OptionMenu popup menus) and some ttk popdowns.
        try:
            self.option_add("*Menu.background", p["FIELD"])
            self.option_add("*Menu.foreground", p["FG"])
            self.option_add("*Menu.activeBackground", p["HOVER"])
            self.option_add("*Menu.activeForeground", p["FG"])
            self.option_add("*Menu.disabledForeground", p["DISABLED"])
            self.option_add("*Menu.borderWidth", 0)
            self.option_add("*Menu.relief", "flat")

            self.option_add("*Listbox.background", p["FIELD"])
            self.option_add("*Listbox.foreground", p["FG"])
            self.option_add("*Listbox.selectBackground", p["ACCENT"])
            self.option_add("*Listbox.selectForeground", "#ffffff")
            self.option_add("*Listbox.borderWidth", 0)
            self.option_add("*Listbox.relief", "flat")
            # Canvas defaults (some scroll viewports are tk.Canvas and default to system light colors)
            self.option_add("*Canvas.background", p["FIELD"])
            self.option_add("*Canvas.highlightThickness", 0)

            # ttk Combobox popdown listbox sometimes overrides generic Listbox options; target it explicitly too
            self.option_add("*TCombobox*Listbox.background", p["FIELD"])
            self.option_add("*TCombobox*Listbox.foreground", p["FG"])
            self.option_add("*TCombobox*Listbox.selectBackground", p["ACCENT"])
            self.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")

        except Exception:
            pass

        try:
            self.configure(bg=p["BG"])
        except Exception:
            pass

    # ----------------------------
    # Widget walking
    # ----------------------------
    def _walk_widgets(self):
        stack = [self]
        while stack:
            w = stack.pop()
            yield w
            try:
                stack.extend(w.winfo_children())
            except Exception:
                pass

    # ----------------------------
    # Assets helpers
    # ----------------------------
    def _img(self, name: str, size=None):
        key = (name, size)
        if key in self._images:
            return self._images[key]
        img = Image.open(relative_to_assets(name)).convert("RGBA")
        if size:
            try:
                resample = Image.Resampling.LANCZOS
            except Exception:
                resample = Image.LANCZOS
            img = img.resize(size, resample=resample)
        tkimg = ImageTk.PhotoImage(img)
        self._images[key] = tkimg
        return tkimg

    # ----------------------------
    # NEW: Custom ttk elements (buttons / checkbuttons / menubuttons)
    # ----------------------------
    def _install_custom_ttk_styles(self):
        style = ttk.Style(self)
        p = self._PAL

        # Keep references alive (PhotoImage must not be GC'ed)
        self._pro_img = {}

        def _load(name):
            self._pro_img[name] = self._img(name)
            return self._pro_img[name]

        # -------- Buttons (9-slice / scalable) --------
        btn_n = _load("btn_flat_normal.png")
        btn_h = _load("btn_flat_hover.png")
        btn_p = _load("btn_flat_pressed.png")
        btn_d = _load("btn_flat_disabled.png")

        pri_n = _load("btn_primary_normal.png")
        pri_h = _load("btn_primary_hover.png")
        pri_p = _load("btn_primary_pressed.png")
        pri_d = _load("btn_primary_disabled.png")

        menu_n = _load("btn_menu_normal.png")
        menu_h = _load("btn_menu_hover.png")
        menu_p = _load("btn_menu_pressed.png")
        menu_d = _load("btn_menu_disabled.png")

        border = (10, 10, 10, 10)

        # Elements may already exist if hot-reloaded; ignore if so
        try:
            style.element_create(
                "Pro.Button",
                "image",
                btn_n,
                ("active", btn_h),
                ("pressed", btn_p),
                ("disabled", btn_d),
                border=border,
                sticky="nsew",
            )
        except tk.TclError:
            pass

        try:
            style.element_create(
                "Pro.PrimaryButton",
                "image",
                pri_n,
                ("active", pri_h),
                ("pressed", pri_p),
                ("disabled", pri_d),
                border=border,
                sticky="nsew",
            )
        except tk.TclError:
            pass

        try:
            style.element_create(
                "Pro.MenuButton",
                "image",
                menu_n,
                ("active", menu_h),
                ("pressed", menu_p),
                ("disabled", menu_d),
                border=border,
                sticky="nsew",
            )
        except tk.TclError:
            pass

        # Layouts
        try:
            style.layout(
                "Pro.TButton",
                [
                    ("Pro.Button", {"sticky": "nsew", "children": [
                        ("Button.padding", {"sticky": "nsew", "children": [
                            ("Button.label", {"sticky": "nsew"})
                        ]})
                    ]})
                ],
            )
        except tk.TclError:
            # If a given Tk build is stricter about layout specs, fall back to inherited layout.
            pass
        style.configure(
            "Pro.TButton",
            foreground=p["FG"],
            padding=(8, 3),
            font=("Segoe UI Semibold", 9),
        )
        style.map("Pro.TButton", foreground=[("disabled", p["DISABLED"])])

        # Smaller font variant (for "shortcut" labels) - still image-based
        style.configure("ProShortcut.TButton", padding=(8, 2), font=("Segoe UI", 9), foreground=p["FG"])
        try:
            style.layout("ProShortcut.TButton", style.layout("Pro.TButton"))
        except tk.TclError:
            pass
        style.map("ProShortcut.TButton", foreground=[("disabled", p["DISABLED"])])

        # Primary variants
        try:
            style.layout(
                "ProPrimary.TButton",
                [
                    ("Pro.PrimaryButton", {"sticky": "nsew", "children": [
                        ("Button.padding", {"sticky": "nsew", "children": [
                            ("Button.label", {"sticky": "nsew"})
                        ]})
                    ]})
                ],
            )
        except tk.TclError:
            pass
        style.configure(
            "ProPrimary.TButton",
            foreground="#ffffff",
            padding=(10, 3),
            font=("Segoe UI Semibold", 9),
        )
        style.map("ProPrimary.TButton", foreground=[("disabled", p["DISABLED"])])

        style.configure(
            "ProPrimaryShortcut.TButton",
            foreground="#ffffff",
            padding=(10, 2),
            font=("Segoe UI Semibold", 9),
        )
        try:
            style.layout("ProPrimaryShortcut.TButton", style.layout("ProPrimary.TButton"))
        except tk.TclError:
            pass
        style.map("ProPrimaryShortcut.TButton", foreground=[("disabled", p["DISABLED"])])

        # -------- Menubutton --------
        try:
            style.layout(
                "Pro.TMenubutton",
                [
                    ("Pro.MenuButton", {"sticky": "nsew", "children": [
                        ("Menubutton.padding", {"sticky": "nsew", "children": [
                            ("Menubutton.label", {"sticky": "nsew"})
                        ]})
                    ]})
                ],
            )
        except tk.TclError:
            pass
        style.configure(
            "Pro.TMenubutton",
            foreground=p["FG"],
            padding=(10, 5),
            font=("Segoe UI", 10),
        )
        style.map("Pro.TMenubutton", foreground=[("disabled", p["DISABLED"])])

        # -------- Checkbutton indicator --------
        cb_off = _load("cb_off.png")
        cb_on = _load("cb_on.png")
        cb_off_d = _load("cb_off_disabled.png")
        cb_on_d = _load("cb_on_disabled.png")

        try:
            style.element_create(
                "Pro.Checkbutton.indicator",
                "image",
                cb_off,
                ("selected", cb_on),
                ("disabled", cb_off_d),
                ("selected", "disabled", cb_on_d),
                border=(4, 4, 4, 4),
                sticky="w",
            )
        except tk.TclError:
            pass

        try:
            style.layout(
                "Pro.TCheckbutton",
                [
                    # NOTE: ttk layout specs are strict about supported keys.
                    # Avoid non-layout keys (e.g. 'padx'), which can break Tcl parsing on some Tk builds.
                    ("Checkbutton.padding", {"sticky": "nsew", "children": [
                        ("Pro.Checkbutton.indicator", {"side": "left", "sticky": ""}),
                        ("Checkbutton.label", {"side": "left", "sticky": "w"}),
                    ]})
                ],
            )
        except tk.TclError:
            # Fall back to inherited layout if this Tk build rejects our spec.
            pass
        style.configure(
            "Pro.TCheckbutton",
            background=p["BG"],
            foreground=p["FG"],
            font=("Segoe UI", 10),
            padding=(2, 2),
        )
        # Force dark background across states (prevents occasional white strips in some Tk builds)
        style.map(
            "Pro.TCheckbutton",
            foreground=[("disabled", p["DISABLED"]), ("!disabled", p["FG"])],
            background=[("disabled", p["BG"]), ("active", p["BG"]), ("selected", p["BG"]), ("!disabled", p["BG"])],
        )

    def _apply_custom_styles_to_widgets(self):
        """
        Assign our custom styles to all existing ttk widgets, without touching BasePreviewGUI layouts.
        Safe to re-run after creating new widgets (e.g. AdvancedPanel).
        """
        for w in self._walk_widgets():
            try:
                self._ensure_wheelrouter_tag(w)
            except Exception:
                pass
            # Buttons
            if isinstance(w, ttk.Button):
                try:
                    cur = str(w.cget("style") or "")
                except Exception:
                    cur = ""
                # Don't override any explicitly custom styles (if user sets them elsewhere)
                if cur in ("", "TButton", "Shortcut.TButton", "Primary.TButton", "PrimaryShortcut.TButton", "Pro.TButton", "ProShortcut.TButton", "ProPrimary.TButton", "ProPrimaryShortcut.TButton"):
                    # Default as Pro.TButton (we refine specific ones later via msgid)
                    try:
                        w.configure(style="Pro.TButton")
                    except Exception:
                        pass

            # Menubuttons (OptionMenu)
            if isinstance(w, ttk.Menubutton):
                try:
                    cur = str(w.cget("style") or "")
                except Exception:
                    cur = ""
                if cur in ("", "TMenubutton", "Pro.TMenubutton"):
                    try:
                        w.configure(style="Pro.TMenubutton")
                    except Exception:
                        pass

            # Checkbuttons
            if isinstance(w, ttk.Checkbutton):
                try:
                    cur = str(w.cget("style") or "")
                except Exception:
                    cur = ""
                if cur in ("", "TCheckbutton", "Pro.TCheckbutton"):
                    try:
                        w.configure(style="Pro.TCheckbutton")
                    except Exception:
                        pass
    # ----------------------------
    # tk widget tuning (includes tk.Button hover fixes)
    # ----------------------------
    def _tune_dark_surfaces(self):
        p = self._PAL

        # Preview canvas bg
        try:
            self.preview_canvas.configure(bg=p["FIELD"])
        except Exception:
            pass

        # Left controls scroll canvas (if exists)
        try:
            self._controls_canvas.configure(bg=p["BG"])
        except Exception:
            pass

        # Fix legacy tk widgets that default to white (OptionMenu, canvases, etc.)
        def _is_light_color(c):
            try:
                r, g, b = self.winfo_rgb(c)
                return (r + g + b) / 3 > 40000
            except Exception:
                s = str(c).strip().lower()
                return s in ("", "white", "#ffffff", "#f0f0f0", "systemwindow", "systembuttonface", "systemwindowframe")
        for w in self._walk_widgets():
            # tk.Button (rare, but keep)
            if isinstance(w, tk.Button) and not isinstance(w, ttk.Button):
                try:
                    w.configure(
                        bg=p["PANEL"],
                        fg=p["FG"],
                        activebackground=p["HOVER"],
                        activeforeground=p["FG"],
                        relief="flat",
                        bd=1,
                        highlightthickness=0,
                    )
                except Exception:
                    pass

            # tk.Scale used for some sliders
            if isinstance(w, tk.Scale):
                try:
                    w.configure(
                        bg=p["BG"],
                        fg=p["FG"],
                        troughcolor=p["FIELD"],
                        highlightthickness=0,
                        bd=0,
                    )
                except Exception:
                    pass

            # tk.Canvas used for scroll viewports (e.g. dyes list) – defaults to system light colors
            if isinstance(w, tk.Canvas):
                try:
                    cur = w.cget("bg")
                    if _is_light_color(cur):
                        w.configure(bg=p["FIELD"])
                    w.configure(highlightthickness=0, bd=0)
                except Exception:
                    pass

            # tk.Listbox sometimes used for search results (can default to light)
            if isinstance(w, tk.Listbox):
                try:
                    cur = w.cget("bg")
                    if _is_light_color(cur):
                        w.configure(
                            bg=p["FIELD"],
                            fg=p["FG"],
                            selectbackground=p["ACCENT"],
                            selectforeground="#ffffff",
                        )
                    w.configure(highlightthickness=0, bd=0, relief="flat")
                except Exception:
                    pass

            
            # tk.Frame embedded inside canvases (scrollable containers) can default to light system bg
            if isinstance(w, tk.Frame) and not isinstance(w, ttk.Frame) and not isinstance(w, (tk.Tk, tk.Toplevel)):
                try:
                    cur = w.cget("bg")
                    if _is_light_color(cur):
                        target = p["FIELD"] if isinstance(getattr(w, "master", None), tk.Canvas) else p["BG"]
                        w.configure(bg=target)
                except Exception:
                    pass

# Classic OptionMenu uses tk.Menubutton + tk.Menu (not ttk)
            if isinstance(w, tk.Menubutton) and not isinstance(w, ttk.Menubutton):
                try:
                    w.configure(
                        bg=p["FIELD"],
                        fg=p["FG"],
                        activebackground=p["HOVER"],
                        activeforeground=p["FG"],
                        disabledforeground=p["DISABLED"],
                        relief="flat",
                        bd=1,
                        highlightthickness=1,
                        highlightbackground=p["BORDER"],
                        highlightcolor=p["ACCENT"],
                        padx=10,
                        pady=5,
                    )
                except Exception:
                    pass
                # Style the popup menu itself
                try:
                    mname = w.cget("menu")
                    if mname:
                        menu = self.nametowidget(mname) if isinstance(mname, str) else mname
                        if isinstance(menu, tk.Menu):
                            menu.configure(
                                bg=p["FIELD"],
                                fg=p["FG"],
                                activebackground=p["HOVER"],
                                activeforeground=p["FG"],
                                disabledforeground=p["DISABLED"],
                                borderwidth=0,
                                relief="flat",
                            )
                except Exception:
                    pass

        
            # ttk.Menubutton / ttk.OptionMenu uses a classic tk.Menu too; ensure popup menu is dark
            if isinstance(w, ttk.Menubutton):
                try:
                    mname = w.cget("menu")
                    if mname:
                        menu = self.nametowidget(mname) if isinstance(mname, str) else mname
                        if isinstance(menu, tk.Menu):
                            menu.configure(
                                bg=p["FIELD"],
                                fg=p["FG"],
                                activebackground=p["HOVER"],
                                activeforeground=p["FG"],
                                disabledforeground=p["DISABLED"],
                                borderwidth=0,
                                relief="flat",
                            )
                except Exception:
                    pass

# Dye swatches background
        try:
            for sw in getattr(self, "dye_swatches", {}).values():
                sw.configure(highlightbackground=p["BORDER"])
                sw.itemconfigure(getattr(sw, "_frame_rect", None), fill=p["FIELD"], outline="")
        except Exception:
            pass


    # ----------------------------
    # Window chrome + icon (Windows dark titlebar + optional border colors)
    # ----------------------------
    def _apply_window_chrome_and_icon(self):
        # Main window
        try:
            self._set_window_icon(self)
        except Exception:
            pass
        try:
            self._apply_windows_dark_titlebar(self)
        except Exception:
            pass

    def _set_window_icon(self, win):
        # Prefer ICO on Windows; fall back to iconphoto with PNG.
        ico_path = relative_to_assets("app_icon.ico")
        png_path = relative_to_assets("app_icon.png")

        if sys.platform.startswith("win") and ico_path.exists():
            try:
                win.iconbitmap(str(ico_path))
                return
            except Exception:
                pass

        # PNG fallback (cross-platform)
        try:
            if png_path.exists():
                img = Image.open(png_path).convert("RGBA")
                self._app_icon_photo = ImageTk.PhotoImage(img)
                win.iconphoto(True, self._app_icon_photo)
        except Exception:
            pass

    def _hex_to_colorref(self, hex_color: str) -> int:
        # Windows COLORREF is 0x00BBGGRR
        s = str(hex_color).strip().lstrip("#")
        if len(s) != 6:
            return 0
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
        return (b << 16) | (g << 8) | r

    def _apply_windows_dark_titlebar(self, win):
        if not sys.platform.startswith("win"):
            return

        try:
            hwnd = win.winfo_id()
        except Exception:
            return

        try:
            dwm = ctypes.windll.dwmapi
        except Exception:
            return

        # Enable dark mode titlebar (attr 20 on newer builds, 19 on older)
        try:
            val = ctypes.c_int(1)
            for attr in (20, 19):
                try:
                    dwm.DwmSetWindowAttribute(hwnd, ctypes.c_int(attr), ctypes.byref(val), ctypes.sizeof(val))
                    break
                except Exception:
                    continue
        except Exception:
            pass

        # Optional (Win11): caption/border colors. Fails silently on older builds.
        try:
            bg = self._PAL.get("BG", "#15171a")
            cref = ctypes.c_int(self._hex_to_colorref(bg))
            # Known attrs (Win11): 34 caption, 35 border
            for attr in (34, 35):
                try:
                    dwm.DwmSetWindowAttribute(hwnd, ctypes.c_int(attr), ctypes.byref(cref), ctypes.sizeof(cref))
                except Exception:
                    pass
        except Exception:
            pass

    # ----------------------------
    # Advanced window: apply chrome/icon when created (no base GUI edits)
    # ----------------------------
    def _ensure_advanced_window(self):
        super()._ensure_advanced_window()
        try:
            aw = getattr(self, "_advanced_win", None)
            if aw is not None and aw.winfo_exists():
                self._set_window_icon(aw)
                self._apply_windows_dark_titlebar(aw)
        except Exception:
            pass

    # ----------------------------
    # Dyes grid: responsive columns (tight fit + compact reflow after filtering)
    # ----------------------------
    def _install_dyes_grid_fit(self):
        # Base GUI creates a scrollable canvas with a create_window id in self._dyes_canvas_window
        try:
            self._dyes_canvas = getattr(self, "dyes_list_frame", None).master
        except Exception:
            self._dyes_canvas = None

        if not isinstance(self._dyes_canvas, tk.Canvas):
            return

        self._dyes_reflow_job = None

        def _on_canvas_cfg(_e=None):
            # Keep the inner frame width synced with the viewport width
            try:
                w = int(self._dyes_canvas.winfo_width())
                if w > 20 and hasattr(self, "_dyes_canvas_window"):
                    self._dyes_canvas.itemconfigure(self._dyes_canvas_window, width=w)
            except Exception:
                pass
            self._schedule_reflow_dyes_grid()

        try:
            self._dyes_canvas.bind("<Configure>", _on_canvas_cfg, add="+")
        except Exception:
            pass

        # Initial reflow after Tk has measured sizes
        try:
            self.after_idle(self._schedule_reflow_dyes_grid)
        except Exception:
            pass

    def _schedule_reflow_dyes_grid(self):
        try:
            if getattr(self, "_dyes_reflow_job", None) is not None:
                self.after_cancel(self._dyes_reflow_job)
        except Exception:
            pass
        try:
            self._dyes_reflow_job = self.after(25, self._reflow_dyes_grid)
        except Exception:
            self._dyes_reflow_job = None

    def _reflow_dyes_grid(self):
        self._dyes_reflow_job = None
        dyes_canvas = getattr(self, "_dyes_canvas", None)
        if not isinstance(dyes_canvas, tk.Canvas):
            return

        frame = getattr(self, "dyes_list_frame", None)
        if frame is None:
            return

        try:
            vw = int(dyes_canvas.winfo_width())
        except Exception:
            vw = 0
        if vw < 40:
            try:
                vw = int(dyes_canvas.winfo_reqwidth())
            except Exception:
                vw = 280

        # Sync create_window width
        try:
            if hasattr(self, "_dyes_canvas_window"):
                dyes_canvas.itemconfigure(self._dyes_canvas_window, width=vw)
        except Exception:
            pass

        swatches = getattr(self, "dye_swatches", {}) or {}
        try:
            sample = next(iter(swatches.values()))
        except Exception:
            return

        try:
            sw = int(sample.winfo_reqwidth()) or 18
        except Exception:
            sw = 18

        min_pad = 3
        pad_y = 3

        max_cols = 24
        cols = max(1, min(max_cols, vw // (sw + 2 * min_pad)))
        while cols > 1:
            cell_w = vw / cols
            pad_x = int((cell_w - sw) / 2)
            if pad_x >= min_pad:
                break
            cols -= 1

        cell_w = vw / cols
        pad_x = max(min_pad, int((cell_w - sw) / 2))
        used_w = cols * (sw + 2 * pad_x)
        margin = max(0, int((vw - used_w) // 2))

        # Center window item in the canvas
        try:
            if hasattr(self, "_dyes_canvas_window"):
                dyes_canvas.coords(self._dyes_canvas_window, margin, 0)
        except Exception:
            pass

        translator = getattr(getattr(self, "controller", None), "_ark_translator", None)
        if translator is None:
            visible = [c for c in swatches.values() if c.winfo_ismapped()]
        else:
            visible = []
            for dye in translator.dyes:
                c = swatches.get(dye.observed_byte)
                if c is not None and c.winfo_ismapped():
                    visible.append(c)

        for i, c in enumerate(visible):
            r = i // cols
            cidx = i % cols
            try:
                c.grid_configure(row=r, column=cidx, padx=pad_x, pady=pad_y)
            except Exception:
                pass

    # Override: make filtering reflow the grid so it stays compact and aligned
    def _filter_dyes_grid(self, event=None):
        query = ""
        try:
            query = self.dyes_search_var.get().strip().lower()
        except Exception:
            query = ""

        translator = getattr(getattr(self, "controller", None), "_ark_translator", None)
        if translator is None:
            return

        swatches = getattr(self, "dye_swatches", {}) or {}

        for dye in translator.dyes:
            swatch = swatches.get(dye.observed_byte)
            if swatch is None:
                continue

            name = str(getattr(dye, "name", "")).lower()
            id_str = str(getattr(dye, "observed_byte", ""))

            match = (not query) or (query in name) or (query in id_str)
            try:
                if match:
                    swatch.grid()
                else:
                    swatch.grid_remove()
            except Exception:
                pass

        self._schedule_reflow_dyes_grid()

    # ----------------------------
    # Popdowns (Combobox list) styling
    # ----------------------------
    def _hook_popdown_styling(self):
        """Bind ttk.Combobox instances so their popdown list inherits our dark palette."""
        try:
            for w in self._walk_widgets():
                if isinstance(w, ttk.Combobox):
                    self._bind_combobox_popdown(w)
        except Exception:
            pass

    def _bind_combobox_popdown(self, cb: ttk.Combobox):
        if getattr(cb, "_pro_popdown_bound", False):
            return
        cb._pro_popdown_bound = True

        def _schedule_apply(_evt=None):
            # Popdown window is typically created on first open; configure after Tk has mapped it.
            # Some Tk builds create/map it slightly later, so we try a few times.
            try:
                self.after_idle(lambda: self._style_combobox_popdown(cb))
                self.after(8, lambda: self._style_combobox_popdown(cb))
                self.after(35, lambda: self._style_combobox_popdown(cb))
                self.after(90, lambda: self._style_combobox_popdown(cb))
            except Exception:
                pass

        try:
            cb.bind("<Button-1>", _schedule_apply, add="+")
            cb.bind("<KeyRelease-Down>", _schedule_apply, add="+")
            cb.bind("<KeyPress-Down>", _schedule_apply, add="+")
            cb.bind("<Alt-Down>", _schedule_apply, add="+")
        except Exception:
            pass

    def _style_combobox_popdown(self, cb: ttk.Combobox):
        p = getattr(self, "_PAL", None) or {}
        if not p:
            return
        try:
            # Tcl helper returns the path of the popdown window for this combobox.
            pop = cb.tk.call("ttk::combobox::PopdownWindow", cb)
            # Tk 8.6 layout: <pop>.f.l (listbox), <pop>.f.sb (scrollbar)
            try:
                pw = self.nametowidget(pop)
                pw.configure(bg=p["FIELD"])
            except Exception:
                pass
            try:
                fr = self.nametowidget(f"{pop}.f")
                fr.configure(bg=p["FIELD"])
            except Exception:
                pass
            lb_path = f"{pop}.f.l"
            sb_path = f"{pop}.f.sb"
            try:
                lb = self.nametowidget(lb_path)
                if isinstance(lb, tk.Listbox):
                    lb.configure(
                        bg=p["FIELD"],
                        fg=p["FG"],
                        selectbackground=p["ACCENT"],
                        selectforeground="#ffffff",
                        highlightthickness=0,
                        bd=0,
                        relief="flat",
                    )
                    try:
                        lb._pro_is_popdown = True
                        self._ensure_wheelrouter_tag(lb)
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                sb = self.nametowidget(sb_path)
                # Note: on some Tk builds this is a ttk scrollbar; ignore if not tk.Scrollbar.
                if isinstance(sb, tk.Scrollbar):
                    sb.configure(
                        bg=p["BG"],
                        troughcolor=p["FIELD"],
                        activebackground=p["HOVER"],
                        highlightthickness=0,
                        bd=0,
                        relief="flat",
                    )
            except Exception:
                pass
        except Exception:
            pass


    # ----------------------------
    # Long OptionMenu dropdowns (replace native Tk Menu popdown for very long lists)
    # ----------------------------
    def _install_long_optionmenu_dropdowns(self):
        """Intercept very long ttk.OptionMenu/ttk.Menubutton popdowns (e.g. dinos/templates).

        Notes:
        - The menu is filled dynamically later, so we decide at click-time.
        - We can safely re-run this method: it will only install the class bindings once,
          but it will (re)tag any newly created Menubutton widgets.
        """
        p = getattr(self, "_PAL", None) or {}
        if not p:
            return

        if not hasattr(self, "_PRO_LD_TAG"):
            self._PRO_LD_TAG = "PCProLongDropdownMB"

        # Bind class once (priority is given by inserting the tag early in bindtags)
        if not getattr(self, "_pro_longdropdown_classbound", False):
            self._pro_longdropdown_classbound = True
            try:
                self.bind_class(self._PRO_LD_TAG, "<Button-1>", self._on_longdropdown_trigger, add="+")
                self.bind_class(self._PRO_LD_TAG, "<KeyPress-space>", self._on_longdropdown_trigger, add="+")
                self.bind_class(self._PRO_LD_TAG, "<Return>", self._on_longdropdown_trigger, add="+")
                self.bind_class(self._PRO_LD_TAG, "<Down>", self._on_longdropdown_trigger, add="+")
                self.bind_class(self._PRO_LD_TAG, "<Alt-Down>", self._on_longdropdown_trigger, add="+")
            except Exception:
                pass

        # Tag all menubuttons we can find (including ones created after init).
        try:
            for w in self._walk_widgets():
                if isinstance(w, (ttk.Menubutton, tk.Menubutton)):
                    self._ensure_longdropdown_tag(w)
        except Exception:
            pass

        # Also tag the known selector if present.
        try:
            cs = getattr(self, "canvas_selector", None)
            if isinstance(cs, (ttk.Menubutton, tk.Menubutton)):
                self._ensure_longdropdown_tag(cs)
        except Exception:
            pass
    def _get_menu_from_menubutton(self, mb):
        try:
            mname = mb.cget("menu")
        except Exception:
            return None
        if not mname:
            return None
        try:
            menu = self.nametowidget(mname) if isinstance(mname, str) else mname
        except Exception:
            menu = None
        return menu if isinstance(menu, tk.Menu) else None

    def _menu_labels_and_map(self, menu: tk.Menu):
        labels = []
        idx_map = []
        try:
            end = menu.index("end")
        except Exception:
            end = None
        if end is None:
            return labels, idx_map
        for i in range(int(end) + 1):
            try:
                t = menu.type(i)
                if t in ("separator", "tearoff"):
                    continue
                lab = menu.entrycget(i, "label")
                if not lab:
                    continue
                labels.append(lab)
                idx_map.append(i)
            except Exception:
                continue
        return labels, idx_map

    def _ensure_longdropdown_tag(self, mb):
        try:
            if getattr(mb, "_pro_ld_tagged", False):
                return
            mb._pro_ld_tagged = True
        except Exception:
            pass

        try:
            tags = list(mb.bindtags())
        except Exception:
            return

        # Insert our tag right after the widget tag so it fires before class bindings.
        try:
            if self._PRO_LD_TAG in tags:
                return
            if len(tags) >= 1:
                tags.insert(1, self._PRO_LD_TAG)
            else:
                tags = [self._PRO_LD_TAG]
            mb.bindtags(tuple(tags))
        except Exception:
            pass

    def _on_longdropdown_trigger(self, event):
        mb = getattr(event, "widget", None)
        if mb is None:
            return None

        # Only intercept if the current menu is long and template-like.
        try:
            if not self._should_use_long_dropdown(mb):
                return None  # let native behavior continue
        except Exception:
            return None

        try:
            self._open_long_dropdown(mb)
        except Exception:
            pass
        return "break"

    def _should_use_long_dropdown(self, mb):
        menu = self._get_menu_from_menubutton(mb)
        if menu is None:
            return False

        labels, _ = self._menu_labels_and_map(menu)
        if len(labels) < 30:
            return False

        # Heuristic: only for blueprint/template names (dinos/structures), avoid small pickers.
        try:
            sample = labels[:120]
            looks_like = any(("_Character_BP_C" in s) or ("StructureBP_" in s) or ("_BP_" in s) for s in sample)
        except Exception:
            looks_like = False
        if not looks_like:
            return False

        return True

    def _open_long_dropdown(self, mb: ttk.Menubutton, *, max_rows: int = 22):
        # Close previous, if any
        try:
            self._close_long_dropdown()
        except Exception:
            pass

        p = getattr(self, "_PAL", None) or {}
        menu = self._get_menu_from_menubutton(mb)
        if menu is None:
            return

        labels, idx_map = self._menu_labels_and_map(menu)
        if not labels:
            return

        try:
            self.update_idletasks()
        except Exception:
            pass

        # Geometry near the button
        try:
            x = int(mb.winfo_rootx())
            y = int(mb.winfo_rooty() + mb.winfo_height())
            w = int(mb.winfo_width()) if int(mb.winfo_width()) > 0 else 280
        except Exception:
            x, y, w = 100, 100, 280

        # Clamp to screen
        try:
            sw = int(self.winfo_screenwidth())
            sh = int(self.winfo_screenheight())
        except Exception:
            sw, sh = 1920, 1080

        rows = min(int(max_rows), len(labels))
        if rows < 6:
            rows = min(12, len(labels))

        top = tk.Toplevel(self)
        top.overrideredirect(True)
        top.configure(bg=p["FIELD"])
        try:
            top.attributes("-topmost", True)
        except Exception:
            pass

        # Content
        frame = tk.Frame(top, bg=p["FIELD"], highlightthickness=1, highlightbackground=p["BORDER"], bd=0)
        frame.pack(fill="both", expand=True)

        lb = tk.Listbox(
            frame,
            height=rows,
            bg=p["FIELD"],
            fg=p["FG"],
            selectbackground=p["ACCENT"],
            selectforeground="#ffffff",
            activestyle="none",
            highlightthickness=0,
            bd=0,
            relief="flat",
            exportselection=False,
        )
        sb = tk.Scrollbar(
            frame,
            orient="vertical",
            command=lb.yview,
            bg=p["BG"],
            troughcolor=p["FIELD"],
            activebackground=p["HOVER"],
            highlightthickness=0,
            bd=0,
            relief="flat",
        )
        lb.configure(yscrollcommand=sb.set)

        lb.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        # Fill
        for lab in labels:
            lb.insert("end", lab)

        # Preselect current value (from textvariable if present)
        cur_val = None
        try:
            tv = mb.cget("textvariable")
            if tv:
                try:
                    cur_val = self.getvar(tv)
                except Exception:
                    cur_val = None
        except Exception:
            pass
        if not cur_val:
            try:
                cur_val = mb.cget("text")
            except Exception:
                cur_val = None

        sel_i = 0
        if cur_val:
            try:
                sel_i = labels.index(cur_val)
            except ValueError:
                sel_i = 0

        try:
            lb.selection_clear(0, "end")
            lb.selection_set(sel_i)
            lb.activate(sel_i)
            lb.see(sel_i)
        except Exception:
            pass

        # Place after we know requested size
        try:
            top.update_idletasks()
            h_px = int(top.winfo_reqheight())
        except Exception:
            h_px = rows * 22 + 6

        if x + w > sw - 10:
            x = max(10, sw - w - 10)
        if y + h_px > sh - 10:
            y = max(10, int(mb.winfo_rooty()) - h_px)
        top.geometry(f"{w}x{h_px}+{x}+{y}")

        # Store refs
        self._pro_long_dropdown = top
        self._pro_long_dropdown_lb = lb
        self._pro_long_dropdown_menu = menu
        self._pro_long_dropdown_map = idx_map

        # Make it behave like a dropdown
        try:
            top.grab_set()
        except Exception:
            pass
        try:
            lb.focus_set()
        except Exception:
            pass

        # Wheel should scroll inside this dropdown
        try:
            lb._pro_is_popdown = True
            self._ensure_wheelrouter_tag(lb)
        except Exception:
            pass

        def _commit(_e=None):
            try:
                sel = lb.curselection()
                if not sel:
                    return
                li = int(sel[0])
                mi = int(idx_map[li])
                try:
                    menu.invoke(mi)
                except Exception:
                    pass
            finally:
                try:
                    self._close_long_dropdown()
                except Exception:
                    pass
            return "break"

        def _close(_e=None):
            try:
                self._close_long_dropdown()
            except Exception:
                pass
            return "break"

        # Basic bindings
        lb.bind("<Double-Button-1>", _commit)
        lb.bind("<Return>", _commit)
        lb.bind("<Escape>", _close)
        top.bind("<Escape>", _close)

        # Click selection commits on release
        lb.bind("<ButtonRelease-1>", _commit)

        # Close when losing focus
        top.bind("<FocusOut>", _close)

        # Type-to-search
        self._pro_ld_typebuf = ""
        self._pro_ld_typeafter = None

        def _clear_typebuf():
            self._pro_ld_typebuf = ""
            self._pro_ld_typeafter = None

        def _on_key(e):
            ch = getattr(e, "char", "")
            if not ch or not ch.isprintable() or ch.isspace():
                return
            self._pro_ld_typebuf += ch.lower()
            # reset timer
            try:
                if self._pro_ld_typeafter is not None:
                    self.after_cancel(self._pro_ld_typeafter)
            except Exception:
                pass
            try:
                self._pro_ld_typeafter = self.after(700, _clear_typebuf)
            except Exception:
                pass

            prefix = self._pro_ld_typebuf
            # start from current selection + 1 for cycling
            try:
                start = int(lb.curselection()[0]) + 1 if lb.curselection() else 0
            except Exception:
                start = 0
            n = len(labels)
            for k in range(n):
                i = (start + k) % n
                if labels[i].lower().startswith(prefix):
                    try:
                        lb.selection_clear(0, "end")
                        lb.selection_set(i)
                        lb.activate(i)
                        lb.see(i)
                    except Exception:
                        pass
                    break

        lb.bind("<KeyPress>", _on_key, add="+")

    def _close_long_dropdown(self):
        top = getattr(self, "_pro_long_dropdown", None)
        if top is None:
            return
        try:
            top.grab_release()
        except Exception:
            pass
        try:
            top.destroy()
        except Exception:
            pass
        self._pro_long_dropdown = None
        self._pro_long_dropdown_lb = None
        self._pro_long_dropdown_menu = None
        self._pro_long_dropdown_map = None
    # ----------------------------
    # MouseWheel routing (prefer scrolling the main block over "value scrolling" in selectors)
    # ----------------------------
    def _install_wheel_router(self):
        if getattr(self, "_wheel_router_installed", False):
            return
        self._wheel_router_installed = True

        self._WHEEL_TAG = "PCWheelRouter"

        # Bind router tag (we will inject it as the first bindtag for most widgets)
        try:
            self.bind_class(self._WHEEL_TAG, "<MouseWheel>", self._on_mousewheel, add="+")
        except Exception:
            pass
        # Linux / X11 wheels
        try:
            self.bind_class(self._WHEEL_TAG, "<Button-4>", self._on_mousewheel_linux, add="+")
            self.bind_class(self._WHEEL_TAG, "<Button-5>", self._on_mousewheel_linux, add="+")
        except Exception:
            pass

        self._main_scroll_canvas = self._discover_main_scroll_canvas()
        self._refresh_wheelrouter_tags()

    def _refresh_wheelrouter_tags(self):
        try:
            self._ensure_wheelrouter_tag(self)
        except Exception:
            pass
        try:
            for w in self._walk_widgets():
                self._ensure_wheelrouter_tag(w)
        except Exception:
            pass

    def _ensure_wheelrouter_tag(self, w):
        tag = getattr(self, "_WHEEL_TAG", None)
        if not tag:
            return
        try:
            tags = list(w.bindtags())
        except Exception:
            return
        if tag in tags:
            return
        # Run BEFORE widget/class default handlers so we can route the wheel consistently.
        tags.insert(0, tag)
        try:
            w.bindtags(tuple(tags))
        except Exception:
            pass

    def _discover_main_scroll_canvas(self):
        # Prefer BasePreviewGUI-provided control canvas (this is the "main block" in your layout)
        c = getattr(self, "_controls_canvas", None)
        if isinstance(c, tk.Canvas):
            return c

        # Fallback: choose the largest scrollable Canvas in the widget tree.
        best = None
        best_area = -1
        try:
            for w in self._walk_widgets():
                if isinstance(w, tk.Canvas):
                    try:
                        ysc = str(w.cget("yscrollcommand") or "")
                    except Exception:
                        ysc = ""
                    if not ysc:
                        continue
                    try:
                        area = int(w.winfo_width()) * int(w.winfo_height())
                    except Exception:
                        area = 0
                    if area > best_area:
                        best = w
                        best_area = area
        except Exception:
            pass
        return best

    def _on_mousewheel_linux(self, event):
        # Convert Button-4/5 to a delta-like value and reuse the main handler.
        try:
            num = int(getattr(event, "num", 0))
            delta = 120 if num == 4 else -120 if num == 5 else 0
            event.delta = delta
        except Exception:
            pass
        return self._on_mousewheel(event)

    def _on_mousewheel(self, event):
        # Route wheel:
        # - If a Combobox popdown list is open under the pointer, scroll the list.
        # - Else, scroll the nearest scrollable container (usually the main controls canvas).
        try:
            x = int(getattr(event, "x_root", 0))
            y = int(getattr(event, "y_root", 0))
            w = self.winfo_containing(x, y)
        except Exception:
            w = None

        if w is None:
            return None

        # Combobox popdown listbox (open dropdown)
        try:
            if isinstance(w, tk.Listbox) and getattr(w, "_pro_is_popdown", False):
                return self._scroll_widget(w, event)
        except Exception:
            pass

        # Don't let wheel "change values" in selectors when dropdown is closed.
        try:
            if self._is_selector_widget(w):
                w = getattr(w, "master", w)
        except Exception:
            pass

        target = self._find_scroll_target(w)
        if target is None:
            target = getattr(self, "_main_scroll_canvas", None)

        if target is None:
            return None

        return self._scroll_widget(target, event)

    def _is_selector_widget(self, w):
        # Widgets where the default wheel behavior changes a value/selection
        try:
            if isinstance(w, (ttk.Combobox, ttk.Spinbox)):
                return True
        except Exception:
            pass
        try:
            if isinstance(w, (tk.Spinbox, tk.Scale)):
                return True
        except Exception:
            pass
        try:
            if isinstance(w, ttk.Scale):
                return True
        except Exception:
            pass
        try:
            if isinstance(w, tk.Menubutton) and not isinstance(w, ttk.Menubutton):
                return True
        except Exception:
            pass
        return False

    def _find_scroll_target(self, w):
        # Prefer scrolling:
        # - popdown listboxes (handled earlier)
        # - true scrollable widgets (Text/Listbox/Treeview) if they are not tiny embedded pickers,
        # - otherwise walk up to find the nearest scrollable Canvas container.
        cur = w
        best_canvas = None
        for _ in range(50):
            if cur is None:
                break

            # Avoid capturing wheel on tiny decorative canvases / swatches
            if self._is_scrollable_widget(cur):
                # For Listbox: only allow if it's a real list under the pointer (not the dyes grid items).
                try:
                    if isinstance(cur, tk.Listbox) and not getattr(cur, "_pro_is_popdown", False):
                        # If listbox has no yscrollcommand, it's usually not meant to scroll.
                        try:
                            if not str(cur.cget("yscrollcommand") or ""):
                                pass
                            else:
                                return cur
                        except Exception:
                            pass
                    else:
                        return cur
                except Exception:
                    return cur

            # Track best scrollable canvas ancestor (yscrollcommand set)
            try:
                if isinstance(cur, tk.Canvas):
                    ysc = str(cur.cget("yscrollcommand") or "")
                    if ysc:
                        best_canvas = cur
            except Exception:
                pass

            try:
                cur = cur.master
            except Exception:
                break

        return best_canvas
    def _is_scrollable_widget(self, w):
        # Common scrollable widgets
        try:
            if isinstance(w, (tk.Text, tk.Listbox, ttk.Treeview)):
                return True
        except Exception:
            pass

        # Canvases are only scrollable if they are configured for scrolling,
        # otherwise we risk "scrolling" tiny paint swatches or decorative canvases.
        if isinstance(w, tk.Canvas):
            try:
                ysc = str(w.cget("yscrollcommand") or "")
            except Exception:
                ysc = ""
            if ysc:
                return True

            # Some scrollable canvases don't set yscrollcommand but do set a scrollregion larger than the viewport.
            try:
                sr = str(w.cget("scrollregion") or "")
                if sr:
                    parts = [float(x) for x in sr.replace(",", " ").split()]
                    if len(parts) == 4:
                        x0, y0, x1, y1 = parts
                        region_h = max(0.0, y1 - y0)
                        try:
                            vh = float(max(1, int(w.winfo_height())))
                        except Exception:
                            vh = 1.0
                        # Only treat it as scrollable if region is meaningfully taller than viewport
                        if region_h > vh + 8:
                            return True
            except Exception:
                pass

            return False

        return False
    def _wheel_units(self, event):
        try:
            delta = int(getattr(event, "delta", 0))
        except Exception:
            delta = 0
        if delta == 0:
            return 0

        # Windows delta is usually 120 per notch; positive = up.
        sign = -1 if delta > 0 else 1
        steps = int(abs(delta) / 120) if abs(delta) >= 120 else 1
        if steps > 3:
            steps = 3
        return sign * steps

    def _scroll_widget(self, w, event):
        units = self._wheel_units(event)
        if units == 0:
            return "break"
        try:
            w.yview_scroll(units, "units")
            return "break"
        except Exception:
            # Some widgets use the yview command instead
            try:
                w.yview("scroll", units, "units")
            except Exception:
                pass
            return "break"

    # ----------------------------
    # Status (uses BasePreviewGUI.gen_status label)
    # ----------------------------
    def _set_status(self, text: str, kind: str = "info"):
        try:
            if not hasattr(self, "gen_status") or self.gen_status is None:
                return
            fg = self._PAL.get("MUTED", "#a9b0bb")
            if kind == "warn":
                fg = "#ffb86b"
            elif kind == "err":
                fg = "#ff6b6b"
            self.gen_status.configure(text=text, foreground=fg)
        except Exception:
            pass

    # ----------------------------
    # Icons for ttk.Button (image + compound)
    # ----------------------------
    def _ico(self, filename: str, size: int = 16):
        key = (filename, size)
        if key in self._ico_cache:
            return self._ico_cache[key]
        try:
            img = Image.open(relative_to_assets(filename)).convert("RGBA")
            if size:
                try:
                    resample = Image.Resampling.LANCZOS
                except Exception:
                    resample = Image.LANCZOS
                img = img.resize((size, size), resample=resample)
            tkimg = ImageTk.PhotoImage(img)
            self._ico_cache[key] = tkimg
            return tkimg
        except Exception:
            return None

    def _promote_primary_buttons(self):
        # Ensure Generate uses primary style (by msgid)
        for w in self._walk_widgets():
            if isinstance(w, ttk.Button) and getattr(w, "_msgid", None) == "btn.generate":
                try:
                    w.configure(style="ProPrimaryShortcut.TButton")
                except Exception:
                    pass

    def _attach_button_icons(self):
        icon_by_msgid = {
            "btn.open_image": "ico_open.png",
            "btn.generate": "ico_generate.png",
            "btn.calculate": "ico_calc.png",

            "btn.load": "ico_load.png",
            "btn.activate_visibles": "ico_eye.png",
            "btn.deactivate_visibles": "ico_eye_off.png",
            "btn.browse": "ico_folder.png",
            "btn.scan": "ico_scan.png",
            "btn.copy_bp_size": "ico_copy.png",
            "btn.use_for_generate": "ico_use.png",
            "btn.apply": "ico_apply.png",
        }
        self._btn_by_msgid = {}
        for w in self._walk_widgets():
            if isinstance(w, ttk.Button):
                msgid = getattr(w, "_msgid", None)
                if msgid:
                    self._btn_by_msgid[msgid] = w
                ico_file = icon_by_msgid.get(msgid)
                if ico_file:
                    ico = self._ico(ico_file, 16)
                    if ico is not None:
                        try:
                            w.configure(image=ico, compound="left")
                        except Exception:
                            pass

    def _update_shortcut_labels(self):
        # Put Ctrl+O / Ctrl+G into existing left buttons
        try:
            self._attach_button_icons()
        except Exception:
            pass

        def _set_btn(msgid, text, style=None):
            btn = getattr(self, "_btn_by_msgid", {}).get(msgid)
            if btn is None:
                return
            try:
                btn.configure(text=text)
            except Exception:
                pass
            if style:
                try:
                    btn.configure(style=style)
                except Exception:
                    pass

        open_label = f"{self.t('btn.open_image')}  (Ctrl+O)"
        gen_label = f"{self.t('btn.generate')}  (Ctrl+G)"

        _set_btn("btn.open_image", open_label, style="ProShortcut.TButton")
        _set_btn("btn.generate", gen_label, style="ProPrimaryShortcut.TButton")
    def _on_toggle_advanced(self):
        # When AdvancedPanel is created/shown, new widgets appear in a Toplevel.
        # Re-apply our ttk styles + dark surfaces + icons after BasePreviewGUI builds
        # and applies language bindings.
        try:
            super()._on_toggle_advanced()
        finally:
            def _refresh_adv():
                try:
                    # Apply custom ttk styles (buttons/menubuttons/checkbuttons)
                    self._apply_custom_styles_to_widgets()
                except Exception:
                    pass
                try:
                    self._tune_dark_surfaces()
                except Exception:
                    pass
                try:
                    # Advanced widgets get msgids during Base _ensure_advanced_window()
                    self._attach_button_icons()
                    self._promote_primary_buttons()
                except Exception:
                    pass
                try:
                    self._tune_action_button_sizes()
                except Exception:
                    pass
                # Ensure window chrome/icon for Advanced toplevel
                try:
                    win = getattr(self, "_advanced_win", None)
                    if win is not None and bool(int(win.winfo_exists())):
                        self._set_window_icon(win)
                        self._apply_windows_dark_titlebar(win)
                except Exception:
                    pass

            # Run a few times to catch late-mapped widgets (Tk can create/mount some elements after idle)
            try:
                self.after_idle(_refresh_adv)
                self.after(40, _refresh_adv)
                self.after(140, _refresh_adv)
            except Exception:
                pass


    def _button_required_chars(self, btn, extra_px: int = 0) -> int:
        """Conservative ttk.Button width (chars) so text+icon never clips."""
        try:
            import tkinter.font as tkfont
            style = ttk.Style(self)
            st = (btn.cget("style") or "").strip() or "TButton"
            fdesc = style.lookup(st, "font") or btn.cget("font") or "TkDefaultFont"
            f = tkfont.Font(root=self, font=fdesc)

            text = str(btn.cget("text") or "")
            px = f.measure(text)

            # Icon allowance when compound is left/right
            try:
                img = str(btn.cget("image") or "")
                comp = str(btn.cget("compound") or "")
                if img and comp in ("left", "right"):
                    px += 22
            except Exception:
                pass

            px += 16 + int(extra_px)

            avg = f.measure("0") or 7
            avg = max(7, int(avg))
            chars = int(math.ceil(px / avg))
            return max(chars, 2)
        except Exception:
            try:
                txt = str(btn.cget("text") or "")
                add = 3 if str(btn.cget("image") or "") else 0
                return max(2, len(txt) + add)
            except Exception:
                return 8

    def _tune_action_button_sizes(self):
        """Make key buttons smaller and avoid clipped labels (Bug-style, no layout forcing)."""
        try:
            if not hasattr(self, "_btn_by_msgid"):
                self._attach_button_icons()

            btns = getattr(self, "_btn_by_msgid", {}) or {}

            groups = [
                ["btn.activate_visibles", "btn.deactivate_visibles"],
                ["btn.scan", "btn.copy_bp_size", "btn.use_for_generate", "btn.apply"],
            ]

            for ids in groups:
                ws = [btns.get(i) for i in ids]
                ws = [b for b in ws if b is not None]
                if not ws:
                    continue
                w = max(self._button_required_chars(b) for b in ws)
                for b in ws:
                    try:
                        b.configure(width=w)
                    except Exception:
                        pass

            for mid in ["btn.calculate", "btn.load", "btn.browse"]:
                b = btns.get(mid)
                if b is None:
                    continue
                try:
                    b.configure(width=self._button_required_chars(b))
                except Exception:
                    pass
        except Exception:
            pass

    def _equalize_visibles_buttons(self):
        # Backwards compatibility for old call sites
        try:
            self._tune_action_button_sizes()
        except Exception:
            pass




    # ----------------------------
    # settings application
    # ----------------------------
    def _apply_settings_early(self):
        s = getattr(self, "_settings", {}) or {}

        # Window geometry (only if enabled)
        try:
            w = (s.get("window") or {})
            if w.get("remember_geometry") and w.get("geometry"):
                self.geometry(w.get("geometry"))
        except Exception:
            pass

        # Language override
        try:
            lang = s.get("language", "auto")
            if lang and lang != "auto":
                try:
                    self.set_language(lang)
                except Exception:
                    pass
        except Exception:
            pass

        # Translation overrides
        try:
            self._apply_user_translation_overrides()
        except Exception:
            pass

        # Defaults (non-destructive: null means do not force)
        try:
            self._apply_settings_defaults()
        except Exception:
            pass


    def _apply_settings_late(self):
        # Advanced & late-created widgets
        try:
            self._apply_settings_advanced()
        except Exception:
            pass


    def _apply_user_translation_overrides(self):
        s = getattr(self, "_settings", {}) or {}
        rel = s.get("user_translation_overrides")
        if not rel or not isinstance(rel, str):
            return

        base_dir = getattr(self, "_settings_dir", OUTPUT_PATH)
        p = Path(rel)
        if not p.is_absolute():
            p = Path(base_dir) / rel
        if not p.exists():
            return

        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return

        i18n = getattr(self, "_i18n", None)
        if i18n is None or not hasattr(i18n, "translations"):
            return

        # Support two shapes:
        # 1) {"msgid":"Text"} (applies to current lang)
        # 2) {"es":{...}, "en":{...}}
        if isinstance(raw, dict) and all(isinstance(k, str) for k in raw.keys()):
            if any(k in ("es", "en", "zh", "ru") and isinstance(raw.get(k), dict) for k in raw.keys()):
                for lang, d in raw.items():
                    if lang in i18n.translations and isinstance(d, dict):
                        try:
                            i18n.translations[lang].update({str(k): str(v) for k, v in d.items()})
                        except Exception:
                            pass
            else:
                cur = getattr(i18n, "lang", "es")
                if cur in i18n.translations:
                    try:
                        i18n.translations[cur].update({str(k): str(v) for k, v in raw.items()})
                    except Exception:
                        pass

        # Re-apply language to refresh widget text
        try:
            self._apply_language()
        except Exception:
            pass


    def _apply_settings_defaults(self):
        s = getattr(self, "_settings", {}) or {}
        d = s.get("defaults") or {}

        # preview_mode
        pm = d.get("preview_mode")
        if pm and hasattr(self, "_set_preview_mode_key"):
            try:
                self._set_preview_mode_key(pm)
            except Exception:
                pass

        # show_game_object
        v = d.get("show_game_object")
        if v is not None and hasattr(self, "show_object_var"):
            try:
                self.show_object_var.set(bool(v))
            except Exception:
                pass

        # use_all_dyes
        v = d.get("use_all_dyes")
        if v is not None and hasattr(self, "use_all_dyes_var"):
            try:
                self.use_all_dyes_var.set(bool(v))
            except Exception:
                pass

        # best_colors
        v = d.get("best_colors")
        if v is not None and hasattr(self, "best_dyes_var"):
            try:
                self.best_dyes_var.set(int(v))
            except Exception:
                pass

        # border_style
        v = d.get("border_style")
        if v is not None and hasattr(self, "border_style"):
            try:
                self.border_style.set(v)
            except Exception:
                pass

        # dither_mode
        v = d.get("dither_mode")
        if v is not None and hasattr(self, "dither_mode"):
            try:
                self.dither_mode.set(v)
            except Exception:
                pass

        # show_advanced handled in late apply


    def _apply_settings_advanced(self):
        s = getattr(self, "_settings", {}) or {}
        d = (s.get("defaults") or {})
        ad = (s.get("advanced_defaults") or {})

        # show advanced
        v = d.get("show_advanced")
        if v is not None and hasattr(self, "show_advanced_var"):
            try:
                self.show_advanced_var.set(bool(v))
                if hasattr(self, "_on_toggle_advanced"):
                    self._on_toggle_advanced()
            except Exception:
                pass

        # Apply advanced vars if they exist
        mapping = {
            "external_enabled": "external_enabled_var",
            "external_recursive": "external_recursive_var",
            "external_detect_guid": "external_detect_guid_var",
            "external_max_files": "external_max_files_var",
            "external_preserve_metadata": "external_preserve_var",
            "show_dino_tools": "show_dino_tools_var",
        }
        for key, attr in mapping.items():
            val = ad.get(key)
            if val is None:
                continue
            if hasattr(self, attr):
                try:
                    getattr(self, attr).set(val)
                except Exception:
                    pass

        # Re-apply styling/icons for late created widgets (Advanced Toplevel)
        try:
            self.after_idle(self._apply_custom_styles_to_widgets)
            self.after_idle(self._attach_button_icons)
            self.after_idle(self._tune_action_button_sizes)
        except Exception:
            pass


    def _install_settings_persistence(self):
        # Wrap BasePreviewGUI close handler to persist geometry (if enabled)
        if getattr(self, "_settings_persistence_installed", False):
            return
        self._settings_persistence_installed = True

        try:
            orig = getattr(self, "_on_main_close", None) or getattr(self, "_on_close", None)
            if orig is None:
                return

            def _wrapped_close(*a, **kw):
                try:
                    self._save_geometry_if_enabled()
                except Exception:
                    pass
                return orig(*a, **kw)

            # Keep name used by protocol
            if hasattr(self, "_on_main_close"):
                self._on_main_close = _wrapped_close
            else:
                self._on_close = _wrapped_close

            try:
                self.protocol("WM_DELETE_WINDOW", _wrapped_close)
            except Exception:
                pass
        except Exception:
            pass


    def _save_geometry_if_enabled(self):
        s = getattr(self, "_settings", {}) or {}
        path = getattr(self, "_settings_path", None)
        if not path:
            return
        w = s.get("window") or {}
        if not w.get("remember_geometry"):
            return
        try:
            w["geometry"] = self.winfo_geometry()
        except Exception:
            return
        s["window"] = w
        try:
            _write_json_atomic(Path(path), s)
        except Exception:
            pass
    def _bind_shortcuts(self):
        """Global shortcuts (Ctrl+O open, Ctrl+G generate)."""
        try:
            self.bind_all('<Control-o>', lambda e: self._open_image())
            self.bind_all('<Control-O>', lambda e: self._open_image())
            self.bind_all('<Control-g>', lambda e: self._on_generate())
            self.bind_all('<Control-G>', lambda e: self._on_generate())
        except Exception:
            pass

    def _wrap_actions_for_status(self):
        """Lightweight status messages around Open/Generate without changing BasePreviewGUI."""
        try:
            if getattr(self, '_wrapped_status', False):
                return
            self._wrapped_status = True

            _orig_open = self._open_image
            _orig_gen = self._on_generate

            def open_wrap(*a, **kw):
                try:
                    self._set_status(self.t('status.opening_image'), kind='info')
                except Exception:
                    pass
                try:
                    return _orig_open(*a, **kw)
                finally:
                    try:
                        self.after(200, lambda: self._set_status('', kind='info'))
                    except Exception:
                        pass

            def gen_wrap(*a, **kw):
                try:
                    self._set_status(self.t('status.generating_pnt'), kind='info')
                except Exception:
                    pass
                try:
                    return _orig_gen(*a, **kw)
                finally:
                    try:
                        self.after(300, lambda: self._set_status('', kind='info'))
                    except Exception:
                        pass

            self._open_image = open_wrap
            self._on_generate = gen_wrap
        except Exception:
            pass

if __name__ == "__main__":
    PreviewGUI().mainloop()
