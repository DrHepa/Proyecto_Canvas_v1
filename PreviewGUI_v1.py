import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
import threading
from pathlib import Path
from PreviewController_v2 import PreviewController
from paths import get_app_root
import json
import queue

from ExternalPntLibrary_v1 import scan_pnts
from MaskExtractor import save_user_mask_pack, refine_user_mask_pack_existing


# ---------------------------
# Default folders (Windows ASA)
# ---------------------------
_DEFAULT_MYPAINTINGS_WIN = r"C:\Program Files (x86)\Steam\steamapps\common\ARK Survival Ascended\ShooterGame\Saved\MyPaintings"


def _pick_existing_dir(*candidates: Path) -> Path:
    for c in candidates:
        try:
            if c and c.exists() and c.is_dir():
                return c
        except Exception:
            continue
    return Path.home()


def default_mypaintings_dir() -> Path:
    """Best-effort default for ASA MyPaintings. Deterministic, no GUI heuristics."""
    # Primary known Steam install path
    p = Path(_DEFAULT_MYPAINTINGS_WIN)
    if os.name == "nt" and p.exists() and p.is_dir():
        return p

    # Secondary: derive from PROGRAMFILES(X86) or PROGRAMFILES
    base = os.environ.get("PROGRAMFILES(X86)") or os.environ.get("PROGRAMFILES")
    if base:
        cand = Path(base) / "Steam" / "steamapps" / "common" / "ARK Survival Ascended" / "ShooterGame" / "Saved" / "MyPaintings"
        if cand.exists() and cand.is_dir():
            return cand

    # Fallback to home (avoids broken initialdir on systems without ASA)
    return Path.home()



# ---------------------------
# i18n (multi-language UI)
# ---------------------------
class _I18n:
    def __init__(self, locales_dir: Path, default_lang: str = "es", strict: bool = True):
        self.locales_dir = Path(locales_dir)
        self.supported = ["es", "en", "zh", "ru"]
        self.translations = {}
        for lang in self.supported:
            p = self.locales_dir / f"{lang}.json"
            if not p.exists():
                raise FileNotFoundError(f"[i18n] Missing locale file: {p}")
            self.translations[lang] = json.loads(p.read_text(encoding="utf-8"))

        if strict:
            keys0 = set(self.translations[self.supported[0]].keys())
            for lang in self.supported[1:]:
                k = set(self.translations[lang].keys())
                if k != keys0:
                    missing = sorted(keys0 - k)
                    extra = sorted(k - keys0)
                    raise ValueError(f"[i18n] Key mismatch for '{lang}'. Missing={missing[:10]} Extra={extra[:10]}")

        self.lang = default_lang if default_lang in self.supported else "es"

    def set_lang(self, lang: str):
        if lang in self.supported:
            self.lang = lang

    def t(self, key: str, **kwargs) -> str:
        d = self.translations.get(self.lang) or self.translations["es"]
        s = d.get(key, key)
        try:
            return s.format(**kwargs)
        except Exception:
            return s

# Base text -> msgid mapping (used to bind existing widgets to translation keys)
_I18N_TEXT_TO_KEY = {
    "Proyecto Canvas – Preview": "app.title",
    "Imagen": "panel.image",
    "Abrir imagen… Ctrol+O": "btn.open_image",
    "(ninguna)": "label.none",
    "Canvas": "panel.canvas",
    "Paint Area": "panel.paint_area",
    "Preview Mode": "panel.preview_mode",
    "Mostrar objeto en el juego": "chk.show_game_object",
    "Dyes (Generación)": "panel.dyes",
    "Usar todos los dyes": "chk.use_all_dyes",
    "Best colors:": "label.best_colors",
    "Calcular": "btn.calculate",
    "Desactivar visibles": "btn.deactivate_visibles",
    "Activar visibles": "btn.activate_visibles",
    "Border": "panel.border",
    "(sin frame)": "label.no_frame",
    "Cargar…": "btn.load",
    "Dithering": "panel.dithering",
    "Advanced": "panel.advanced",
    "Mostrar advanced": "chk.show_advanced",
    "Generar .PNT Ctrol+G": "btn.generate",
    "Scanning…": "status.scanning",
    "External .pnt": "panel.external_pnt",
    "Activar External .pnt (advanced)": "chk.enable_external",
    "Carpeta:": "label.folder",
    "Browse…": "btn.browse",
    "Recursive": "chk.recursive",
    "Detect GUID": "chk.detect_guid",
    "Max:": "label.max",
    "Use for Generate": "btn.use_for_generate",
    "Copy BP/Size": "btn.copy_bp_size",
    "Hint Y:": "label.hint_y",
    "Preserve metadata (GUID/suffix)": "chk.preserve_metadata",
    "Archivo": "column.file",
    "Tipo": "column.type",
    "Blueprint": "column.blueprint",
    "Size": "column.size",
    "Apply": "btn.apply",
    "Multi-Canvas Grid": "panel.multicanvas_grid",
    "Rows:": "label.rows",
    "Cols:": "label.cols",
}


class PreviewGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        # ---------------------------------
        # Definir Paths
        # ---------------------------------       
        
        app_root = get_app_root()
        self.templates_root = app_root / "Templates"
        self.tabla_dyes_path = get_app_root() / "TablaDyes_v1.json"
        self._user_cache_path = app_root / "user_cache.json"
        self._default_mypaintings_dir = default_mypaintings_dir()
        self._default_scan_dir = self._default_mypaintings_dir
        self._default_generate_dir = self._default_mypaintings_dir
        self._default_border_dir = self.templates_root / "TiableBorder"


        # ---------------------------------
        # Inicializar variables de cache
        # ---------------------------------
        self._last_open_image_dir = None
        self._last_generate_dir = None
        self._last_scan_dir = None
        self._last_border_dir = None

        # Cargar Cache persistente
        self._load_user_cache()
        # i18n
        self._locales_dir = app_root / "locales"
        preferred_lang = None
        try:
            preferred_lang = (getattr(self, "_user_cache", {}) or {}).get("ui_lang")
        except Exception:
            preferred_lang = None
        self._i18n = _I18n(self._locales_dir, default_lang=(preferred_lang or "es"))
        self._i18n_bindings = []
        self._i18n_treeviews = []
        self._save_user_cache()

        # ---------------------------------
        # Controller (ÚNICO punto de entrada)
        # ---------------------------------
        self.controller = PreviewController(
            templates_root=self.templates_root
        )
        self._templates_by_category = self._group_templates_by_category()
        
        # ---------------------------------
        # Estado interno GUI (AQUÍ)
        # ---------------------------------
        self.dye_vars = {}
        self.dye_swatches = {}
        self._redraw_job = None
        self._is_generating = False
        self._photo = None
        self._render_native = None
        self._draw_job = None
        self._redraw_delay_ms = 45

        # ---------------------------------
        # Async Preview (background render)
        # ---------------------------------
        self._async_preview_enabled = True

        self._preview_req_q = queue.Queue()
        self._preview_res_q = queue.Queue()

        self._preview_seq = 0
        self._preview_last_applied_seq = 0
        self._preview_poll_job = None

        self._preview_thread = threading.Thread(
            target=self._preview_worker_loop,
            name="PreviewRenderWorker",
            daemon=True,
        )
        self._preview_thread.start()
        # ---------------------------------
        # Async Generation (background export)
        # ---------------------------------
        self._gen_req_q = queue.Queue()
        self._gen_res_q = queue.Queue()
        self._gen_job_seq = 0
        self._gen_active_job_id = 0
        self._gen_modal = None
        self._gen_progress = None
        self._gen_poll_job = None

        self._gen_thread = threading.Thread(
            target=self._gen_worker_loop,
            name="PntGenerationWorker",
            daemon=True,
        )
        self._gen_thread.start()

        # ---------------------------------
        # Advanced: external .pnt scanning (MyPaintings / LocalSaved)
        # ---------------------------------
        self._ext_items = []
        self._ext_scan_res_q = queue.Queue()
        self._ext_scan_seq = 0
        self._ext_scan_poll_job = None
        self._external_prev_selection = None  # (category, template, writer_mode)

        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # ---------------------------------
        # Window
        # ---------------------------------
        self.title(self.t("app.title"))
        self._is_closing = False
        self._is_closing = False
        self.geometry("1100x800")
        self.minsize(900, 600)

        # ---------------------------------
        # Layout
        # ---------------------------------
        self._build_layout()
        self._build_controls()
        self._build_preview()
        self._init_tooltip()
        self._i18n_scan_and_bind()
        self._apply_language()

        self.preview_canvas.bind("<Configure>", lambda e: self._schedule_draw())

        # Keep Advanced window docked to the main window
        self._adv_pos_scheduled = False
        try:
            self.bind("<Configure>", self._on_main_configure)
        except Exception:
            pass

        # One-click close (also destroys Advanced if open)
        try:
            self.protocol("WM_DELETE_WINDOW", self._on_main_close)
        except Exception:
            pass

    def _init_tooltip(self):
        self._tooltip = tk.Toplevel(self)
        self._tooltip.withdraw()
        self._tooltip.overrideredirect(True)
        self._tooltip.attributes("-topmost", True)

        self._tooltip_label = ttk.Label(
            self._tooltip,
            text="",
            background="#222",
            foreground="#fff",
            padding=(6, 3)
        )
        self._tooltip_label.pack()

    def _on_close(self):
        try:
            self._preview_req_q.put(None)  # sentinel
            self._gen_req_q.put(None)  # sentinel
        except Exception:
            pass
        self.destroy()

        
    # ---------------------------------
    # Load and Save user Cache
    # ---------------------------------
    def _load_user_cache(self):
        """
        Carga el cache persistente de usuario (solo GUI).
        No lanza excepciones.
        """
        if not self._user_cache_path.exists():
            self._user_cache = {}
            # Defaults (scan + generate) -> MyPaintings
            try:
                if self._last_generate_dir is None:
                    self._last_generate_dir = _pick_existing_dir(self._default_generate_dir, self.templates_root, Path.home())
                if self._last_scan_dir is None:
                    self._last_scan_dir = _pick_existing_dir(self._default_scan_dir, self._last_generate_dir or Path.home(), Path.home())
                if self._last_border_dir is None:
                    self._last_border_dir = _pick_existing_dir(self._default_border_dir, self.templates_root, Path.home())
            except Exception:
                pass
            return

        try:
            with open(self._user_cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._user_cache = data

            open_dir = data.get("last_open_image_dir")
            gen_dir = data.get("last_generate_dir")
            scan_dir = data.get("last_scan_dir")
            border_dir = data.get("last_border_dir")

            if open_dir:
                p = Path(open_dir)
                if p.exists() and p.is_dir():
                        self._last_open_image_dir = p

            if gen_dir:
                 p = Path(gen_dir)
                 if p.exists() and p.is_dir():
                        self._last_generate_dir = p

            if scan_dir:
                 p = Path(scan_dir)
                 if p.exists() and p.is_dir():
                        self._last_scan_dir = p

            if border_dir:
                 p = Path(border_dir)
                 if p.exists() and p.is_dir():
                        self._last_border_dir = p

            # Defaults (scan + generate) -> MyPaintings
            if self._last_generate_dir is None:
                self._last_generate_dir = _pick_existing_dir(self._default_generate_dir, self.templates_root, Path.home())

            if self._last_scan_dir is None:
                self._last_scan_dir = _pick_existing_dir(self._default_scan_dir, self._last_generate_dir or Path.home(), Path.home())

            # Default for borders -> project Templates/TiableBorder
            if self._last_border_dir is None:
                self._last_border_dir = _pick_existing_dir(self._default_border_dir, self.templates_root, Path.home())
        except Exception:

            # Cache corrupto o ilegible → se ignora silenciosamente
            self._user_cache = {}
            try:
                if self._last_generate_dir is None:
                    self._last_generate_dir = _pick_existing_dir(self._default_generate_dir, self.templates_root, Path.home())
                if self._last_scan_dir is None:
                    self._last_scan_dir = _pick_existing_dir(self._default_scan_dir, self._last_generate_dir or Path.home(), Path.home())
                if self._last_border_dir is None:
                    self._last_border_dir = _pick_existing_dir(self._default_border_dir, self.templates_root, Path.home())
            except Exception:
                pass
            pass

    def _save_user_cache(self):
        """
        Guarda el cache persistente de usuario (solo GUI).
        """
        if not hasattr(self, "_user_cache") or self._user_cache is None:
            self._user_cache = {}

        data = dict(self._user_cache)
        data.update({
            "last_open_image_dir": (
                str(self._last_open_image_dir)
                if self._last_open_image_dir else None
            ),
            "last_generate_dir": (
                str(self._last_generate_dir)
                if self._last_generate_dir else None
            ),
            "last_scan_dir": (
                str(self._last_scan_dir)
                if getattr(self, "_last_scan_dir", None) else None
            ),
            "last_border_dir": (
                str(self._last_border_dir)
                if getattr(self, "_last_border_dir", None) else None
            ),
            "ui_lang": getattr(getattr(self, "_i18n", None), "lang", data.get("ui_lang", "es")),
        })

        self._user_cache = data

        try:
             with open(self._user_cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            # Fallo de escritura → no bloquea la app
            pass
        

    # ======================================================
    # i18n helpers (GUI-level)
    # ======================================================
    def t(self, key: str, **kwargs) -> str:
        try:
            return self._i18n.t(key, **kwargs)
        except Exception:
            return key

    def _i18n_bind(self, widget, msgid: str, option: str = "text"):
        try:
            setattr(widget, "_msgid", msgid)
        except Exception:
            pass
        self._i18n_bindings.append((widget, option, msgid))

    def _i18n_walk(self):
        stack = [self]
        seen = set()
        while stack:
            w = stack.pop()
            wid = str(w)
            if wid in seen:
                continue
            seen.add(wid)
            yield w
            try:
                stack.extend(w.winfo_children())
            except Exception:
                pass

    def _i18n_scan_and_bind(self):
        for w in self._i18n_walk():
            if isinstance(w, ttk.Treeview) and w not in self._i18n_treeviews:
                self._i18n_treeviews.append(w)
            if hasattr(w, "_msgid"):
                continue
            try:
                txt = w.cget("text")
            except Exception:
                continue
            if not isinstance(txt, str):
                continue
            txt = txt.strip()
            if not txt:
                continue
            msgid = _I18N_TEXT_TO_KEY.get(txt)
            if msgid:
                self._i18n_bind(w, msgid, "text")

    # ---------------------------
    # i18n: translated combo models
    # ---------------------------
    def _category_display(self, key: str) -> str:
        k = (key or "").lower()
        if "struct" in k:
            return self.t("category.structures")
        if "dino" in k:
            return self.t("category.dinosaurs")
        if "human" in k:
            return self.t("category.humans")
        return key

    def _rebuild_category_combo(self):
        if not hasattr(self, "_category_combo") or not hasattr(self, "_category_keys"):
            return
        self._category_key_to_display = {k: self._category_display(k) for k in self._category_keys}
        self._category_display_to_key = {v: k for k, v in self._category_key_to_display.items()}
        values = [self._category_key_to_display[k] for k in self._category_keys]
        try:
            self._category_combo.configure(values=values)
        except Exception:
            try:
                self._category_combo["values"] = values
            except Exception:
                pass
        cur_key = getattr(self, "_current_category_key", None) or (self._category_keys[0] if self._category_keys else "")
        try:
            self.canvas_category_var.set(self._category_key_to_display.get(cur_key, cur_key))
        except Exception:
            pass

    def _preview_mode_display(self, key: str) -> str:
        if key == "ark_simulation":
            return self.t("preview_mode.ark_simulation")
        return self.t("preview_mode.visual")

    def _rebuild_preview_mode_combo(self):
        if not hasattr(self, "_preview_mode_combo"):
            return
        self._preview_mode_key_to_display = {
            "visual": self._preview_mode_display("visual"),
            "ark_simulation": self._preview_mode_display("ark_simulation"),
        }
        self._preview_mode_display_to_key = {v: k for k, v in self._preview_mode_key_to_display.items()}
        vals = [self._preview_mode_key_to_display["visual"], self._preview_mode_key_to_display["ark_simulation"]]
        try:
            self._preview_mode_combo.configure(values=vals)
        except Exception:
            try:
                self._preview_mode_combo["values"] = vals
            except Exception:
                pass
        cur = getattr(self, "_current_preview_mode_key", "visual")
        try:
            self.preview_mode.set(self._preview_mode_key_to_display.get(cur, self._preview_mode_key_to_display["visual"]))
        except Exception:
            pass

    def _set_preview_mode_key(self, key: str):
        if key not in ("visual", "ark_simulation"):
            return
        self._current_preview_mode_key = key
        try:
            if hasattr(self, "_preview_mode_key_to_display"):
                self.preview_mode.set(self._preview_mode_key_to_display.get(key, key))
            else:
                self.preview_mode.set(self._preview_mode_display(key))
        except Exception:
            pass
        try:
            self.controller.set_preview_mode(key)
        except Exception:
            pass

    def _apply_language(self):
        try:
            self.title(self.t("app.title"))
        except Exception:
            pass

        # Advanced window title (if open)
        try:
            if getattr(self, "_advanced_win", None) is not None and self._advanced_win.winfo_exists():
                self._advanced_win.title(self.t("panel.advanced"))
        except Exception:
            pass

        # Update language combobox
        if hasattr(self, "_lang_combo"):
            vals = [self.t("language.es"), self.t("language.en"), self.t("language.zh"), self.t("language.ru")]
            try:
                self._lang_combo.configure(values=vals)
            except Exception:
                try:
                    self._lang_combo["values"] = vals
                except Exception:
                    pass
            try:
                self._lang_display_to_code = {
                    self.t("language.es"): "es",
                    self.t("language.en"): "en",
                    self.t("language.zh"): "zh",
                    self.t("language.ru"): "ru",
                }
            except Exception:
                pass
            code = getattr(self._i18n, "lang", "es")
            label = {"es": self.t("language.es"), "en": self.t("language.en"), "zh": self.t("language.zh"), "ru": self.t("language.ru")}.get(code, self.t("language.es"))
            try:
                self._lang_combo.set(label)
            except Exception:
                pass

        # Apply bindings
        for w, opt, msgid in list(self._i18n_bindings):
            try:
                w.configure(**{opt: self.t(msgid)})
            except Exception:
                pass

        # Rebuild translated combos
        try:
            self._rebuild_category_combo()
        except Exception:
            pass
        try:
            self._rebuild_preview_mode_combo()
        except Exception:
            pass

        # Hook for external wrappers
        try:
            self._on_language_changed()
        except Exception:
            pass

    def set_language(self, lang_code: str):
        if lang_code not in ("es", "en", "zh", "ru"):
            return
        try:
            self._i18n.set_lang(lang_code)
        except Exception:
            return
        try:
            if not hasattr(self, "_user_cache") or self._user_cache is None:
                self._user_cache = {}
            self._user_cache["ui_lang"] = lang_code
            self._save_user_cache()
        except Exception:
            pass
        self._apply_language()

    def _on_language_changed(self):
        return


    def _schedule_draw(self):
        if not hasattr(self, 'preview_canvas'):
            return
        if self._draw_job is not None:
            try:
                self.after_cancel(self._draw_job)
            except Exception:
                pass
        self._draw_job = self.after(25, self._draw_preview_to_canvas)

    def _draw_preview_to_canvas(self):
        if not hasattr(self, 'preview_canvas'):
            return
        self.preview_canvas.delete("all")

        img = self._render_native
        if img is None:
            return

        cw = self.preview_canvas.winfo_width()
        ch = self.preview_canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            return

        base_w, base_h = img.size
        if base_w <= 0 or base_h <= 0:
            return

        max_scale = min(cw / base_w, ch / base_h)
        max_scale = min(max_scale, 1.0)

        draw_w = max(1, int(base_w * max_scale))
        draw_h = max(1, int(base_h * max_scale))

        if img.size != (draw_w, draw_h):
            resample = Image.BILINEAR if (draw_w < base_w or draw_h < base_h) else Image.NEAREST
            img2 = img.resize((draw_w, draw_h), resample)
        else:
            img2 = img

        self._photo = ImageTk.PhotoImage(img2)
        self.preview_canvas.create_image(
            cw // 2,
            ch // 2,
            image=self._photo,
            anchor="center",
        )        
        

    # ======================================================
    # Layout
    # ======================================================

    def _build_layout(self):
        self.main = ttk.Frame(self)
        self.main.pack(fill=tk.BOTH, expand=True)

        # Left pane (scrollable, fixed width)
        self.controls = ttk.Frame(self.main, width=340)
        self.controls.pack(side=tk.LEFT, fill=tk.Y)
        self.controls.pack_propagate(False)

        self._controls_canvas = tk.Canvas(self.controls, highlightthickness=0, borderwidth=0)
        self._controls_scrollbar = ttk.Scrollbar(self.controls, orient="vertical", command=self._controls_canvas.yview)
        self._controls_canvas.configure(yscrollcommand=self._controls_scrollbar.set)

        self._controls_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._controls_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.controls_inner = ttk.Frame(self._controls_canvas)
        self._controls_canvas_window = self._controls_canvas.create_window(
            (0, 0), window=self.controls_inner, anchor="nw"
        )

        def _on_inner_configure(_event=None):
            try:
                self._controls_canvas.configure(scrollregion=self._controls_canvas.bbox("all"))
            except Exception:
                pass

        def _on_canvas_configure(event):
            # Keep the inner frame width matched to the canvas width
            try:
                self._controls_canvas.itemconfigure(self._controls_canvas_window, width=event.width)
            except Exception:
                pass

        self.controls_inner.bind("<Configure>", _on_inner_configure)
        self._controls_canvas.bind("<Configure>", _on_canvas_configure)

        # Mouse wheel scroll for the left pane (dyes list overrides when hovered)
        def _on_mousewheel(event):
            self._controls_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        self._controls_canvas.bind("<Enter>", lambda e: self._controls_canvas.bind_all("<MouseWheel>", _on_mousewheel))
        self._controls_canvas.bind("<Leave>", lambda e: self._controls_canvas.unbind_all("<MouseWheel>"))

        self.preview_frame = ttk.Frame(self.main)
        self.preview_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)


    def _build_controls(self):
        # Asegurar que la columna se expande bien
        self.controls_inner.columnconfigure(0, weight=1)

        # ==================================================
        # Idioma / Language
        # ==================================================
        lang_frame = ttk.LabelFrame(self.controls_inner, text=self.t("panel.language"))
        lang_frame.pack(fill=tk.X, padx=10, pady=(8, 0))
        self._lang_display_to_code = {
            self.t("language.es"): "es",
            self.t("language.en"): "en",
            self.t("language.zh"): "zh",
            self.t("language.ru"): "ru",
        }
        self._lang_combo = ttk.Combobox(lang_frame, state="readonly", values=list(self._lang_display_to_code.keys()))
        cur_label = {
            "es": self.t("language.es"),
            "en": self.t("language.en"),
            "zh": self.t("language.zh"),
            "ru": self.t("language.ru"),
        }.get(self._i18n.lang, self.t("language.es"))
        self._lang_combo.set(cur_label)
        self._lang_combo.pack(anchor="w", fill=tk.X)
        self._i18n_bind(lang_frame, "panel.language", "text")

        def _on_lang_selected(_e=None):
            disp = (self._lang_combo.get() or "").strip()
            code = self._lang_display_to_code.get(disp)
            if code:
                self.set_language(code)

        self._lang_combo.bind("<<ComboboxSelected>>", _on_lang_selected)


        # ==================================================
        # Imagen
        # ==================================================
        image_frame = ttk.LabelFrame(self.controls_inner, text="Imagen")
        image_frame.pack(fill=tk.X, padx=10, pady=8)

        ttk.Button(
            image_frame,
            text="Abrir imagen…",
            command=self._open_image
        ).pack(anchor="w", fill=tk.X, pady=(0, 4))

        self.image_label = ttk.Label(image_frame, text="(ninguna)")
        self.image_label.pack(anchor="w")

        # ==================================================
        # Canvas
        # ==================================================
        canvas_frame = ttk.LabelFrame(self.controls_inner, text="Canvas")
        canvas_frame.pack(fill=tk.X, padx=10, pady=8)

        # -------- Categoría --------
        self.canvas_category_var = tk.StringVar()
        self._category_keys = list(self._templates_by_category.keys())
        if not self._category_keys:
            self._category_keys = ["Structures"]
        self._current_category_key = self._category_keys[0]

        self._category_key_to_display = {k: self._category_display(k) for k in self._category_keys}
        self._category_display_to_key = {v: k for k, v in self._category_key_to_display.items()}
        self.canvas_category_var.set(self._category_key_to_display.get(self._current_category_key, self._current_category_key))

        self._category_combo = ttk.Combobox(
            canvas_frame,
            textvariable=self.canvas_category_var,
            state="readonly",
            values=[self._category_key_to_display[k] for k in self._category_keys],
        )
        self._category_combo.pack(anchor="w", fill=tk.X, pady=(0, 4))

        def _on_cat_sel(_e=None):
            try:
                self._on_category_selected(self.canvas_category_var.get())
            except Exception:
                pass

        self._category_combo.bind("<<ComboboxSelected>>", _on_cat_sel)

        # -------- Template --------
        self.canvas_var = tk.StringVar()

        self.canvas_selector = ttk.OptionMenu(
            canvas_frame,
            self.canvas_var,
            "",   # se rellena dinámicamente
        )
        self.canvas_selector.pack(anchor="w", fill=tk.X)

        # ------ Dynamic canvas controls (X/Y) -----
        self.dynamic_frame = ttk.LabelFrame(canvas_frame, text="Paint Area")
        self.dynamic_frame.pack(anchor="w", fill=tk.X, padx=6, pady=(6, 0))

        # Oculto por defecto
        self.dynamic_frame.pack_forget()

        # ------ Multi-canvas controls (Rows / Cols) -----
        self.multicanvas_frame = ttk.LabelFrame(canvas_frame, text="Multi-Canvas Grid")

        # Variables
        self.multicanvas_rows_var = tk.IntVar()
        self.multicanvas_cols_var = tk.IntVar()

        ttk.Label(self.multicanvas_frame, text="Rows:").grid(row=0, column=0, sticky="w")
        rows_spin = ttk.Spinbox(
            self.multicanvas_frame,
            from_=1,
            to=10,
            width=5,
            textvariable=self.multicanvas_rows_var,
            command=self._on_multicanvas_changed,
        )
        rows_spin.grid(row=0, column=1, sticky="w", padx=(6, 0))

        ttk.Label(self.multicanvas_frame, text="Cols:").grid(row=1, column=0, sticky="w")
        cols_spin = ttk.Spinbox(
            self.multicanvas_frame,
            from_=1,
            to=10,
            width=5,
            textvariable=self.multicanvas_cols_var,
            command=self._on_multicanvas_changed,
        )
        cols_spin.grid(row=1, column=1, sticky="w", padx=(6, 0))

        # Reaccionar también a edición manual
        rows_spin.bind("<KeyRelease>", lambda e: self._on_multicanvas_changed())
        cols_spin.bind("<KeyRelease>", lambda e: self._on_multicanvas_changed())

        # Oculto por defecto
        self.multicanvas_frame.pack_forget()
    
        # ==================================================
        # Preview Mode
        # ==================================================
        preview_frame = ttk.LabelFrame(self.controls_inner, text="Preview Mode")
        preview_frame.pack(fill=tk.X, padx=10, pady=4)

        # Preview mode: translated labels with stable internal keys
        self.preview_mode = tk.StringVar()
        self._current_preview_mode_key = "visual"
        self._preview_mode_key_to_display = {
            "visual": self._preview_mode_display("visual"),
            "ark_simulation": self._preview_mode_display("ark_simulation"),
        }
        self._preview_mode_display_to_key = {v: k for k, v in self._preview_mode_key_to_display.items()}
        self.preview_mode.set(self._preview_mode_key_to_display["visual"])

        self._preview_mode_combo = ttk.Combobox(
            preview_frame,
            textvariable=self.preview_mode,
            state="readonly",
            values=[
                self._preview_mode_key_to_display["visual"],
                self._preview_mode_key_to_display["ark_simulation"],
            ],
        )
        self._preview_mode_combo.pack(anchor="w", fill=tk.X)

        def _on_pm_sel(_e=None):
            try:
                self._update_preview_mode()
            except Exception:
                pass

        self._preview_mode_combo.bind("<<ComboboxSelected>>", _on_pm_sel)

        self.show_object_var = tk.BooleanVar(value=True)

        ttk.Checkbutton(
            preview_frame,
            text="Mostrar objeto en el juego",
            variable=self.show_object_var,
            command=self._update_show_object,
        ).pack(anchor="w", pady=(6, 0))

        # ==================================================
        # Writer Mode (.pnt)
        # ==================================================
        # Oculto en la GUI principal (por defecto raster20).
        # Se mantiene la variable para compat/debug y para restaurar tras External mode.
        self.writer_mode_var = tk.StringVar(value="raster20")
        try:
            self.controller.set_writer_mode("raster20")
        except Exception:
            pass

# Dyes (Generation)
        # ==================================================
        dyes_frame = ttk.LabelFrame(self.controls_inner, text="Dyes (Generación)")
        dyes_frame.pack(fill=tk.X, padx=8, pady=4)

        self.use_all_dyes_var = tk.BooleanVar(value=True)

        ttk.Checkbutton(
            dyes_frame,
            text="Usar todos los dyes",
            variable=self.use_all_dyes_var,
            command=self._toggle_all_dyes,
        ).pack(anchor="w")

        # --------------------------------------------------
        # Best colors (auto palette)
        # --------------------------------------------------
        best_frame = ttk.Frame(dyes_frame)
        best_frame.pack(fill=tk.X, pady=(4, 6))

        ttk.Label(best_frame, text="Best colors:").pack(side="left")

        self.best_dyes_var = tk.IntVar(value=40)

        best_spin = ttk.Spinbox(
            best_frame,
            from_=1,
            to=len(self.dye_vars) if self.dye_vars else 255,
            textvariable=self.best_dyes_var,
            width=5
        )
        best_spin.pack(side="left", padx=(6, 6))

        ttk.Button(
            best_frame,
            text="Calcular",
            command=self._on_calculate_best_dyes
        ).pack(side="left")
        search_frame = ttk.Frame(dyes_frame)
        search_frame.pack(fill=tk.X, pady=(0, 1))

        search_frame.columnconfigure(0, weight=1)
        search_frame.columnconfigure(1, weight=1, uniform="visbtn")
        search_frame.columnconfigure(2, weight=1, uniform="visbtn")

        self.dyes_search_var = tk.StringVar()
        search_entry = ttk.Entry(
            search_frame,
            textvariable=self.dyes_search_var,
            width=9
        )
        search_entry.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        search_entry.bind("<KeyRelease>", self._filter_dyes_grid)

        self._btn_activate_visibles = ttk.Button(
            search_frame,
            text="Activar visibles",
            command=self._activate_visible_dyes
        )
        self._btn_activate_visibles.grid(row=0, column=1, sticky="ew", padx=(0, 6))

        self._btn_deactivate_visibles = ttk.Button(
            search_frame,
            text="Desactivar visibles",
            command=self._deactivate_visible_dyes
        )
        self._btn_deactivate_visibles.grid(row=0, column=2, sticky="ew")

        default_canvas = "Sign_PaintingCanvas_C"

        self.canvas_var.set(default_canvas)

        # --------------------------------------------------
        # Scrollable dyes list
        # --------------------------------------------------
        dyes_canvas = tk.Canvas(
            dyes_frame,
            height=140,            # altura visible del "viewport"
            highlightthickness=0
        )
        dyes_scrollbar = ttk.Scrollbar(
            dyes_frame,
            orient="vertical",
            command=dyes_canvas.yview
        )

        dyes_canvas.configure(yscrollcommand=dyes_scrollbar.set)

        dyes_canvas.pack(side="left", fill="both", expand=True)
        dyes_scrollbar.pack(side="right", fill="y")

        self.dyes_list_frame = ttk.Frame(dyes_canvas)

        self._dyes_canvas_window = dyes_canvas.create_window(
            (0, 0),
            window=self.dyes_list_frame,
            anchor="nw"
        )

        self._init_dye_vars()
        self._populate_dyes_grid()

        # Inicializar selector Canvas
        self._on_category_selected(self.canvas_category_var.get())
        
        # --------------------------------------------------
        # Scroll helpers (LOCAL functions)
        # --------------------------------------------------
        def _on_dyes_frame_configure(event):
            dyes_canvas.configure(
                scrollregion=dyes_canvas.bbox("all")
            )

        self.dyes_list_frame.bind(
            "<Configure>",
            _on_dyes_frame_configure
        )

        def _on_mousewheel(event):
            # Windows / Linux
            dyes_canvas.yview_scroll(
                int(-1 * (event.delta / 120)),
                "units"
            )

        dyes_canvas.bind("<Enter>", lambda e: dyes_canvas.bind_all("<MouseWheel>", _on_mousewheel))
        dyes_canvas.bind("<Leave>", lambda e: dyes_canvas.unbind_all("<MouseWheel>"))

        # ==================================================
        # Border
        # ==================================================
        border_frame = ttk.LabelFrame(self.controls_inner, text="Border")
        border_frame.pack(fill=tk.X, padx=10, pady=4)

        # Compact row: mode + load button
        border_row = ttk.Frame(border_frame)
        border_row.pack(fill=tk.X, pady=(0, 2))

        self.border_style = tk.StringVar(value="none")
        ttk.OptionMenu(
            border_row,
            self.border_style,
            "none", "none", "image",
            command=self._on_border_style_changed
        ).pack(side="left", fill=tk.X, expand=True)

        self.border_load_button = ttk.Button(
            border_row,
            text="Cargar…",
            command=self._load_border_image
        )
        self.border_load_button.pack(side="right", padx=(6, 0))

        self.border_image_label = ttk.Label(border_frame, text="(sin frame)")
        self.border_image_label.pack(anchor="w", pady=(0, 2))

        self.border_size = tk.IntVar(value=0)
        self.border_thickness_scale = tk.Scale(
            border_frame,
            from_=0,
            to=20,
            orient="horizontal",
            length=120,
            sliderlength=10,
            width=8,
            showvalue=False,
            command=lambda v: self._on_border_changed()
        )
        self.border_thickness_scale.pack(fill=tk.X, padx=4)
        self._sync_border_controls()

# Dithering
        # ==================================================
        dither_frame = ttk.LabelFrame(self.controls_inner, text="Dithering")
        dither_frame.pack(fill=tk.X, padx=10, pady=4)

        self.dither_mode = tk.StringVar(value="none")
        ttk.OptionMenu(
            dither_frame,
            self.dither_mode,
            "none",
            "none",
            "palette_fs",
            "palette_ordered",
            
            command=lambda _: self._update_dither()
        ).pack(anchor="w", fill=tk.X, pady=(0, 4))

        self.dither_strength = tk.DoubleVar(value=0.5)
        self.dithering_strength_scale = tk.Scale(
            dither_frame,
            from_=0,
            to=100,
            orient="horizontal",
            length=120,
            sliderlength=10,
            width=8,
            showvalue=False,
            command=lambda v: self._on_dithering_changed()
        )
        self.dithering_strength_scale.pack(fill=tk.X, padx=4)
        
        # ==================================================
        # Advanced (hidden by default)
        # ==================================================
        adv_toggle = ttk.Frame(self.controls_inner)
        adv_toggle.pack(fill=tk.X, padx=10, pady=(6, 2))

        self.show_advanced_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            adv_toggle,
            text="Mostrar advanced",
            variable=self.show_advanced_var,
            command=self._on_toggle_advanced,
        ).pack(anchor="w")

        # Advanced controls live in an external window (see _ensure_advanced_window)
        self._advanced_win = None


        # ==================================================
        # Generar
        # ==================================================
        ttk.Button(
            self.controls_inner,
            text="Generar .PNT",
            command=self._on_generate
        ).pack(anchor="w", fill=tk.X, padx=10, pady=(20, 6))

        self.gen_status = ttk.Label(self.controls_inner, text="", foreground="orange")
        self.gen_status.pack(anchor="w", padx=10, pady=(4, 0))


    # ======================================================
    # Preview
    # ======================================================

    def _build_preview(self):
        self.preview_canvas = tk.Canvas(self.preview_frame, bg="#2b2b2b", highlightthickness=0)
        self.preview_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self._photo = None

    # ======================================================
    # Advanced panel (Tkinter)
    # ======================================================



    def _position_advanced_window(self):
        """Place the advanced window attached to the LEFT of the main window."""

        if getattr(self, "_is_closing", False):
            return
        if getattr(self, "_advanced_win", None) is None:
            return
        try:
            self.update_idletasks()
            self._advanced_win.update_idletasks()

            w = self._advanced_win.winfo_width()
            h = self._advanced_win.winfo_height()
            if w <= 1 or h <= 1:
                w, h = 420, 640

            x = self.winfo_x() - w - 8
            y = self.winfo_y()
            if x < 0:
                x = 0
            if y < 0:
                y = 0

            self._advanced_win.geometry(f"{w}x{h}+{x}+{y}")
        except Exception:
            pass

    def _schedule_position_advanced_window(self):
        if getattr(self, "_is_closing", False):
            return

        if getattr(self, "_advanced_win", None) is None:
            return
        try:
            if not self._advanced_win.winfo_exists():
                return
        except Exception:
            return
        try:
            if not self.show_advanced_var.get():
                return
        except Exception:
            return

        if getattr(self, "_adv_pos_scheduled", False):
            return
        self._adv_pos_scheduled = True

        def _do():
            self._adv_pos_scheduled = False
            try:
                if str(self.state()) == "iconic":
                    try:
                        self._advanced_win.withdraw()
                    except Exception:
                        pass
                    return
            except Exception:
                pass
            self._position_advanced_window()

        try:
            self.after_idle(_do)
        except Exception:
            try:
                self.after(0, _do)
            except Exception:
                pass

    def _on_main_configure(self, _e=None):
        self._schedule_position_advanced_window()


    def _on_main_close(self):
        """Close the whole app (main + advanced) in one click."""
        self._is_closing = True
        try:
            if getattr(self, "_advanced_win", None) is not None and self._advanced_win.winfo_exists():
                try:
                    self._advanced_win.destroy()
                except Exception:
                    pass
        except Exception:
            pass

        # Use the standard close path to stop background workers cleanly
        try:
            self._on_close()
        except Exception:
            try:
                self.destroy()
            except Exception:
                pass

    def _ensure_advanced_window(self):
        if getattr(self, "_advanced_win", None) is not None:
            try:
                if self._advanced_win.winfo_exists():
                    return
            except Exception:
                pass

        self._advanced_win = tk.Toplevel(self)
        try:
            self._advanced_win.title(self.t("panel.advanced"))
        except Exception:
            self._advanced_win.title("Advanced")

        def _on_adv_close():
            # Closing Advanced should hide it without re-opening.
            if getattr(self, "_is_closing", False):
                try:
                    self._advanced_win.destroy()
                except Exception:
                    pass
                return
            # Closing the Advanced window should NOT reopen it
            try:
                self.show_advanced_var.set(False)
            except Exception:
                pass
            try:
                self._advanced_win.withdraw()
            except Exception:
                pass

        try:
            self._advanced_win.protocol("WM_DELETE_WINDOW", _on_adv_close)
        except Exception:
            pass


        self._advanced_win.transient(self)
        self._advanced_win.resizable(True, True)

        # Close behavior: keep state in sync (no "magia")
        def _on_close():
            try:
                self.show_advanced_var.set(False)
            except Exception:
                pass
            try:
                self._advanced_win.withdraw()
            except Exception:
                pass

        # Content
        body = ttk.Frame(self._advanced_win)
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Build advanced controls inside this window
        self._build_advanced_controls(parent=body)

        # Ensure language applies to newly created widgets
        try:
            self._i18n_scan_and_bind()
            self._apply_language()
        except Exception:
            pass

        # Initial geometry
        try:
            self._advanced_win.geometry("420x640")
        except Exception:
            pass
        self._position_advanced_window()


    def _on_toggle_advanced(self):
        if self.show_advanced_var.get():
            self._ensure_advanced_window()
            try:
                self._advanced_win.deiconify()
            except Exception:
                pass
            self._position_advanced_window()
            try:
                self._advanced_win.lift()
            except Exception:
                pass
        else:
            try:
                if getattr(self, "_advanced_win", None) is not None:
                    self._advanced_win.withdraw()
            except Exception:
                pass

    def _build_advanced_controls(self, *, parent: ttk.Frame):
        """Minimal advanced options, hidden by default.

        Current scope: External .pnt picker for MyPaintings/LocalSaved.
        """
        parent.columnconfigure(0, weight=1)

        # -------- External mode --------
        ext = ttk.LabelFrame(parent, text="External .pnt")
        ext.pack(fill=tk.X, padx=6, pady=6)
        ext.columnconfigure(1, weight=1)

        self.external_enabled_var = tk.BooleanVar(value=False)
        self.external_root_var = tk.StringVar(value=str(self._last_scan_dir or self._default_scan_dir or ""))
        self.external_recursive_var = tk.BooleanVar(value=True)
        self.external_detect_guid_var = tk.BooleanVar(value=True)
        self.external_max_files_var = tk.IntVar(value=1200)
        self.external_preserve_var = tk.BooleanVar(value=True)

        ttk.Checkbutton(
            ext,
            text="Activar External .pnt (advanced)",
            variable=self.external_enabled_var,
            command=self._on_external_enabled_changed,
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 6))

        ttk.Label(ext, text="Carpeta:").grid(row=1, column=0, sticky="w")
        self._ext_root_entry = ttk.Entry(ext, textvariable=self.external_root_var)
        self._ext_root_entry.grid(row=1, column=1, sticky="ew", padx=(6, 6))
        self._ext_browse_btn = ttk.Button(ext, text="Browse…", command=self._browse_external_root)
        self._ext_browse_btn.grid(row=1, column=2, sticky="e")

        opts = ttk.Frame(ext)
        opts.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(6, 6))

        ttk.Checkbutton(opts, text="Recursive", variable=self.external_recursive_var).pack(side="left")
        ttk.Checkbutton(opts, text="Detect GUID", variable=self.external_detect_guid_var).pack(side="left", padx=(8, 0))
        ttk.Label(opts, text="Max:").pack(side="left", padx=(8, 0))
        ttk.Spinbox(opts, from_=50, to=20000, width=6, textvariable=self.external_max_files_var).pack(side="left")

        act = ttk.Frame(ext)
        act.grid(row=3, column=0, columnspan=3, sticky="ew")
        act.columnconfigure(0, weight=1)
        # --- Row 1: actions ---
        act_top = ttk.Frame(act)
        act_top.grid(row=0, column=0, sticky="ew")
        act_top.columnconfigure(0, weight=1, uniform="advbtn")
        act_top.columnconfigure(1, weight=1, uniform="advbtn")
        act_top.columnconfigure(2, weight=1, uniform="advbtn")

        self._ext_scan_btn = ttk.Button(act_top, text="Scan", command=self._scan_external_pnts_async)
        self._ext_scan_btn.grid(row=0, column=0, sticky="ew")

        self._ext_copy_btn = ttk.Button(act_top, text="Copy BP/Size", command=self._copy_external_bp_size)
        self._ext_copy_btn.grid(row=0, column=1, sticky="ew", padx=(8, 0))

        self._ext_use_btn = ttk.Button(act_top, text="Use for Generate", command=self._use_external_for_generate)
        self._ext_use_btn.grid(row=0, column=2, sticky="ew", padx=(8, 0))
        # --- Row 2: hint + preserve ---
        act_bot = ttk.Frame(act)
        act_bot.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        act_bot.columnconfigure(0, weight=0)
        act_bot.columnconfigure(1, weight=0)
        act_bot.columnconfigure(2, weight=0)
        act_bot.columnconfigure(3, weight=1)

        ttk.Label(act_bot, text="Hint Y:").grid(row=0, column=0, sticky="w")
        self.external_hint_y_var = tk.StringVar(value="")
        self._ext_hint_y_entry = ttk.Entry(act_bot, textvariable=self.external_hint_y_var, width=6)
        self._ext_hint_y_entry.grid(row=0, column=1, sticky="w", padx=(6, 8))
        self._ext_apply_y_btn = ttk.Button(act_bot, text="Apply", command=self._apply_external_hint_y)
        self._ext_apply_y_btn.grid(row=0, column=2, sticky="w", padx=(0, 12))

        self._ext_preserve_cb = ttk.Checkbutton(
            act_bot,
            text="Preserve metadata (GUID/suffix)",
            variable=self.external_preserve_var,
            command=self._on_external_preserve_changed,
        )
        self._ext_preserve_cb.grid(row=0, column=3, sticky="e")

        # List (Treeview so the information fits)
        list_frame = ttk.Frame(ext)
        list_frame.grid(row=4, column=0, columnspan=3, sticky="nsew", pady=(6, 0))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)

        cols = ("name", "size", "blueprint", "kind")
        self.external_tree = ttk.Treeview(
            list_frame,
            columns=cols,
            show="headings",
            height=10,
            selectmode="browse",
        )
        self.external_tree.heading("name", text="Archivo")
        self.external_tree.heading("size", text="Size")
        self.external_tree.heading("blueprint", text="Blueprint")
        self.external_tree.heading("kind", text="Tipo")

        # Reasonable default widths; Treeview will expand with the frame
        self.external_tree.column("name", width=220, anchor="w")
        self.external_tree.column("size", width=120, anchor="center")
        self.external_tree.column("blueprint", width=260, anchor="w")
        self.external_tree.column("kind", width=60, anchor="center")

        sb = ttk.Scrollbar(list_frame, orient="vertical", command=self.external_tree.yview)
        self.external_tree.configure(yscrollcommand=sb.set)
        self.external_tree.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")

        self.external_tree.bind("<<TreeviewSelect>>", self._on_external_list_select)

        self._ext_status = ttk.Label(ext, text="", foreground="gray")
        self._ext_status.grid(row=5, column=0, columnspan=3, sticky="w", pady=(6, 0))

        # Disable controls by default (since advanced is off)
        self._set_external_controls_enabled(False)

        # ==================================================
        # Dino tools (hidden by default)
        # ==================================================
        ttk.Separator(parent, orient="horizontal").pack(fill=tk.X, padx=6, pady=(10, 6))

        self.show_dino_tools_var = tk.BooleanVar(value=False)
        self._dino_tools_toggle = ttk.Checkbutton(
            parent,
            text=self.t("chk.show_dino_tools"),
            variable=self.show_dino_tools_var,
            command=self._on_toggle_dino_tools,
        )
        self._dino_tools_toggle.pack(anchor="w", padx=6, pady=(0, 4))
        self._i18n_bind(self._dino_tools_toggle, "chk.show_dino_tools", "text")

        self._dino_tools_frame = ttk.LabelFrame(parent, text=self.t("panel.dino_tools"))
        self._i18n_bind(self._dino_tools_frame, "panel.dino_tools", "text")

        # Content (kept simple, no pipeline integration)
        self._dino_sel_label = ttk.Label(self._dino_tools_frame, text=self.t("hint.dino_select"), foreground="gray")
        self._dino_sel_label.pack(anchor="w", padx=6, pady=(4, 6))
        self._i18n_bind(self._dino_sel_label, "hint.dino_select", "text")

        btns = ttk.Frame(self._dino_tools_frame)
        btns.pack(fill=tk.X, padx=6, pady=(0, 6))
        btns.columnconfigure(0, weight=1)
        btns.columnconfigure(1, weight=1)

        self._btn_export_user_mask = ttk.Button(
            btns,
            text=self.t("btn.export_user_mask"),
            command=self._export_selected_user_mask,
        )
        self._btn_export_user_mask.grid(row=0, column=0, sticky="ew")
        self._i18n_bind(self._btn_export_user_mask, "btn.export_user_mask", "text")

        self._btn_open_usermasks = ttk.Button(
            btns,
            text=self.t("btn.open_user_masks"),
            command=self._open_user_masks_folder,
        )
        self._btn_open_usermasks.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        self._i18n_bind(self._btn_open_usermasks, "btn.open_user_masks", "text")

        # Refine (optional): generates *_mask_user_refined.png using *_mask_user_P.png if present
        self._btn_refine_user_mask = ttk.Button(
            btns,
            text=self.t("btn.refine_user_mask"),
            command=self._refine_selected_user_mask,
        )
        self._btn_refine_user_mask.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        self._i18n_bind(self._btn_refine_user_mask, "btn.refine_user_mask", "text")

        self._dino_status = ttk.Label(self._dino_tools_frame, text="", foreground="gray")
        self._dino_status.pack(anchor="w", padx=6, pady=(0, 4))

        # Internal selection state
        self._dino_selected_item = None
        self._set_dino_tools_enabled(False)

    def _set_external_controls_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for w in (
            getattr(self, "_ext_root_entry", None),
            getattr(self, "_ext_browse_btn", None),
            getattr(self, "_ext_scan_btn", None),
            getattr(self, "_ext_copy_btn", None),
            getattr(self, "_ext_use_btn", None),
            getattr(self, "_ext_hint_y_entry", None),
            getattr(self, "_ext_apply_y_btn", None),
            getattr(self, "_ext_preserve_cb", None),
            getattr(self, "external_tree", None),
        ):
            if w is None:
                continue
            try:
                w.configure(state=state)
            except Exception:
                pass

    def _browse_external_root(self):
        initial = (self.external_root_var.get().strip() or str(getattr(self, "_last_scan_dir", None) or self._default_scan_dir or "") or str(self._last_open_image_dir))
        d = filedialog.askdirectory(initialdir=initial)
        if d:
            self.external_root_var.set(d)
            try:
                self._last_scan_dir = Path(d)
                self._save_user_cache()
            except Exception:
                pass

    def _on_external_enabled_changed(self):
        enabled = bool(self.external_enabled_var.get())

        if enabled:
            # store previous selection once
            if self._external_prev_selection is None:
                self._external_prev_selection = (
                    getattr(self, "_current_category_key", None) or self.canvas_category_var.get(),
                    self.canvas_var.get(),
                    self.writer_mode_var.get(),
                )

            # lock standard selectors to avoid contract mismatches
            try:
                self._category_combo.configure(state="disabled")
            except Exception:
                pass
            try:
                self.canvas_selector.configure(state="disabled")
            except Exception:
                pass
            try:
                self.writer_mode_menu.configure(state="disabled")
            except Exception:
                pass

            # Hide template-specific controls to avoid stale UI
            try:
                self._hide_dynamic_size_controls()
            except Exception:
                pass
            try:
                self._hide_multicanvas_controls()
            except Exception:
                pass

            self._set_external_controls_enabled(True)
            self._ext_status.config(text=self.t("status.external_mode_active"))

            # If there is already a selected entry, apply it
            self._apply_selected_external_if_any()
        else:
            # Clear controller external selection
            try:
                self.controller.clear_external_pnt()
            except Exception:
                pass

            # restore standard selectors
            try:
                self._category_combo.configure(state="readonly")
            except Exception:
                pass
            try:
                self.canvas_selector.configure(state="normal")
            except Exception:
                pass
            try:
                self.writer_mode_menu.configure(state="normal")
            except Exception:
                pass

            self._set_external_controls_enabled(False)
            self._ext_status.config(text="")

            # Restore previous selection if captured
            if self._external_prev_selection:
                cat, tid, wmode = self._external_prev_selection
                try:
                    # cat may be internal key or translated label
                    self._on_category_selected(cat)
                    if tid:
                        self._on_template_selected(tid)
                except Exception:
                    pass
                try:
                    self.writer_mode_var.set(wmode)
                    self._on_writer_mode_changed(wmode)
                except Exception:
                    pass

            self._external_prev_selection = None
            self._schedule_redraw()

    def _on_external_preserve_changed(self):
        if not self.external_enabled_var.get():
            return
        self._apply_external_writer_mode()

    def _apply_external_writer_mode(self):
        if not self.external_enabled_var.get():
            return
        try:
            if self.external_preserve_var.get():
                self.controller.set_writer_mode("preserve_source")
            else:
                # In external mode, non-preserve defaults to raster20
                self.controller.set_writer_mode("raster20")
        except Exception:
            pass

    def _scan_external_pnts_async(self):
        root = self.external_root_var.get().strip()
        if not root:
            # Default to ASA MyPaintings (user requested)
            try:
                root = str(self._default_scan_dir)
                self.external_root_var.set(root)
            except Exception:
                root = ""
        if not root:
            self._ext_status.config(text="Selecciona una carpeta primero.", foreground="orange")
            return
        try:
            self._last_scan_dir = Path(root)
            self._save_user_cache()
        except Exception:
            pass
            return

        self._ext_scan_seq += 1
        seq = self._ext_scan_seq

        self._ext_status.config(text=self.t("status.scanning"), foreground="gray")
        try:
            self._ext_scan_btn.configure(state="disabled")
        except Exception:
            pass

        recursive = bool(self.external_recursive_var.get())
        detect_guid = bool(self.external_detect_guid_var.get())
        max_files = int(self.external_max_files_var.get() or 1200)

        def _worker():
            try:
                res = scan_pnts(
                    Path(root),
                    recursive=recursive,
                    max_files=max_files,
                    detect_guid=detect_guid,
                )
            except Exception as e:
                self._ext_scan_res_q.put((seq, None, None, str(e)))
                return
            self._ext_scan_res_q.put((seq, res, root, None))

        threading.Thread(target=_worker, name="ExternalPntScan", daemon=True).start()

        if self._ext_scan_poll_job is None:
            self._ext_scan_poll_job = self.after(80, self._poll_external_scan)

    def _poll_external_scan(self):
        self._ext_scan_poll_job = None
        latest = None
        try:
            while True:
                latest = self._ext_scan_res_q.get_nowait()
        except queue.Empty:
            pass

        if latest is None:
            self._ext_scan_poll_job = self.after(80, self._poll_external_scan)
            return

        seq, res, root, err = latest
        if seq != self._ext_scan_seq:
            # stale
            self._ext_scan_poll_job = self.after(80, self._poll_external_scan)
            return

        if isinstance(res, dict):
            items = res.get("items") or []
            meta = res
        else:
            items = res or []
            meta = None

        self._ext_items = list(items)
        try:
            for iid in self.external_tree.get_children(""):
                self.external_tree.delete(iid)
        except Exception:
            pass

        for idx, it in enumerate(self._ext_items):
            name = it.get("name", "")
            kind = str(it.get("kind") or ("H20" if it.get("is_header20") else "UNK"))

            # Size: exact WxH if resolved; else show best|alt1 (and +N if needed).
            size_txt = "?"
            w = it.get("width")
            h = it.get("height")
            if isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0:
                size_txt = f"{w}x{h}"
            else:
                cands = it.get("candidates") or []
                try:
                    if cands:
                        best = cands[0]
                        bw = int(best.get("w", 0))
                        bh = int(best.get("h", 0))
                        if bw > 0 and bh > 0:
                            size_txt = f"{bw}x{bh}"
                        if len(cands) >= 2:
                            alt = cands[1]
                            aw = int(alt.get("w", 0))
                            ah = int(alt.get("h", 0))
                            if aw > 0 and ah > 0:
                                size_txt = f"{size_txt}|{aw}x{ah}"
                        if len(cands) > 2:
                            size_txt = f"{size_txt} (+{len(cands)-2})"
                except Exception:
                    pass

            bp = it.get("blueprint") or it.get("class_name") or ""

            try:
                self.external_tree.insert(
                    "",
                    "end",
                    iid=str(idx),
                    values=(name, size_txt, bp, kind),
                )
            except Exception:
                self.external_tree.insert("", "end", values=(name, size_txt, bp, kind))
        if err:
            self._ext_status.config(text=self.t("status.scan_error", err=err), foreground="orange")
        else:
            if len(self._ext_items) == 0:
                # If the scan was truncated, say so explicitly.
                extra = ""
                try:
                    if meta and meta.get("truncated"):
                        extra = f" (scan truncated: {meta.get('reason')})"
                except Exception:
                    extra = ""
                self._ext_status.config(
                    text=self.t("status.scan_ok_zero_long", extra=extra),
                    foreground="gray",
                )
            else:
                # Show brief stats so the user knows it did scan something.
                stats = ""
                try:
                    if meta:
                        stats = self.t("status.scan_stats", dirs=meta.get("walk_dirs",0), files=meta.get("walk_files",0), secs=meta.get("elapsed_s",0))
                        if meta.get("truncated"):
                            stats += self.t("status.scan_stats_trunc", reason=meta.get("reason"))
                except Exception:
                    stats = ""
                self._ext_status.config(text=self.t("status.scan_ok", n=len(self._ext_items), stats=stats), foreground="gray")

        try:
            self._ext_scan_btn.configure(state="normal")
        except Exception:
            pass

        # Auto-select first entry for convenience
        if self._ext_items:
            try:
                self.external_tree.selection_set("0")
                self.external_tree.focus("0")
                self.external_tree.see("0")
                self.external_tree.event_generate("<<TreeviewSelect>>")
            except Exception:
                pass

    def _on_external_list_select(self, _evt=None):
        if not self.external_enabled_var.get():
            return
        self._apply_selected_external_if_any()
        # Dino tools (user masks) are driven by the external selection too.
        try:
            self._update_dino_tools_from_external_selection()
        except Exception:
            pass

    def _apply_selected_external_if_any(self):
        # Treeview selection
        try:
            sel = self.external_tree.selection()
            if not sel:
                return
            idx = int(sel[0])
        except Exception:
            return

        if idx < 0 or idx >= len(self._ext_items):
            return

        it = self._ext_items[idx]
        p = it.get("path")
        if not p:
            return

        # Only header20 files are valid as "External base" for the writer mode.
        # Non-header20 (EXT_GUID) entries are for inspection (blueprint/size).
        if not bool(it.get("is_header20", False)):
            kind = it.get("kind") or "EXT"
            bp = it.get("blueprint") or it.get("class_name") or "?"
            # Size: exact if resolved; else best|alt1.
            size_txt = "?"
            w = it.get("width")
            h = it.get("height")
            if isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0:
                size_txt = f"{w}x{h}"
            else:
                cands = it.get("candidates") or []
                try:
                    if cands:
                        bw = int(cands[0].get("w", 0))
                        bh = int(cands[0].get("h", 0))
                        if bw > 0 and bh > 0:
                            size_txt = f"{bw}x{bh}"
                        if len(cands) >= 2:
                            aw = int(cands[1].get("w", 0))
                            ah = int(cands[1].get("h", 0))
                            if aw > 0 and ah > 0:
                                size_txt = f"{size_txt}|{aw}x{ah}"
                        if len(cands) > 2:
                            size_txt = f"{size_txt} (+{len(cands)-2})"
                except Exception:
                    pass
            disp_name = it.get("name") or Path(p).stem
            self._ext_status.config(text=self.t("status.inspect", name=disp_name, kind=kind, bp=bp, size=size_txt), foreground="gray")
            return

        try:
            self.controller.set_external_pnt(Path(p))
            self._apply_external_writer_mode()
            try:
                self._ext_status.config(text=self.t("status.selected", p=p), foreground="gray")
            except Exception:
                pass
            self._schedule_redraw()
        except Exception as e:
            self._ext_status.config(text=self.t("status.external_select_error", err=e), foreground="orange")

    
    def _copy_external_bp_size(self):
        """Copies blueprint + size (or candidates) of the selected scanned .pnt to clipboard."""
        try:
            sel = self.external_tree.selection()
            if not sel:
                return
            idx = int(sel[0])
        except Exception:
            return
        if idx < 0 or idx >= len(self._ext_items):
            return
        it = self._ext_items[idx]
        bp = it.get("blueprint") or it.get("class_name") or ""
        size_txt = ""
        w = it.get("width")
        h = it.get("height")
        if isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0:
            size_txt = f"{w}x{h}"
        else:
            cands = it.get("candidates") or []
            try:
                if cands:
                    bw = int(cands[0].get("w", 0))
                    bh = int(cands[0].get("h", 0))
                    if bw > 0 and bh > 0:
                        size_txt = f"{bw}x{bh}"
                    if len(cands) >= 2:
                        aw = int(cands[1].get("w", 0))
                        ah = int(cands[1].get("h", 0))
                        if aw > 0 and ah > 0:
                            size_txt = f"{size_txt}|{aw}x{ah}"
                    if len(cands) > 2:
                        size_txt = f"{size_txt} (+{len(cands)-2})"
            except Exception:
                size_txt = ""
        txt = ""
        if bp and size_txt:
            txt = f"{bp}\t{size_txt}"
        elif bp:
            txt = str(bp)
        elif size_txt:
            txt = str(size_txt)
        if not txt:
            txt = "?"
        try:
            self.clipboard_clear()
            self.clipboard_append(txt)
            self.update_idletasks()
            self._ext_status.config(text=self.t("status.copied", txt=txt), foreground="gray")
        except Exception as e:
            try:
                self._ext_status.config(text=self.t("status.copy_failed", err=e), foreground="orange")
            except Exception:
                pass

    
    # ------------------------------------------------------------
    # External scan helpers
    # ------------------------------------------------------------

    def _get_selected_external_index(self) -> int | None:
        try:
            sel = self.external_tree.selection()
            if not sel:
                return None
            return int(sel[0])
        except Exception:
            return None

    def _format_external_item_size(self, it: dict) -> str:
        # exact if resolved
        w = it.get("width")
        h = it.get("height")
        if isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0:
            return f"{w}x{h}"

        cands = it.get("candidates") or []
        try:
            if cands:
                bw = int(cands[0].get("w", 0))
                bh = int(cands[0].get("h", 0))
                if bw > 0 and bh > 0:
                    s = f"{bw}x{bh}"
                else:
                    s = "?"
                if len(cands) >= 2:
                    aw = int(cands[1].get("w", 0))
                    ah = int(cands[1].get("h", 0))
                    if aw > 0 and ah > 0:
                        s = f"{s}|{aw}x{ah}"
                if len(cands) > 2:
                    s = f"{s} (+{len(cands)-2})"
                return s
        except Exception:
            pass
        return "?"

    def _update_external_tree_row(self, idx: int) -> None:
        try:
            if idx < 0 or idx >= len(self._ext_items):
                return
            it = self._ext_items[idx]
            name = it.get("name", "")
            kind = str(it.get("kind") or ("H20" if it.get("is_header20") else "UNK"))
            size_txt = self._format_external_item_size(it)
            bp = it.get("blueprint") or it.get("class_name") or ""
            self.external_tree.item(str(idx), values=(name, size_txt, bp, kind))
        except Exception:
            pass

    def _apply_external_hint_y(self):
        """User provides Y (height shown by the game) to disambiguate best vs alt1."""
        if not self.external_enabled_var.get():
            return
        idx = self._get_selected_external_index()
        if idx is None or idx < 0 or idx >= len(self._ext_items):
            return

        try:
            y = int(str(self.external_hint_y_var.get()).strip())
        except Exception:
            try:
                self._ext_status.config(text="Hint Y inválido. Ej: 128 / 256 / 512 / 968 / 712", foreground="orange")
            except Exception:
                pass
            return

        it = self._ext_items[idx]
        cands = it.get("candidates") or []
        if not cands:
            try:
                self._ext_status.config(text="Este .pnt no tiene candidates para resolver por Y.", foreground="orange")
            except Exception:
                pass
            return

        best = None
        best_score = 10**9
        best_swap = False
        for c in cands:
            try:
                w = int(c.get("w", 0))
                h = int(c.get("h", 0))
            except Exception:
                continue
            if w <= 0 or h <= 0:
                continue
            d_h = abs(h - y)
            d_w = abs(w - y)
            if d_h <= d_w:
                score = d_h
                swap = False
            else:
                score = d_w
                swap = True
            if score < best_score:
                best_score = score
                best = (w, h)
                best_swap = swap

        if not best:
            try:
                self._ext_status.config(text="No se pudo resolver size con ese Y.", foreground="orange")
            except Exception:
                pass
            return

        w, h = best
        if best_swap:
            w, h = h, w  # force Y as height

        it["width"] = int(w)
        it["height"] = int(h)
        it["resolved_by_y"] = int(y)

        # Update row (hide alt by showing exact WxH)
        self._update_external_tree_row(idx)

        try:
            self._ext_status.config(text=self.t("status.resolved_by_y", y=y, bp=it.get("blueprint","?"), w=w, h=h), foreground="gray")
        except Exception:
            pass

    def _use_external_for_generate(self):
        """Apply scanned blueprint+size into the normal generation flow (header20)."""
        idx = self._get_selected_external_index()
        if idx is None or idx < 0 or idx >= len(self._ext_items):
            return

        it = self._ext_items[idx]
        bp = (it.get("blueprint") or it.get("class_name") or "").strip()
        if not bp:
            try:
                self._ext_status.config(text="No blueprint detectado en este .pnt (necesito blueprint para generar).", foreground="orange")
            except Exception:
                pass
            return

        # Determine size (best) if present
        w = it.get("width")
        h = it.get("height")
        if not (isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0):
            cands = it.get("candidates") or []
            try:
                if cands:
                    w = int(cands[0].get("w", 0))
                    h = int(cands[0].get("h", 0))
            except Exception:
                w, h = None, None

        # Normalize blueprint -> template_id (dynamic physical ids should map to abstract)
        template_id = bp
        if "Canvas_Dynamic" in bp:
            template_id = "StructureBP_Canvas_Dynamic_C"

        # Find category (Structures/Dinos/Humans)
        category = None
        try:
            for cat, tids in self._templates_by_category.items():
                if template_id in tids:
                    category = cat
                    break
        except Exception:
            category = None

        # No fuzzy matching here: if we can't find an exact template, we fail
        # deterministically to avoid wrong blueprint/template selection (e.g. Raptor vs Microraptor).

        if not category:
            try:
                self._ext_status.config(text=self.t("err.no_template_for_blueprint", bp=bp), foreground="orange")
            except Exception:
                pass
            return

        # Always exit external mode first (no restore)
        if self.external_enabled_var.get():
            try:
                self._external_prev_selection = None
            except Exception:
                pass
            try:
                self.external_enabled_var.set(False)
                self._on_external_enabled_changed()
            except Exception:
                # if something fails, keep going
                pass

        # Select template in the normal UI
        try:
            self.canvas_category_var.set(category)
            self._on_category_selected(category)
            self._on_template_selected(template_id)
        except Exception as e:
            try:
                self._ext_status.config(text=f"Error al seleccionar template: {e}", foreground="orange")
            except Exception:
                pass
            return

        # If dynamic, apply visible area
        try:
            descriptor = self.controller.state.preview_descriptor
            if descriptor and descriptor.get("dynamic") is not None and isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0:
                # Ensure vars exist
                if hasattr(self, "visible_x_var") and hasattr(self, "visible_y_var"):
                    self.visible_x_var.set(int(w))
                    self.visible_y_var.set(int(h))
                    self._on_dynamic_visible_area_changed()
        except Exception:
            pass

        try:
            sz = f"{w}x{h}" if isinstance(w, int) and isinstance(h, int) and w and h else "?"
            self._ext_status.config(text=self.t("status.applied_to_generate", template_id=template_id, sz=sz), foreground="gray")
        except Exception:
            pass

        self._schedule_redraw()


    # ==================================================
    # Dino tools (UserMasks) — prepared, not integrated
    # ==================================================
    def _on_toggle_dino_tools(self):
        show = bool(self.show_dino_tools_var.get())
        try:
            if show:
                # pack just under the toggle
                self._dino_tools_frame.pack(fill=tk.X, padx=6, pady=(0, 6))
                self._grow_advanced_window_to_fit()
            else:
                self._dino_tools_frame.pack_forget()
        except Exception:
            pass

        # Always refresh enable/label state (so it is ready when shown)
        try:
            self._update_dino_tools_from_external_selection()
        except Exception:
            pass

    def _grow_advanced_window_to_fit(self):
        """Grow the docked Advanced window vertically to fit newly shown sections."""
        win = getattr(self, "_advanced_win", None)
        if win is None:
            return
        try:
            if not win.winfo_exists():
                return
        except Exception:
            return

        try:
            win.update_idletasks()
            req_h = int(win.winfo_reqheight())
            cur_h = int(win.winfo_height())
            cur_w = int(win.winfo_width())
            x = int(win.winfo_x())
            y = int(win.winfo_y())
            if req_h > cur_h:
                try:
                    win.minsize(cur_w, req_h)
                except Exception:
                    pass
                win.geometry(f"{cur_w}x{req_h}+{x}+{y}")
        except Exception:
            pass

        # Re-dock using the current size
        try:
            self._position_advanced_window()
        except Exception:
            pass

    def _set_dino_tools_enabled(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        for w in (
            getattr(self, "_btn_export_user_mask", None),
            getattr(self, "_btn_open_usermasks", None),
            getattr(self, "_btn_refine_user_mask", None),
        ):
            if w is None:
                continue
            try:
                w.configure(state=state)
            except Exception:
                pass

    def _get_selected_external_item(self) -> dict | None:
        idx = None
        try:
            idx = self._get_selected_external_index()
        except Exception:
            idx = None
        if idx is None:
            return None
        if idx < 0 or idx >= len(getattr(self, "_ext_items", []) or []):
            return None
        return self._ext_items[idx]

    def _update_dino_tools_from_external_selection(self):
        it = self._get_selected_external_item()
        if not it:
            self._dino_selected_item = None
            try:
                self._dino_sel_label.configure(text=self.t("hint.dino_select"))
            except Exception:
                pass
            self._set_dino_tools_enabled(False)
            return

        bp = (it.get("blueprint") or it.get("class_name") or "").strip()
        if not bp or not bp.endswith("_Character_BP_C"):
            self._dino_selected_item = None
            try:
                self._dino_sel_label.configure(text=self.t("hint.dino_not_dino", bp=(bp or "?")))
            except Exception:
                pass
            self._set_dino_tools_enabled(False)
            return

        # Size (best-effort)
        w = it.get("width")
        h = it.get("height")
        if not (isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0):
            cands = it.get("candidates") or []
            try:
                if cands:
                    w = int(cands[0].get("w", 0))
                    h = int(cands[0].get("h", 0))
            except Exception:
                w, h = None, None

        name = it.get("name") or "?"
        sz = f"{w}x{h}" if isinstance(w, int) and isinstance(h, int) and w and h else "?"
        try:
            self._dino_sel_label.configure(text=self.t("hint.dino_selected", name=name, bp=bp, sz=sz))
        except Exception:
            try:
                self._dino_sel_label.configure(text=f"{name} | {bp} | {sz}")
            except Exception:
                pass

        self._dino_selected_item = it
        self._set_dino_tools_enabled(True)

        self._update_refine_button_state(bp)

    def _export_selected_user_mask(self):
        it = getattr(self, "_dino_selected_item", None)
        if not it:
            try:
                messagebox.showwarning(self.t("msg.title"), self.t("err.no_dino_selected"))
            except Exception:
                pass
            return

        p = it.get("path")
        bp = (it.get("blueprint") or it.get("class_name") or "").strip()
        if not p or not bp:
            try:
                messagebox.showwarning(self.t("msg.title"), self.t("err.no_dino_selected"))
            except Exception:
                pass
            return

        # Size (best-effort)
        w = it.get("width")
        h = it.get("height")
        if not (isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0):
            cands = it.get("candidates") or []
            try:
                if cands:
                    w = int(cands[0].get("w", 0))
                    h = int(cands[0].get("h", 0))
            except Exception:
                w, h = None, None

        out_dir = Path(self.templates_root) / "UserMasks"
        try:
            meta = save_user_mask_pack(
                Path(p),
                blueprint=bp,
                out_dir=out_dir,
                width=(int(w) if isinstance(w, int) and w > 0 else None),
                height=(int(h) if isinstance(h, int) and h > 0 else None),
                crop_to_bbox=False,
            )
        except Exception as e:
            try:
                self._dino_status.configure(text=self.t("err.user_mask_export", err=str(e)), foreground="orange")
            except Exception:
                pass
            try:
                messagebox.showerror(self.t("msg.title"), f"{self.t('err.user_mask_export', err=str(e))}")
            except Exception:
                pass
            return

        try:
            self._dino_status.configure(text=self.t("status.user_mask_exported", bp=bp, out=str(out_dir)), foreground="gray")
        except Exception:
            pass

        # Refresh templates so the virtual template appears
        try:
            self.controller.template_loader.reload_virtual_templates()
        except Exception:
            pass
        self._reload_current_category_templates()
        self._update_refine_button_state(bp)

    def _open_user_masks_folder(self):
        p = Path(self.templates_root) / "UserMasks"
        try:
            p.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        try:
            if os.name == "nt":
                os.startfile(str(p))  # type: ignore[attr-defined]
                return
        except Exception:
            pass

        # Non-Windows fallback
        try:
            import subprocess
            import sys
            if sys.platform == "darwin":
                subprocess.Popen(["open", str(p)])
            else:
                subprocess.Popen(["xdg-open", str(p)])
        except Exception:
            pass



    def _update_refine_button_state(self, blueprint: str):
        try:
            btn = getattr(self, "_btn_refine_user_mask", None)
            if btn is None:
                return
            bp = (blueprint or "").strip()
            if not bp:
                btn.configure(state="disabled")
                return
            um_dir = Path(self.templates_root) / "UserMasks"
            base_png = um_dir / f"{bp}_mask_user.png"
            pairs_png = um_dir / f"{bp}_mask_user_P.png"
            state = "normal" if (base_png.exists() and pairs_png.exists()) else "disabled"
            btn.configure(state=state)
        except Exception:
            pass

    def _reload_current_category_templates(self):
        """Refresh current category template menu (used after exporting/refining UserMasks)."""
        try:
            self._templates_by_category = self._group_templates_by_category()
            cat_key = getattr(self, "_current_category_key", None) or "Structures"
            templates = self._templates_by_category.get(cat_key, []) or []
            if not templates:
                return
            cur_tid = (self.canvas_var.get() or "").strip()
            menu = self.canvas_selector["menu"]
            menu.delete(0, "end")
            for tid in templates:
                menu.add_command(
                    label=tid,
                    command=lambda v=tid: self._on_template_selected(v)
                )
            # preserve selection if possible
            if cur_tid in templates:
                self._on_template_selected(cur_tid)
            else:
                self._on_template_selected(templates[0])
        except Exception:
            pass

    def _refine_selected_user_mask(self):
        it = getattr(self, "_dino_selected_item", None)
        if not it:
            try:
                messagebox.showwarning(self.t("msg.title"), self.t("err.no_dino_selected"))
            except Exception:
                pass
            return

        bp = (it.get("blueprint") or it.get("class_name") or "").strip()
        if not bp:
            try:
                messagebox.showwarning(self.t("msg.title"), self.t("err.no_dino_selected"))
            except Exception:
                pass
            return

        out_dir = Path(self.templates_root) / "UserMasks"
        try:
            meta = refine_user_mask_pack_existing(blueprint=bp, out_dir=out_dir)
        except Exception as e:
            try:
                self._dino_status.configure(text=self.t("err.user_mask_refine", err=str(e)), foreground="orange")
            except Exception:
                pass
            try:
                messagebox.showerror(self.t("msg.title"), f"{self.t('err.user_mask_refine', err=str(e))}")
            except Exception:
                pass
            return

        try:
            self._dino_status.configure(text=self.t("status.user_mask_refined", bp=bp, out=str(out_dir)), foreground="gray")
        except Exception:
            pass

        # Refresh templates so the virtual template appears (e.g., Doggo_Character_BP_C)
        try:
            self.controller.template_loader.reload_virtual_templates()
        except Exception:
            pass
        self._reload_current_category_templates()

    def _update_preview_mode(self):
        disp = self.preview_mode.get()
        try:
            key = getattr(self, "_preview_mode_display_to_key", {}).get(disp, disp)
        except Exception:
            key = disp
        if key not in ("visual", "ark_simulation"):
            key = "visual"
        self._current_preview_mode_key = key
        self.controller.set_preview_mode(key)
        self._request_preview_render()

    def _schedule_redraw(self):
        if self._redraw_job is not None:
            try:
                self.after_cancel(self._redraw_job)
            except Exception:
                pass
        self._redraw_job = self.after(self._redraw_delay_ms, self._request_preview_render)

    def _render_preview_native(self):
        descriptor = self.controller.state.preview_descriptor

        if descriptor and descriptor["identity"]["type"] == "multi_canvas":
            img = self.controller.render_preview_multicanvas_cached()
        else:
            img = self.controller.render_preview_if_possible()

        self._render_native = img
        self._schedule_draw()

    def _request_preview_render(self):
        # Fallback sync si quieres poder desactivar async rápidamente
        if not getattr(self, "_async_preview_enabled", True):
            self._render_preview_native()
            return

        self._preview_seq += 1
        seq = self._preview_seq

        snapshot = self.controller.build_preview_snapshot()
        self._preview_req_q.put((seq, snapshot))

        self._ensure_preview_polling()

    def _ensure_preview_polling(self):
        if self._preview_poll_job is None:
            self._preview_poll_job = self.after(30, self._poll_preview_results)

    def _poll_preview_results(self):
        self._preview_poll_job = None

        latest = None
        try:
            while True:
                latest = self._preview_res_q.get_nowait()
        except queue.Empty:
            pass

        if latest is not None:
            seq, img = latest

            # Discard resultados viejos
            if seq == self._preview_seq:
                self._preview_last_applied_seq = seq
                self._render_native = img
                self._schedule_draw()

        # Mientras haya una request más nueva pendiente, seguimos poll
        if self._preview_last_applied_seq < self._preview_seq:
            self._ensure_preview_polling()

    def _preview_worker_loop(self):
        while True:
            item = self._preview_req_q.get()
            if item is None:
                break

            seq, snapshot = item

            # Coalescing: quedarse con la última request disponible
            try:
                while True:
                    nxt = self._preview_req_q.get_nowait()
                    if nxt is None:
                        return
                    seq, snapshot = nxt
            except queue.Empty:
                pass

            img = None
            try:
                img = self.controller.render_preview_from_snapshot(snapshot)
            except Exception:
                img = None

            self._preview_res_q.put((seq, img))


            
    def _rgb_to_hex(self, rgb):
        r, g, b = rgb
        return f"#{int(r):02x}{int(g):02x}{int(b):02x}"

    def _linear_to_srgb_channel(self, c: float) -> float:
        if c <= 0.0031308:
            return 12.92 * c
        else:
            return 1.055 * (c ** (1 / 2.4)) - 0.055

    def _linear_rgb_to_hex(self, linear_rgb):
        r, g, b = linear_rgb

        sr = self._linear_to_srgb_channel(r)
        sg = self._linear_to_srgb_channel(g)
        sb = self._linear_to_srgb_channel(b)

        return self._rgb_to_hex((
            sr * 255,
            sg * 255,
            sb * 255
        ))
    
    def _on_calculate_best_dyes(self):
        try:
            X = int(self.best_dyes_var.get())
        except Exception:
            return
        if X <= 0:
            return
        if self.controller.state.image_original is None:
            return

        if getattr(self, "_best_dyes_running", False):
            return

        self._best_dyes_running = True
        self.gen_status.config(text="Calculando best dyes…")

        threading.Thread(
            target=self._best_dyes_worker,
            args=(X,),
            daemon=True
        ).start()

    def _best_dyes_worker(self, X: int):
        try:
            selected = self.controller.calculate_best_dyes(X, sample_side=256, max_pixels=65536)
        except Exception:
            selected = []
        self.after(0, lambda: self._apply_best_dyes_result(selected))

    def _apply_best_dyes_result(self, selected: list[int]):
        self._best_dyes_running = False
        self.gen_status.config(text="")

        selected_set = set(selected)
        for dye_id, var in self.dye_vars.items():
            var.set(dye_id in selected_set)

        self.use_all_dyes_var.set(len(selected_set) == len(self.dye_vars))
        self._update_all_dye_swatches()
        self._schedule_redraw()


    def _on_border_changed(self):
        size = int(self.border_thickness_scale.get())
        self.controller.set_border_size(size)
        self._schedule_redraw()
        
    def _sync_border_controls(self):
        """
        Habilita/deshabilita controles de Border según border_style.
        - none: todo disabled
        - image: enabled
        """
        is_image = (self.border_style.get() == "image")

        # Slider grosor
        try:
            self.border_thickness_scale.configure(state=("normal" if is_image else "disabled"))
        except Exception:
            pass

        # Botón cargar imagen
        try:
            self.border_load_button.configure(state=("normal" if is_image else "disabled"))
        except Exception:
            pass


    def _sync_dither_strength_visibility(self):
        """Muestra/oculta el slider de fuerza según el modo de dithering.

        - none / palette_ordered: no se usa → oculto
        - palette_fs: se usa → visible
        """
        mode = str(self.dither_mode.get() or "none")
        if mode == "palette_fs":
            if not self.dithering_strength_scale.winfo_ismapped():
                self.dithering_strength_scale.pack(fill=tk.X, padx=4)
        else:
            if self.dithering_strength_scale.winfo_ismapped():
                self.dithering_strength_scale.pack_forget()
    def _on_border_style_changed(self, val: str):
        # Sanear
        if val not in ("none", "image"):
            val = "none"

        # UI + Controller
        self.border_style.set(val)
        self.controller.set_border_style(val)

        # En/disable controles
        self._sync_border_controls()

        # Redraw
        self._schedule_redraw()

    def _on_dithering_changed(self):
        strength = float(self.dithering_strength_scale.get()) / 100.0
        self.controller.set_dithering_config(
            mode=self.dither_mode.get(),
            strength=strength
        )
        self._schedule_redraw()


    # ======================================================
    # Events → Controller
    # ======================================================

    def _open_image(self):
        path = filedialog.askopenfilename(
            initialdir=self._last_open_image_dir,
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")]
         )
        if not path:
            return

        # Cache
        self._last_open_image_dir = Path(path).parent
        self._save_user_cache()
        
        try:
            self._last_border_dir = Path(path).parent
            self._save_user_cache()
        except Exception:
            pass

        img = Image.open(path).convert("RGBA")
        self.controller.set_image(img)
        self.image_label.config(text=os.path.basename(path))
        self._schedule_redraw()

    def _on_category_selected(self, category: str):
        try:
            cat_key = getattr(self, "_category_display_to_key", {}).get(category, category)
        except Exception:
            cat_key = category
        self._current_category_key = cat_key
        templates = self._templates_by_category.get(cat_key, [])
        if not templates:
            return

        menu = self.canvas_selector["menu"]
        menu.delete(0, "end")

        for tid in templates:
            menu.add_command(
                label=tid,
                command=lambda v=tid: self._on_template_selected(v)
            )

        # seleccionar primero
        self._on_template_selected(templates[0])

    def _on_template_selected(self, template_id: str):
        self.canvas_var.set(template_id)
        self._on_canvas_selected(template_id)
        self.controller.state.selected_template_id = template_id

    def _on_canvas_selected(self, template_id: str):
        """
        Maneja la selección de un canvas/template desde la GUI,
        siguiendo estrictamente el schema 1.1.
        """
        # --------------------------------------------------
        # Cargar descriptor NORMALIZADO (NO resuelto)
        # --------------------------------------------------
        descriptor = self.controller.template_loader.load(template_id)
        self.controller.state.preview_descriptor = descriptor

        # Reset UI común
        self._hide_dynamic_size_controls()
        self._hide_multicanvas_controls()

        # Reset estado base
        self.controller.state.template = None
        self.controller.state.canvas_resolved = None

        tpl_type = descriptor["identity"]["type"]

        # --------------------------------------------------
        # Caso 1: Multi-canvas
        # --------------------------------------------------
        if tpl_type == "multi_canvas":
            # Limpiar request multi-canvas
            self.controller.state.canvas_is_dynamic = False
            self.controller.state.canvas_request = None

            # Mostrar controles multi-canvas
            self._show_multicanvas_controls(descriptor)

            # Preview multi-canvas se basa SOLO en preview_descriptor
            self._schedule_redraw()
            return

        # --------------------------------------------------
        # Caso 2: Canvas dinámico
        # --------------------------------------------------
        if descriptor["dynamic"] is not None:
            self.controller.state.canvas_is_dynamic = True
            # Mostrar controles dinámicos (X / Y)
            self._show_dynamic_size_controls(descriptor)

            # NO resolvemos aún el template
            # El usuario debe introducir X/Y
            self._schedule_redraw()
            return

        # --------------------------------------------------
        # Caso 3: Canvas estático normal
        # --------------------------------------------------
        # Resolver directamente el template
        self.controller.state.canvas_is_dynamic = False
        self.controller.set_template(template_id)

        # Redibujar preview normal
        self._schedule_redraw()

    def _show_dynamic_size_controls(self, template: dict):
        """
        Muestra controles para seleccionar área visible (X / Y)
        en canvas dinámicos.
        """

        # Asegurar que el frame esté visible
        self.dynamic_frame.pack(anchor="w", fill=tk.X, padx=10, pady=6)

        base_w = 128
        base_h = 128

        # Variables (crear una sola vez)
        if not hasattr(self, "visible_x_var"):
            self.visible_x_var = tk.IntVar(value=base_w)
            self.visible_y_var = tk.IntVar(value=base_h)

            # ---- X ----
            x_spin = ttk.Spinbox(
                self.dynamic_frame,
                from_=128,
                to=968,
                textvariable=self.visible_x_var,
                width=6,
                command=self._on_dynamic_visible_area_changed
            )

            x_spin.grid(row=0, column=1, sticky="w", padx=(6, 0))

            # ---- Y ----
            y_spin = ttk.Spinbox(
                self.dynamic_frame,
                from_=128,
                to=968,
                textvariable=self.visible_y_var,
                width=6,
                command=self._on_dynamic_visible_area_changed
            )

            y_spin.grid(row=1, column=1, sticky="w", padx=(6, 0))

            # También reaccionar a edición manual
            x_spin.bind("<KeyRelease>", lambda e: self._on_dynamic_visible_area_changed())
            y_spin.bind("<KeyRelease>", lambda e: self._on_dynamic_visible_area_changed())

        else:
            # Resetear valores al cambiar de template
            self.visible_x_var.set(base_w)
            self.visible_y_var.set(base_h)

        # Aplicar inmediatamente
        self._on_dynamic_visible_area_changed()


    def _hide_dynamic_size_controls(self):
        """
        Oculta controles dinámicos.
        """
        if hasattr(self, "dynamic_frame"):
            self.dynamic_frame.pack_forget()

    def _hide_multicanvas_controls(self):
        """
        Oculta los controles de multi-canvas (cols / rows).
        """
        if hasattr(self, "multicanvas_frame"):
            self.multicanvas_frame.pack_forget()

    def _show_multicanvas_controls(self, descriptor):
        multi = descriptor["multi_canvas"]

        # Defaults del descriptor
        default_rows = multi["rows"]["default"]
        default_cols = multi["cols"]["default"]

        self.multicanvas_rows_var.set(default_rows)
        self.multicanvas_cols_var.set(default_cols)

        # Crear request inicial
        self.controller.set_multicanvas_request(
            rows=default_rows,
            cols=default_cols,
        )

        self.multicanvas_frame.pack(anchor="w", fill=tk.X, padx=10, pady=6)

    def _on_multicanvas_changed(self):
        try:
            rows = int(self.multicanvas_rows_var.get())
            cols = int(self.multicanvas_cols_var.get())
        except Exception:
            return

        # Actualizar request
        self.controller.set_multicanvas_request(
            rows=rows,
            cols=cols,
        )
        self._schedule_redraw()

    def _on_dynamic_visible_area_changed(self):
        try:
            visible_w = int(self.visible_x_var.get())
            visible_h = int(self.visible_y_var.get())
        except Exception:
            return

        # Guardar request visible
        self.controller.state.canvas_is_dynamic = True
        self.controller.set_dynamic_canvas_request(
            rows_y=visible_h,
            blocks_x=visible_w,
            mode="visible_area",
        )

        # 🔹 PREVIEW VIRTUAL
        self.controller.set_dynamic_preview_canvas(
            width=visible_w,
            height=visible_h,
        )

        self._schedule_redraw()

    def _load_border_image(self):
        initial = str(getattr(self, "_last_border_dir", None) or self._default_border_dir or Path.home())
        path = filedialog.askopenfilename(
            initialdir=initial,
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")],
        )
        if not path:
            return

        try:
            self._last_border_dir = Path(path).parent
            self._save_user_cache()
        except Exception:
            pass

        img = Image.open(path).convert("RGBA")

        # Guardar frame image en controller
        self.controller.set_border_frame_image(img)
        self.border_image_label.config(text=os.path.basename(path))

        # Forzar modo image (UI + controller) y habilitar controles
        self._on_border_style_changed("image")

    def _on_writer_mode_changed(self, mode: str):
        try:
            self.controller.set_writer_mode(mode)
        except Exception as e:
            print(f"[WARN] writer_mode inválido: {e}")
            self.writer_mode_var.set("raster20")
            self.controller.set_writer_mode("raster20")
        self._schedule_redraw()


    def _update_dither(self):
        mode = self.dither_mode.get()

        if mode.startswith("palette") and getattr(self, "_current_preview_mode_key", "visual") != "ark_simulation":
            self._set_preview_mode_key("ark_simulation")
            # controller updated by _set_preview_mode_key

        strength = float(self.dithering_strength_scale.get()) / 100.0
        self.controller.set_dithering_config(mode=mode, strength=strength)
        self._schedule_redraw()

    def _update_show_object(self):
        self.controller.state.show_game_object = self.show_object_var.get()
        self._schedule_redraw()

    def _init_dye_vars(self):
        if self.dye_vars:
            return
        
        translator = self.controller._ark_translator
        if translator is None:
            return
        
        for dye in translator.dyes:
            self.dye_vars[dye.observed_byte] = tk.BooleanVar(value=True)

    def _populate_dyes_grid(self):
        DYE_COLS = 11
        SWATCH_SIZE = 18
        SWATCH_PAD = 3
        for widget in self.dyes_list_frame.winfo_children():
            widget.destroy()
            
        self.dye_swatches.clear()
            
        translator = self.controller._ark_translator

        for idx, dye in enumerate(translator.dyes):
            row = idx // DYE_COLS
            col = idx % DYE_COLS

            hex_color = self._linear_rgb_to_hex(dye.linear_rgb)

            swatch = tk.Canvas(
                self.dyes_list_frame,
                width=SWATCH_SIZE,
                height=SWATCH_SIZE,
                highlightbackground="#444444",
                bd=0
            )

            # Marco exterior (siempre igual)
            frame_rect = swatch.create_rectangle(
                0, 0, SWATCH_SIZE, SWATCH_SIZE,
                fill="#2b2b2b",
                outline=""
            )

            # Rectángulo de color (este cambia)
            color_rect = swatch.create_rectangle(
                4, 4, SWATCH_SIZE-4, SWATCH_SIZE-4,
                fill=hex_color,
                outline=""
            )

            swatch._frame_rect = frame_rect
            swatch._color_rect = color_rect
            swatch._base_color = hex_color
            
            overlay_id = swatch.create_rectangle(
                0, 0, SWATCH_SIZE, SWATCH_SIZE,
                fill="#000000",
                stipple="gray50",
                outline="",
                state="hidden"
            )
            swatch._overlay_id = overlay_id

            swatch.grid(
                row=row,
                column=col,
                padx=SWATCH_PAD,
                pady=SWATCH_PAD
            )
            self.dye_swatches[dye.observed_byte] = swatch
            tooltip_text = f"{dye.name} ({dye.observed_byte})"

            swatch.bind(
                "<Enter>",
                lambda e, t=tooltip_text: self._show_dye_tooltip(t, e.x_root, e.y_root)
            )
            swatch.bind(
                "<Motion>",
                lambda e: self._move_dye_tooltip(e.x_root, e.y_root)
            )
            swatch.bind(
                "<Leave>",
                lambda e: self._hide_dye_tooltip()
            )

            def on_click(event, dye_id=dye.observed_byte):
                var = self.dye_vars[dye_id]
                var.set(not var.get())
                
                self._update_all_dye_swatches()                
                self._on_dye_toggled(dye_id)
                self._schedule_redraw()
                
            swatch.bind("<Button-1>", on_click)

    def _filter_dyes_grid(self, event=None):
        query = self.dyes_search_var.get().strip().lower()

        translator = self.controller._ark_translator
        if translator is None:
            return

        for dye in translator.dyes:
            swatch = self.dye_swatches.get(dye.observed_byte)
            if swatch is None:
                continue

            name = dye.name.lower()
            id_str = str(dye.observed_byte)

            match = (
                not query
                or query in name
                or query in id_str
            )

            if match:
                swatch.grid()
            else:
                swatch.grid_remove()
                
    def _toggle_all_dyes(self):
        if self.use_all_dyes_var.get():
            # Activar todos
            for dye_id, var in self.dye_vars.items():
                var.set(True)
            self.controller.set_enabled_dyes(None)
        else:
            # Desactivar todos
            for dye_id, var in self.dye_vars.items():
                var.set(False)
            self.controller.set_enabled_dyes(set())

        self._update_all_dye_swatches()                
        self._schedule_redraw()

    def _on_dye_toggled(self, dye_id):
        enabled = {d for d, var in self.dye_vars.items() if var.get()}

        if not enabled:
            self.use_all_dyes_var.set(False)
            self.controller.set_enabled_dyes(set())
        elif len(enabled) == len(self.dye_vars):
            self.use_all_dyes_var.set(True)
            self.controller.set_enabled_dyes(None)
        else:
            self.use_all_dyes_var.set(False)
            self.controller.set_enabled_dyes(enabled)

    def _show_dye_tooltip(self, text, x, y):
        self._tooltip_label.config(text=text)
        self._tooltip.geometry(f"+{x+12}+{y+12}")
        self._tooltip.deiconify()

    def _move_dye_tooltip(self, x, y):
        self._tooltip.geometry(f"+{x+12}+{y+12}")

    def _hide_dye_tooltip(self):
        self._tooltip.withdraw()
    def _activate_visible_dyes(self):
        changed = False
        for dye_id, swatch in self.dye_swatches.items():
            if not swatch.winfo_ismapped():
                continue

            var = self.dye_vars[dye_id]
            if not var.get():
                var.set(True)
                self._on_dye_toggled(dye_id)
                changed = True

        if changed:
            self._schedule_redraw()
        self._update_all_dye_swatches()
            
    def _deactivate_visible_dyes(self):
        changed = False
        for dye_id, swatch in self.dye_swatches.items():
            if not swatch.winfo_ismapped():
                continue

            var = self.dye_vars[dye_id]
            if var.get():
                var.set(False)
                self._on_dye_toggled(dye_id)
                changed = True

        if changed:
            self._schedule_redraw()
        self._update_all_dye_swatches()
            
    def _update_all_dye_swatches(self):
        for dye_id, swatch in self.dye_swatches.items():
            base = swatch._base_color

            if self.dye_vars[dye_id].get():
                dark = self._darken_hex(base, 0.8)
                swatch.coords(
                    swatch._color_rect,
                    5, 5, 14, 14
                )
                swatch.itemconfigure(
                    swatch._color_rect,
                    fill=dark
                )
            else:
                swatch.coords(
                    swatch._color_rect,
                    2, 2, 20, 20
                )
                swatch.itemconfigure(
                    swatch._color_rect,
                    fill=base
                )
            
    def _darken_hex(self, hex_color, factor=0.4):
        hex_color = hex_color.lstrip("#")
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
            
        r = int(r * factor)
        g = int(g * factor)
        b = int(b * factor)

        return f"#{r:02x}{g:02x}{b:02x}"
    
    def _group_templates_by_category(self):
        """
        Agrupa templates por dominio usando
        resolved.asset_dir (fuente canónica).
        """

        result = {
            "Structures": [],
            "Dinos": [],
            "Humans": [],
        }

        loader = self.controller.template_loader
        template_ids = loader.list_templates(include_abstract=False)

        for tid in template_ids:
            tpl = loader.load(tid)

            asset_dir = tpl["resolved"]["asset_dir"]
            if asset_dir in result:
                result[asset_dir].append(tid)

        for k in result:
            result[k].sort()

        return result


    # ======================================================
    # Render
    # ======================================================
    def _redraw(self):
        self.preview_canvas.delete("all")
        # --------------------------------------------------
        # Obtener imagen (fuente)
        # --------------------------------------------------
        descriptor = self.controller.state.preview_descriptor
        if descriptor and descriptor["identity"]["type"] == "multi_canvas":
            img = self.controller.render_preview_multicanvas()
        else:
            img = self.controller.render_preview_if_possible()

        if img is None:
                return

        # --------------------------------------------------
        # Flujo normal
        # --------------------------------------------------

        cw = self.preview_canvas.winfo_width()
        ch = self.preview_canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            return

        base_w, base_h = img.size
        if base_w <= 0 or base_h <= 0:
            return

        # --------------------------------------------------
        # Ajustar al canvas SIN romper proporción
        # --------------------------------------------------
        max_scale = min(
            cw / base_w,
            ch / base_h,
        )
        # Nunca escalar hacia arriba
        max_scale = min(max_scale, 1.0)

        # Clamp: no permitir que la imagen crezca más allá del canvas
        fit_scale = max_scale

        draw_w = max(1, int(base_w * fit_scale))
        draw_h = max(1, int(base_h * fit_scale))

        # --------------------------------------------------
        # Redimensionar SOLO para mostrar
        # --------------------------------------------------
        if img.size != (draw_w, draw_h):
            if draw_w < base_w or draw_h < base_h:
                resample = Image.BILINEAR
            else:
                resample = Image.NEAREST

            img = img.resize((draw_w, draw_h), resample)

        # --------------------------------------------------
        # Dibujar centrado
        # --------------------------------------------------
        self._photo = ImageTk.PhotoImage(img)
        self.preview_canvas.create_image(
            cw // 2,
            ch // 2,
            image=self._photo,
            anchor="center",
        )
        
        
          

    # ======================================================
    # Generate PNT (threaded)
    # ======================================================

    def _set_generating_ui(self, generating: bool):
        self._is_generating = generating
        self.gen_status.config(
            text="Generando .PNT…" if generating else ""
        )

        state = "disabled" if generating else "normal"
        for child in self.controls_inner.winfo_children():
            if child is self.gen_status:
                continue
            try:
                child.configure(state=state)
            except Exception:
                pass

    def _on_generate(self):
        from pathlib import Path
        if self._is_generating:
            return

        # Descriptor activo
        descriptor = self.controller.state.preview_descriptor
        if descriptor is None:
            raise RuntimeError("No hay descriptor activo")

        identity = descriptor["identity"]
        tpl_type = identity["type"]

        # Resolver dinámico justo antes de generar (solo legacy_copy)
        eff_writer = self.controller.get_effective_writer_mode(descriptor)
        if descriptor.get("dynamic") is not None and eff_writer == "legacy_copy":
            req = self.controller.state.canvas_request
            if not req:
                raise RuntimeError("Canvas dinámico sin área visible definida")
            template_id = self.controller.resolve_dynamic_physical_template_id(descriptor, req)
            self.controller.set_template(template_id, None)

        # Template final (físico, multi o None en raster20)
        template = self.controller.state.template
        if tpl_type == "multi_canvas":
            identity = descriptor["identity"]
        else:
            identity = (template["identity"] if template is not None else descriptor["identity"])
        tpl_type = identity["type"]

        # Nombre por defecto
        if tpl_type == "multi_canvas":
            default_name = identity["id"]
        else:
            if template is None:
                c = self.controller.state.canvas_resolved or {}
                w = int(c.get("width", 0) or 0)
                h = int(c.get("height", 0) or 0)
                base = descriptor.get("preview", {}).get("base_name") or identity.get("id") or "Canvas"
                default_name = f"{w}x{h}_{base}.pnt"
            else:
                default_name = template["resolved"]["pnt"]

        path = filedialog.asksaveasfilename(
            initialdir=self._last_generate_dir,
            initialfile=default_name,
            defaultextension=".pnt",
            filetypes=[("ARK Painting", "*.pnt")],
        )
        if not path:
            return

        self._last_generate_dir = Path(path).parent
        self._save_user_cache()

        output_path = Path(path)

        # Encolar generación (1 job máximo)
        self._gen_job_seq += 1
        self._gen_active_job_id = self._gen_job_seq

        # Pausar preview durante export para evitar concurrencia con controller
        self._async_preview_enabled = False

        self._set_generating_ui(True)
        self._show_generation_modal(f"Generando…\n{output_path.name}")

        kind = "multi" if tpl_type == "multi_canvas" else "single"
        self._gen_req_q.put((self._gen_active_job_id, kind, output_path, self.tabla_dyes_path))
        self._ensure_gen_polling()

    def _show_generation_modal(self, title: str = "Generando…"):
        if self._gen_modal is not None:
            return

        win = tk.Toplevel(self)
        win.title("Generando")
        win.transient(self)
        win.resizable(False, False)
        win.grab_set()

        # Centrado aproximado
        try:
            self.update_idletasks()
            x = self.winfo_x() + (self.winfo_width() // 2) - 170
            y = self.winfo_y() + (self.winfo_height() // 2) - 60
            win.geometry(f"340x120+{x}+{y}")
        except Exception:
            pass

        lbl = ttk.Label(win, text=title, justify="center")
        lbl.pack(padx=12, pady=(14, 8), fill="x")

        pb = ttk.Progressbar(win, mode="indeterminate")
        pb.pack(padx=14, pady=(0, 12), fill="x")
        pb.start(10)

        self._gen_modal = win
        self._gen_progress = pb

    def _hide_generation_modal(self):
        win = self._gen_modal
        if win is None:
            return
        try:
            if self._gen_progress is not None:
                self._gen_progress.stop()
        except Exception:
            pass
        try:
            win.grab_release()
        except Exception:
            pass
        try:
            win.destroy()
        except Exception:
            pass
        self._gen_modal = None
        self._gen_progress = None

    def _ensure_gen_polling(self):
        if self._gen_poll_job is None:
            self._gen_poll_job = self.after(50, self._poll_gen_results)

    def _poll_gen_results(self):
        self._gen_poll_job = None
        handled = False
        while True:
            try:
                job_id, ok, payload = self._gen_res_q.get_nowait()
            except queue.Empty:
                break

            # Solo aplicamos el último job activo (1 job máximo de todos modos)
            if job_id != self._gen_active_job_id:
                continue

            handled = True
            self._hide_generation_modal()
            self._set_generating_ui(False)

            # Reanudar previews
            self._async_preview_enabled = True
            self._schedule_redraw()

            if ok:
                # Feedback mínimo (sin modal extra)
                self.gen_status.config(text="Generación completada")
            else:
                err_msg, tb = payload
                print(tb)
                from tkinter import messagebox
                messagebox.showerror(self.t("title.error_generate"), err_msg)

        if not handled and self._is_generating:
            self._ensure_gen_polling()

    def _gen_worker_loop(self):
        import traceback
        while True:
            job = self._gen_req_q.get()
            if job is None:
                break

            job_id, kind, output_path, tabla_dyes_path = job
            try:
                if kind == "multi":
                    self.controller.requests_generation(
                        output_path=output_path,
                        tabla_dyes_path=tabla_dyes_path,
                    )
                else:
                    self.controller.request_generation(
                        output_path=output_path,
                        tabla_dyes_path=tabla_dyes_path,
                    )
                self._gen_res_q.put((job_id, True, None))
            except Exception as e:
                tb = traceback.format_exc()
                self._gen_res_q.put((job_id, False, (str(e), tb)))
if __name__ == "__main__":
    PreviewGUI().mainloop()
