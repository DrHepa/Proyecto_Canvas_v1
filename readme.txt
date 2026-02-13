Proyecto Canvas - settings.json

Coloca settings.json junto a run_gui_flatdark_pro.py (o junto al .exe cuando empaquetes).

Si la carpeta no es escribible (por ejemplo, Program Files), la app usara:
- Windows: %APPDATA%\ProyectoCanvas\settings.json
- Linux/macOS: ~/.proyecto_canvas/settings.json

El archivo se valida al iniciar:
- valores invalidos se ignoran (se usa el default actual).
- si el JSON esta corrupto, se renombra a settings.bad_YYYYMMDD_HHMMSS.json y se recrea.

Campos:

1) language
  auto | es | en | zh | ru
  - auto no fuerza idioma (usa el que ya gestiona la GUI / user_cache).

2) user_translation_overrides
  Ruta a un JSON con overrides de traduccion. Formatos soportados:
  A) {"msgid": "Texto", ...} (se aplica al idioma actual)
  B) {"es": { ... }, "en": { ... }}

3) defaults (null = no forzar)
  preview_mode: visual | ark_simulation | null
  show_game_object: true | false | null
  use_all_dyes: true | false | null
  best_colors: 1..255 | null
  border_style: none | image | null
  dither_mode: none | palette_fs | palette_ordered | null
  show_advanced: true | false | null

4) advanced_defaults (se aplica cuando Advanced existe)
  external_enabled, external_recursive, external_detect_guid, external_preserve_metadata: true/false/null
  external_max_files: 50..20000|null
  show_dino_tools: true/false/null

5) window
  remember_geometry: true/false (default false)
  geometry: "1100x800+100+80" (si remember_geometry es true)

Notas:
- Los ultimos directorios ya se guardan en user_cache por la GUI.
- Este settings.json no toca layouts (grid/pack) para evitar romper la interfaz.
