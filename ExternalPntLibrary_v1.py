from __future__ import annotations

import os
import re
import time
import json
import math
import struct
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

from PntIO import peek_pnt_info
from PntExtGuidExtractor_v1 import extract_guid_from_pnt_tail, extract_guid_with_offset_from_pnt_tail

# ------------------------------------------------------------
# Blueprint extraction (best-effort)
# ------------------------------------------------------------

_BP_PATTERNS = [
    r"[A-Za-z0-9_]+_Character_BP_C",
    r"StructureBP_[A-Za-z0-9_]+_C",
    r"Sign_[A-Za-z0-9_]+_C",
    r"[A-Za-z0-9_]+_BP_C",
]
_BP_RE = re.compile("|".join(f"(?:{p})" for p in _BP_PATTERNS))
# Lookahead variant to allow *overlapping* matches.
# This is critical for filenames that prepend markers before the real blueprint,
# e.g. 'blue_to_red_Raptor_Character_BP_C' where the intended blueprint is
# 'Raptor_Character_BP_C'. A normal finditer would only return the greedy, full
# match; the lookahead lets us consider suffix candidates and choose deterministically.
_BP_LOOKAHEAD_RE = re.compile(r"(?=(" + "|".join(_BP_PATTERNS) + r"))")
_BP_BYTES_RE = re.compile(rb"|".join(
    [rb"(?:[A-Za-z0-9_]+_Character_BP_C)",
     rb"(?:StructureBP_[A-Za-z0-9_]+_C)",
     rb"(?:Sign_[A-Za-z0-9_]+_C)",
     rb"(?:[A-Za-z0-9_]+_BP_C)"]
))

_GUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
_HINT_WH_RE = re.compile(r"(\d{2,4})x(\d{2,4})")

def extract_blueprint_from_filename(stem: str) -> Optional[str]:
    if not stem:
        return None
    # Use lookahead to allow overlapping candidates.
    matches = list(_BP_LOOKAHEAD_RE.finditer(stem))
    if not matches:
        return None

    def _score(m: re.Match) -> Tuple[int, int, int]:
        s = m.group(1)
        bonus = 0
        if s.endswith("_Character_BP_C"):
            bonus += 10_000
        elif s.startswith("StructureBP_"):
            bonus += 5_000
        elif s.startswith("Sign_"):
            bonus += 4_000
        elif s.endswith("_BP_C"):
            bonus += 1_000

        # Prefer candidates that start at a later position (strip prefixes like 'blue_to_red_').
        bonus += int(m.start())

        # Prefer blueprint-like capitalization when ties exist.
        try:
            if s and s[0].isupper():
                bonus += 250
        except Exception:
            pass

        # Slight preference for shorter matches when otherwise equal.
        return (bonus, -(len(s)), int(m.start()))

    best = max(matches, key=_score)
    return best.group(1)

def extract_blueprint_from_bytes(buf: bytes) -> Optional[str]:
    if not buf:
        return None
    matches = list(_BP_BYTES_RE.finditer(buf))
    if not matches:
        return None

    def _score(m: re.Match) -> Tuple[int, int]:
        s = m.group(0)
        bonus = 0
        if s.endswith(b"_Character_BP_C"):
            bonus = 10_000
        elif s.startswith(b"StructureBP_"):
            bonus = 5_000
        elif s.startswith(b"Sign_"):
            bonus = 4_000
        return (bonus + (m.end() - m.start()), m.end())

    matches.sort(key=_score, reverse=True)
    try:
        return matches[0].group(0).decode("ascii", errors="ignore")
    except Exception:
        return None

def _extract_blueprint_light(p: Path, *, head_bytes: int = 256_000, tail_bytes: int = 256_000) -> Optional[str]:
    try:
        size = int(p.stat().st_size)
        with p.open("rb") as f:
            head = f.read(min(head_bytes, size))
            if size > tail_bytes:
                f.seek(-tail_bytes, 2)
                tail = f.read(tail_bytes)
            else:
                tail = b""
        return extract_blueprint_from_bytes(head + b"\n" + tail)
    except Exception:
        return None

def _u32(b: bytes, off: int) -> Tuple[int, int]:
    return struct.unpack_from("<I", b, off)[0], off + 4

def _parse_hint_wh(name: str) -> Tuple[Optional[int], Optional[int]]:
    if not name:
        return None, None
    m = _HINT_WH_RE.search(name)
    if not m:
        return None, None
    try:
        return int(m.group(1)), int(m.group(2))
    except Exception:
        return None, None

def _factor_pairs(n: int, limit_dim: int = 8192) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    if n <= 0:
        return out
    r = int(math.isqrt(n))
    for i in range(1, r + 1):
        if n % i == 0:
            j = n // i
            if i <= limit_dim and j <= limit_dim:
                out.append((i, j))
                if i != j:
                    out.append((j, i))
    return out

# ------------------------------------------------------------
# ASA (UE5 / Ascended) "GUID-header" .pnt parser
# ------------------------------------------------------------

def try_parse_asa_guid_header_pnt(p: Path) -> Optional[Dict[str, Any]]:
    """
    ASA .pnt files (cache/MyPaintings 'EXT...') often start with:
      GUID_ASCII + \\0
      u32 version
      u32 name_len + name (utf-8)
      u32 blueprint_len + blueprint (utf-8)   # can be 0 in numeric cache
      u32 a1, a2, a3, raster_len
      raster bytes (raster_len)
      tail (unknown)
    """
    try:
        data = p.read_bytes()
    except Exception:
        return None

    nul = data.find(b"\x00")
    if nul <= 0 or nul > 80:
        return None
    try:
        guid = data[:nul].decode("ascii", errors="strict")
    except Exception:
        return None
    if not _GUID_RE.match(guid):
        return None

    off = nul + 1
    if off + 4 * 3 > len(data):
        return None

    version, off = _u32(data, off)
    name_len, off = _u32(data, off)
    if name_len < 0 or off + name_len > len(data):
        return None
    name = data[off:off + name_len].decode("utf-8", errors="replace").rstrip("\x00").strip()
    off += name_len

    bp_len, off = _u32(data, off)
    if bp_len < 0 or off + bp_len > len(data):
        return None
    blueprint = data[off:off + bp_len].decode("utf-8", errors="replace").rstrip("\x00").strip()
    off += bp_len

    if off + 16 > len(data):
        return None
    a1, off = _u32(data, off)
    a2, off = _u32(data, off)
    a3, off = _u32(data, off)
    raster_len, off = _u32(data, off)

    if raster_len <= 0 or off + raster_len > len(data):
        # some files might be truncated; treat as unknown
        return None

    return {
        "guid": guid,
        "version": int(version),
        "internal_name": name,
        "blueprint": blueprint,
        "a1": int(a1),
        "a2": int(a2),
        "a3": int(a3),
        "raster_len": int(raster_len),
        "raster_off": int(off),
        "file_size": len(data),
    }

# ------------------------------------------------------------
# Dynamic canvas size filtering (deterministic, no GUI "magic")
# ------------------------------------------------------------

# Nominal families observed in ASA Dynamic Canvas UX (used only for classification hints):
# - Rectangles: 128x{256,512,968} (and swapped)
# - Square max: 712x712
_DYN_NOMINAL_PAIRS = [
    (128, 256),
    (128, 512),
    (128, 968),
    (712, 712),
]

def _near_pair(w: int, h: int, W: int, H: int, tol: int) -> bool:
    return (abs(w - W) <= tol and abs(h - H) <= tol) or (abs(w - H) <= tol and abs(h - W) <= tol)

def _looks_like_dynamic_family(pairs: List[Tuple[int, int]]) -> bool:
    """Conservative detector for numeric cache files with missing blueprint.

    We only classify as dynamic if at least one factor-pair is close to known ASA dynamic families.
    """
    for (w, h) in pairs:
        for (W, H) in _DYN_NOMINAL_PAIRS:
            tol = 40 if (W, H) == (712, 712) else 16
            if _near_pair(w, h, W, H, tol):
                return True
    return False

def _filter_dynamic_candidates(pairs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Dynamic candidates should be *bounded* but never filtered to a nominal family.

    The game (ASA) can produce sizes outside the few nominal UI presets because the serialized raster
    can include padding/packing artifacts. We therefore keep all factor pairs in a broad, sane range.
    """
    if not pairs:
        return []
    # broad bounds first (eliminate absurd shapes)
    bounded = [(w, h) for (w, h) in pairs if 64 <= w <= 2000 and 64 <= h <= 2000]
    if not bounded:
        bounded = pairs[:]
    # de-dup preserving order
    seen = set()
    dedup = []
    for wh in bounded:
        if wh in seen:
            continue
        seen.add(wh)
        dedup.append(wh)
    return dedup

def _dyn_nominal_distance(w: int, h: int) -> int:
    best = 10**9
    for (W, H) in _DYN_NOMINAL_PAIRS:
        d1 = abs(w - W) + abs(h - H)
        d2 = abs(w - H) + abs(h - W)
        best = min(best, d1, d2)
    return int(best)

def _rank_pairs(pairs: List[Tuple[int, int]], hint_w: Optional[int], hint_h: Optional[int], *, prefer_nominal: bool = False) -> List[Tuple[int, int]]:
    if not pairs:
        return []

    def score(w: int, h: int) -> int:
        s = 0
        if hint_w is not None and hint_h is not None:
            s += abs(w - hint_w) + abs(h - hint_h)
        else:
            if prefer_nominal:
                # soft preference towards known UI families, but never filters them out
                s += _dyn_nominal_distance(w, h)
            # very soft square-ish preference for stability
            s += int(abs(w - h) * 0.05)
        return int(s)

    return sorted(pairs, key=lambda wh: (score(wh[0], wh[1]), max(wh[0], wh[1]), abs(wh[0] - wh[1])))

def _best_and_candidates_from_raster(
    *,
    raster_len: int,
    internal_name: str,
    blueprint: str,
    a1: int,
    a2: int,
) -> Tuple[Optional[int], Optional[int], List[Tuple[int, int]]]:
    """Returns (best_w, best_h, ordered_candidates).

    For dynamic blueprints we *bound* candidates but do not filter to nominal families.
    """
    # strong shortcut: some cache files may expose exact dims in a1/a2
    if a1 > 0 and a2 > 0 and a1 * a2 == raster_len and a1 <= 8192 and a2 <= 8192:
        return a1, a2, [(a1, a2)]

    pairs = _factor_pairs(raster_len, limit_dim=8192)
    if not pairs:
        return None, None, []

    hint_w, hint_h = _parse_hint_wh(internal_name)
    is_dynamic = ("Canvas_Dynamic" in (blueprint or "")) or (blueprint == "StructureBP_Canvas_Dynamic_C")

    # Conservative detection for numeric cache missing blueprint.
    if (not is_dynamic) and (not (blueprint or "").strip()):
        try:
            if str(internal_name or "").strip().isdigit() and _looks_like_dynamic_family(pairs):
                is_dynamic = True
        except Exception:
            pass

    if is_dynamic:
        pairs = _filter_dynamic_candidates(pairs)
        pairs = _rank_pairs(pairs, hint_w, hint_h, prefer_nominal=True)
    else:
        # generic: keep bounded and prefer square-ish; use hint if present
        bounded = [(w, h) for (w, h) in pairs if 32 <= w <= 4096 and 32 <= h <= 4096]
        if not bounded:
            bounded = pairs[:]

        def score2(w: int, h: int) -> int:
            s = 0
            if hint_w is not None and hint_h is not None:
                s += abs(w - hint_w) + abs(h - hint_h)
            else:
                s += abs(w - h)
            return int(s)

        bounded = sorted(bounded, key=lambda wh: (score2(wh[0], wh[1]), max(wh[0], wh[1])))
        pairs = bounded

    if not pairs:
        return None, None, []
    best = pairs[0]
    return best[0], best[1], pairs

# ------------------------------------------------------------
# Extended GUID registry (legacy EXT tail-guid)
# ------------------------------------------------------------

_REG_CACHE = None

def _load_registry() -> dict:
    global _REG_CACHE
    if _REG_CACHE is not None:
        return _REG_CACHE

    base = Path(__file__).resolve().parent / "Templates" / "game_extended_guid_registry.json"
    try:
        reg = json.loads(base.read_text(encoding="utf-8"))
    except Exception:
        reg = {}

    footer_map = {}
    for section in ("dynamic_codes", "drawing_sheet_codes"):
        sec = reg.get(section) or {}
        for _, ent in sec.items():
            fh = (ent.get("footer24_hex") or "").lower()
            if fh and len(fh) >= 48:
                footer_map[fh] = ent

    _REG_CACHE = {"raw": reg, "footer_map": footer_map}
    return _REG_CACHE

def _read_footer24(p: Path, guid_offset: int) -> Optional[bytes]:
    try:
        if guid_offset is None:
            return None
        start = max(0, int(guid_offset) - 24)
        with p.open("rb") as f:
            f.seek(start, 0)
            buf = f.read(24)
        return buf if len(buf) == 24 else None
    except Exception:
        return None

def _dims_from_entry(ent: dict) -> Tuple[Optional[int], Optional[int], list]:
    cands = ent.get("candidates_128_968") or []
    norm = []
    for c in cands:
        try:
            w, h = int(c[0]), int(c[1])
            if w > 0 and h > 0:
                norm.append((w, h))
        except Exception:
            continue
    if len(norm) == 1:
        return norm[0][0], norm[0][1], norm
    return None, None, norm

# ------------------------------------------------------------
# Public scan API
# ------------------------------------------------------------

def scan_pnts(
    root: Path,
    *,
    recursive: bool = True,
    max_files: int = 2000,
    detect_guid: bool = True,
    guid_tail_bytes: int = 4096,
    time_limit_s: float = 6.0,
    max_walk_files: int = 200_000,
    max_walk_dirs: int = 20_000,
) -> dict:
    """
    Scans a directory for .pnt files.

    Kinds:
      - H20: our header20 writer output
      - ASA: game GUID-header (MyPaintings EXT... + ServerPaintingsCache numerics)
      - EXT: legacy tail-guid (registry)
      - UNK: unknown
    """
    root = Path(root)
    items: list[dict] = []

    if not root.exists() or not root.is_dir():
        return {"items": [], "elapsed_s": 0.0, "walk_files": 0, "walk_dirs": 0, "truncated": False, "reason": "not_a_dir"}

    max_files = max(1, int(max_files))
    t0 = time.perf_counter()
    walk_files = 0
    walk_dirs = 0
    truncated = False
    reason = None

    def _should_stop() -> bool:
        nonlocal truncated, reason
        if len(items) >= max_files:
            truncated = True
            reason = "max_pnt_files"
            return True
        if (time.perf_counter() - t0) >= float(time_limit_s):
            truncated = True
            reason = "time_limit"
            return True
        if walk_files >= int(max_walk_files):
            truncated = True
            reason = "max_walk_files"
            return True
        if walk_dirs >= int(max_walk_dirs):
            truncated = True
            reason = "max_walk_dirs"
            return True
        return False

    def _add(p: Path):
        try:
            items.append(_inspect_one(p, detect_guid=detect_guid, guid_tail_bytes=guid_tail_bytes))
        except Exception:
            items.append({"path": str(p), "name": p.stem, "kind": "UNK", "is_header20": False})

    if recursive:
        for dirpath, _, filenames in os.walk(root):
            walk_dirs += 1
            if _should_stop():
                break
            for fn in filenames:
                walk_files += 1
                if _should_stop():
                    break
                if not fn.lower().endswith(".pnt"):
                    continue
                _add(Path(dirpath) / fn)
            if _should_stop():
                break
    else:
        try:
            for p in root.iterdir():
                walk_files += 1
                if _should_stop():
                    break
                if p.is_file() and p.name.lower().endswith(".pnt"):
                    _add(p)
        except Exception:
            pass

    # Post-pass: enrich ASA cache numerics missing blueprint using GUID match.
    guid_to_bp: Dict[str, str] = {}
    for it in items:
        g = it.get("guid")
        bp = (it.get("blueprint") or "").strip()
        if g and bp:
            guid_to_bp[g] = bp
    for it in items:
        if it.get("kind") == "ASA":
            g = it.get("guid")
            if g and not (it.get("blueprint") or "").strip():
                bp = guid_to_bp.get(g)
                if bp:
                    it["blueprint"] = bp
                    # re-rank candidates with dynamic hint enabled
                    try:
                        w, h, pairs = _best_and_candidates_from_raster(
                            raster_len=int(it.get("raster_len", 0)),
                            internal_name=str(it.get("internal_name") or it.get("name") or ""),
                            blueprint=str(bp),
                            a1=int(it.get("a1", 0) or 0),
                            a2=int(it.get("a2", 0) or 0),
                        )
                        if pairs:
                            it["candidates"] = [{"w": ww, "h": hh} for (ww, hh) in pairs[:60]]
                            if w and h:
                                it["best_w"], it["best_h"] = int(w), int(h)
                    except Exception:
                        pass

    return {
        "items": items,
        "elapsed_s": float(time.perf_counter() - t0),
        "walk_files": int(walk_files),
        "walk_dirs": int(walk_dirs),
        "truncated": bool(truncated),
        "reason": reason,
    }

def _inspect_one(p: Path, *, detect_guid: bool, guid_tail_bytes: int) -> dict:
    p = Path(p)
    item: dict = {"path": str(p), "name": p.stem}

    # 1) header20?
    info = peek_pnt_info(p)
    if bool(info.get("is_header20", False)):
        item["kind"] = "H20"
        item["is_header20"] = True
        item.update({
            "width": int(info.get("width", 0)),
            "height": int(info.get("height", 0)),
            "has_suffix": bool(info.get("has_suffix", False)),
            "suffix_len": int(info.get("suffix_len", 0)),
            "file_size": int(info.get("file_size", 0)),
        })
        # blueprint best-effort from filename, then bytes
        bp = extract_blueprint_from_filename(p.stem) or _extract_blueprint_light(p)
        if bp:
            item["blueprint"] = bp
        if detect_guid and item.get("has_suffix"):
            guid = extract_guid_from_pnt_tail(p, tail_bytes=guid_tail_bytes)
            if guid:
                item["guid"] = guid
        return item

    # 2) ASA GUID-header?
    asa = try_parse_asa_guid_header_pnt(p)
    if asa:
        item["kind"] = "ASA"
        item["is_header20"] = False
        item["guid"] = asa["guid"]
        item["internal_name"] = asa["internal_name"]
        item["name"] = str(asa["internal_name"] or p.stem)  # show internal name in UI
        item["blueprint"] = (asa.get("blueprint") or "").strip()
        item["version"] = int(asa.get("version", 0))
        item["a1"] = int(asa.get("a1", 0))
        item["a2"] = int(asa.get("a2", 0))
        item["a3"] = int(asa.get("a3", 0))
        item["raster_len"] = int(asa.get("raster_len", 0))
        item["file_size"] = int(asa.get("file_size", 0))

        # If blueprint missing, best-effort extract from bytes (bounded)
        if not item["blueprint"]:
            bp = _extract_blueprint_light(p)
            if bp:
                item["blueprint"] = bp

        best_w, best_h, pairs = _best_and_candidates_from_raster(
            raster_len=item["raster_len"],
            internal_name=str(item.get("internal_name") or ""),
            blueprint=str(item.get("blueprint") or ""),
            a1=item.get("a1", 0) or 0,
            a2=item.get("a2", 0) or 0,
        )
        if pairs:
            item["candidates"] = [{"w": ww, "h": hh} for (ww, hh) in pairs[:60]]
            if best_w and best_h:
                item["best_w"], item["best_h"] = int(best_w), int(best_h)
        return item

    # 3) Legacy EXT tail-guid?
    guid = None
    guid_off = None
    if detect_guid:
        guid, guid_off = extract_guid_with_offset_from_pnt_tail(p, tail_bytes=guid_tail_bytes)
    if guid:
        item["kind"] = "EXT"
        item["is_header20"] = False
        item["guid"] = guid
        item["guid_offset"] = int(guid_off) if guid_off is not None else None

        footer24 = _read_footer24(p, int(guid_off) if guid_off is not None else None)
        if footer24:
            fh = footer24.hex()
            item["footer24_hex"] = fh
            reg = _load_registry()
            ent = reg.get("footer_map", {}).get(fh.lower())
            if ent:
                item["class_name"] = ent.get("class_name")
                w, h, cands = _dims_from_entry(ent)
                if w and h:
                    item["width"], item["height"] = int(w), int(h)
                if cands:
                    item["candidates"] = [{"w": ww, "h": hh} for (ww, hh) in cands]

        bp = _extract_blueprint_light(p)
        if bp:
            item["blueprint"] = bp
        elif item.get("class_name"):
            item["blueprint"] = str(item["class_name"])
        return item

    item["kind"] = "UNK"
    item["is_header20"] = False
    return item
