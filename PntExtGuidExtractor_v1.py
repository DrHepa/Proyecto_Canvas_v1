from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


# Common GUID formats encountered in ARK save suffixes
_GUID_DASHED_RE = re.compile(
    rb"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
_GUID_HEX32_RE = re.compile(rb"\b[0-9a-fA-F]{32}\b")


def extract_guid_from_bytes(buf: bytes) -> Optional[str]:
    """Best-effort extraction of an ASCII GUID from a bytes buffer."""
    if not buf:
        return None

    m = _GUID_DASHED_RE.search(buf)
    if m:
        try:
            return m.group(0).decode("ascii", errors="ignore")
        except Exception:
            return None

    m = _GUID_HEX32_RE.search(buf)
    if m:
        try:
            return m.group(0).decode("ascii", errors="ignore")
        except Exception:
            return None

    return None


def extract_guid_from_pnt_tail(pnt_path: Path, *, tail_bytes: int = 4096) -> Optional[str]:
    """Reads the tail of a .pnt and tries to find an ASCII GUID."""
    p = Path(pnt_path)
    try:
        size = p.stat().st_size
        n = max(0, min(int(tail_bytes), int(size)))
        with p.open("rb") as f:
            if n:
                f.seek(-n, 2)
                buf = f.read(n)
            else:
                buf = b""
    except Exception:
        return None

    return extract_guid_from_bytes(buf)


def extract_guid_with_offset_from_pnt_tail(pnt_path: Path, *, tail_bytes: int = 4096) -> tuple[Optional[str], Optional[int]]:
    """Reads the tail of a .pnt and tries to find an ASCII GUID, returning (guid, absolute_offset).

    absolute_offset is the byte index in the file where the GUID string starts, if found.
    """
    p = Path(pnt_path)
    try:
        size = int(p.stat().st_size)
        n = max(0, min(int(tail_bytes), size))
        with p.open("rb") as f:
            if n:
                f.seek(-n, 2)
                buf = f.read(n)
                base = size - n
            else:
                buf = b""
                base = 0
    except Exception:
        return (None, None)

    if not buf:
        return (None, None)

    m = _GUID_DASHED_RE.search(buf)
    if m:
        try:
            return (m.group(0).decode("ascii", errors="ignore"), base + m.start(0))
        except Exception:
            return (None, None)

    m = _GUID_HEX32_RE.search(buf)
    if m:
        try:
            return (m.group(0).decode("ascii", errors="ignore"), base + m.start(0))
        except Exception:
            return (None, None)

    return (None, None)

