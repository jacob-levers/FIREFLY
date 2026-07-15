"""Parse the bundled ``CHANGELOG.md`` into the landing screen's "Recent updates"
timeline — so it stays in sync with releases automatically instead of a
hand-maintained list.  Pure stdlib (re + file read); no Qt import, unit-testable.

Heading format (date optional, em-dash or hyphen):
    ## v2.76.27 — 27 Jun 2026
The summary is the first bullet's bold lead-in:
    - **Settings nav labels are now consistently left-aligned.** …
"""
from __future__ import annotations

import os
import re
import sys

_HEADING = re.compile(r'^##\s+(v\d[\w.\-]*)\s*(?:[—–-]\s*(.+?))?\s*$', re.M)
_SUMMARY = re.compile(r'-\s+\*\*(.+?)\*\*', re.S)


def parse_recent_updates(text: str, limit: int = 3) -> list:
    """Return the top ``limit`` STABLE version sections as
    ``[{"version", "date", "summary"}]`` (newest first).

    Pre-release headings (a version with a ``-rc`` / ``-dev`` suffix) are
    skipped, so the timeline shows only stable releases — each of which
    consolidates the changes from its whole pre-release series rather than
    fragmenting them across rc entries."""
    out = []
    heads = list(_HEADING.finditer(text or ""))
    for i, m in enumerate(heads):
        version = m.group(1)
        if "-" in version.lstrip("vV"):          # e.g. v2.76.44-rc.1 → skip
            continue
        start = m.end()
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        body = text[start:end]
        sm = _SUMMARY.search(body)
        summary = re.sub(r'\s+', ' ', sm.group(1)).strip() if sm else ""
        out.append({"version": version,
                    "date": (m.group(2) or "").strip(),
                    "summary": summary})
        if len(out) >= limit:
            break
    return out


def _changelog_path() -> "str | None":
    """Locate CHANGELOG.md — bundled next to the frozen app, else the repo root."""
    base = getattr(sys, "_MEIPASS", None)
    if base:
        p = os.path.join(base, "CHANGELOG.md")
        if os.path.isfile(p):
            return p
    here = os.path.dirname(os.path.abspath(__file__))     # …/firefly/ui
    cur = here
    for _ in range(4):                                    # walk up to the repo root
        p = os.path.join(cur, "CHANGELOG.md")
        if os.path.isfile(p):
            return p
        cur = os.path.dirname(cur)
    return None


def recent_updates(limit: int = 3) -> list:
    """Read + parse the bundled changelog; ``[]`` if it can't be found/read."""
    path = _changelog_path()
    if not path:
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return parse_recent_updates(fh.read(), limit)
    except Exception:
        return []
