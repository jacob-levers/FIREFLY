"""The landing "Recent updates" timeline is parsed from CHANGELOG.md."""
from firefly.ui.changelog import parse_recent_updates, recent_updates

_SAMPLE = """# Changelog

## v2.76.27 — 27 Jun 2026

### Fixed

- **Settings nav labels are now consistently left-aligned.** The menu text was
  being centred within its column.

## v2.76.26 — 27 Jun 2026

### Changed

- **Redesigned home screen.** Two-column layout.

## v2.76.25

### Added

- **A real progress bar while generating a full report.** It fills as it loads.

## v2.76.24 — 26 Jun 2026
- **Older entry.** Should be dropped at limit 3.
"""


def test_parse_version_date_summary():
    out = parse_recent_updates(_SAMPLE, limit=3)
    assert len(out) == 3
    assert out[0] == {"version": "v2.76.27", "date": "27 Jun 2026",
                      "summary": "Settings nav labels are now consistently left-aligned."}
    assert out[1]["version"] == "v2.76.26" and out[1]["summary"] == "Redesigned home screen."
    # date is optional — a heading without one still parses, just blank
    assert out[2]["version"] == "v2.76.25" and out[2]["date"] == ""
    assert out[2]["summary"].startswith("A real progress bar")


def test_limit_caps_results():
    assert len(parse_recent_updates(_SAMPLE, limit=2)) == 2
    assert len(parse_recent_updates("# Changelog\n\n(no versions yet)\n")) == 0


def test_real_changelog_parses():
    # the bundled/repo CHANGELOG must yield well-formed entries (newest first)
    out = recent_updates(3)
    assert len(out) == 3
    for e in out:
        assert e["version"].startswith("v") and e["summary"]
