"""Shared helpers for tests that read QML source."""
import re

# One pass, leftmost match wins: a // inside a string is consumed as part of the
# string, and a quote inside a comment as part of the comment.  Stripping
# comments first and strings second cut "https://…" mid-string and let the
# orphaned quote swallow code up to the next quote — which hid two-thirds of
# AnalysisTab.qml from the first version of these checks.
_TOKEN = re.compile(r'"(?:\\.|[^"\\\n])*"'
                    r"|'(?:\\.|[^'\\\n])*'"
                    r'|`(?:\\.|[^`\\])*`'
                    r'|/\*.*?\*/'
                    r'|//[^\n]*', re.S)


def strip_qml(text):
    """QML source with comments and string literals blanked, line numbers kept."""
    def rep(m):
        s = m.group(0)
        if s.startswith("//"):
            return ""
        return '""' + "\n" * s.count("\n")
    return _TOKEN.sub(rep, text)


def block_end(text, open_brace):
    """Index of the brace closing the one at ``open_brace``."""
    depth = 0
    for i in range(open_brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return len(text)
