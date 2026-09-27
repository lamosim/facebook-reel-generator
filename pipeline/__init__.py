"""Trip-Footage → Facebook Reel Generator pipeline package."""
import sys as _sys

# Log lines carry arrows, curly quotes and ellipses. A Windows console or a piped stdout can
# default to a legacy code page (cp1252) and raise on them - force UTF-8 and never crash on output.
for _stream in (_sys.stdout, _sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # a replaced/closed stream: leave it alone
        pass
