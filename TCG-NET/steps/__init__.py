"""Pipeline step functions.

Each step contains a small, testable function that:
- loads its required inputs
- runs the core processing logic
- saves its artifacts

The original scripts (main.py / normalize_main.py / ...) now simply call
these functions, so the code logic is no longer duplicated across files.
"""
