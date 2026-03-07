#!/usr/bin/env python3
"""Detect and optionally remove hidden/bidirectional unicode chars from Python files."""

from __future__ import annotations

import argparse
import unicodedata
from pathlib import Path

DEFAULT_EXCLUDES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "artifacts",
    ".mypy_cache",
    ".ruff_cache",
}
BIDI_RANGES = (
    (0x202A, 0x202E),
    (0x2066, 0x2069),
)
ZERO_WIDTH = {0x200B, 0x200C, 0x200D, 0xFEFF}
ALLOWED_CONTROL = {"\n", "\r", "\t"}


def is_bidi(codepoint: int) -> bool:
    return any(start <= codepoint <= end for start, end in BIDI_RANGES)


def is_disallowed_char(ch: str) -> bool:
    codepoint = ord(ch)
    category = unicodedata.category(ch)
    if codepoint in ZERO_WIDTH:
        return True
    if is_bidi(codepoint):
        return True
    if category == "Cf" and ch not in ALLOWED_CONTROL:
        return True
    return False


def iter_py_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*.py"):
        if any(part in DEFAULT_EXCLUDES for part in path.parts):
            continue
        files.append(path)
    return sorted(files)


def find_issues(content: str) -> list[tuple[int, int, str, str]]:
    issues: list[tuple[int, int, str, str]] = []
    for line_no, line in enumerate(content.splitlines(keepends=True), start=1):
        for col_no, ch in enumerate(line, start=1):
            if is_disallowed_char(ch):
                issues.append(
                    (line_no, col_no, f"U+{ord(ch):04X}", unicodedata.name(ch, "UNKNOWN"))
                )
    return issues


def clean_content(content: str) -> str:
    return "".join(ch for ch in content if not is_disallowed_char(ch))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check", action="store_true", help="Only check; non-zero exit if issues found"
    )
    parser.add_argument("--fix", action="store_true", help="Remove problematic characters in-place")
    parser.add_argument("--root", type=Path, default=Path("."), help="Repository root")
    args = parser.parse_args()

    if not args.check and not args.fix:
        parser.error("specify at least one of --check or --fix")

    root = args.root.resolve()
    found_any = False

    for path in iter_py_files(root):
        content = path.read_text(encoding="utf-8")
        issues = find_issues(content)
        if not issues:
            continue

        found_any = True
        rel = path.relative_to(root)
        print(f"{rel}:")
        for line_no, col_no, code, name in issues:
            print(f"  L{line_no}:C{col_no} {code} {name}")

        if args.fix:
            cleaned = clean_content(content)
            if cleaned != content:
                path.write_text(cleaned, encoding="utf-8")
                print("  fixed")

    if found_any and args.check:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
