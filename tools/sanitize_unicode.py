#!/usr/bin/env python3
"""Detect and optionally sanitize hidden Unicode text hazards."""

from __future__ import annotations

import argparse
import unicodedata
from dataclasses import dataclass
from pathlib import Path

DEFAULT_INCLUDE_EXTS = {".py", ".toml", ".yml", ".yaml", ".md"}
DEFAULT_EXCLUDES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "artifacts",
    "build",
    "dist",
    ".mypy_cache",
    ".ruff_cache",
}
BIDI_RANGES = ((0x202A, 0x202E), (0x2066, 0x2069))
ZERO_WIDTH = {0x200B, 0x200C, 0x200D, 0xFEFF}
NONSTANDARD_NEWLINE_MAP = {
    0x2028: "\n",  # LINE SEPARATOR
    0x2029: "\n",  # PARAGRAPH SEPARATOR
    0x0085: "\n",  # NEXT LINE (NEL)
    0x000B: "\n",  # VERTICAL TAB
    0x000C: "\n",  # FORM FEED
}
ALLOWED_CONTROL_CHARS = {"\n", "\t"}


@dataclass(frozen=True)
class Issue:
    line: int
    column: int
    codepoint: str
    name: str
    category: str
    bidi: str


def is_bidi(codepoint: int) -> bool:
    return any(start <= codepoint <= end for start, end in BIDI_RANGES)


def is_disallowed_control(ch: str) -> bool:
    category = unicodedata.category(ch)
    if category == "Cf":
        return True
    if category != "Cc":
        return False
    return ch not in ALLOWED_CONTROL_CHARS


def is_forbidden_char(ch: str) -> bool:
    cp = ord(ch)
    if cp in NONSTANDARD_NEWLINE_MAP:
        return True
    if cp in ZERO_WIDTH:
        return True
    if is_bidi(cp):
        return True
    if ch in ALLOWED_CONTROL_CHARS:
        return False
    return is_disallowed_control(ch)


def iter_target_files(root: Path, include_exts: set[str]) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in include_exts:
            continue
        if any(part in DEFAULT_EXCLUDES for part in path.parts):
            continue
        files.append(path)
    return sorted(files)


def scan_text(text: str) -> list[Issue]:
    issues: list[Issue] = []
    line = 1
    column = 1
    for ch in text:
        if is_forbidden_char(ch):
            issues.append(
                Issue(
                    line=line,
                    column=column,
                    codepoint=f"U+{ord(ch):04X}",
                    name=unicodedata.name(ch, "UNKNOWN"),
                    category=unicodedata.category(ch),
                    bidi=unicodedata.bidirectional(ch),
                )
            )
        if ch == "\n" or ord(ch) in NONSTANDARD_NEWLINE_MAP:
            line += 1
            column = 1
        else:
            column += 1
    return issues


def fix_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    out: list[str] = []
    for ch in normalized:
        cp = ord(ch)
        if cp in NONSTANDARD_NEWLINE_MAP:
            out.append("\n")
            continue
        if ch in {"\n", "\t"}:
            out.append(ch)
            continue
        if cp in ZERO_WIDTH or is_bidi(cp) or is_disallowed_control(ch):
            continue
        out.append(ch)
    return "".join(out)


def parse_include_exts(raw: str) -> set[str]:
    parsed = {item.strip() for item in raw.split(",") if item.strip()}
    return {ext if ext.startswith(".") else f".{ext}" for ext in parsed}


def read_text_preserve_newlines(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as file:
        return file.read()


def scan_path(path: Path) -> list[Issue]:
    return scan_text(read_text_preserve_newlines(path))


def format_issue(issue: Issue) -> str:
    return (
        f"L{issue.line}:C{issue.column} {issue.codepoint} {issue.name} "
        f"(category={issue.category}, bidi={issue.bidi})"
    )


def run_check(root: Path, include_exts: set[str], verbose: bool = False) -> int:
    found_any = False
    for path in iter_target_files(root, include_exts):
        issues = scan_path(path)
        if not issues:
            continue
        found_any = True
        rel = path.relative_to(root)
        print(f"{rel} ({len(issues)} issue(s))")
        if verbose:
            for issue in issues:
                print(f"  {format_issue(issue)}")
    return 1 if found_any else 0


def run_fix(root: Path, include_exts: set[str], verbose: bool = False) -> int:
    changed = 0
    issue_count = 0
    for path in iter_target_files(root, include_exts):
        content = read_text_preserve_newlines(path)
        issues = scan_text(content)
        if not issues:
            continue
        issue_count += len(issues)
        fixed = fix_text(content)
        if fixed != content:
            path.write_text(fixed, encoding="utf-8")
            changed += 1
        rel = path.relative_to(root)
        print(f"fixed: {rel} ({len(issues)} issue(s))")
        if verbose:
            for issue in issues:
                print(f"  {format_issue(issue)}")
    print(f"summary: {changed} file(s) changed, {issue_count} issue(s) processed")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check", action="store_true", help="Only check; non-zero exit if issues found"
    )
    parser.add_argument("--fix", action="store_true", help="Fix problematic characters in-place")
    parser.add_argument("--root", type=Path, default=Path("."), help="Repository root")
    parser.add_argument(
        "--include-ext",
        default=",".join(sorted(DEFAULT_INCLUDE_EXTS)),
        help="Comma-separated extension list (for example: .py,.toml,.yml,.yaml,.md)",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-issue detail")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.check and not args.fix:
        parser.error("specify at least one of --check or --fix")

    root = args.root.resolve()
    include_exts = parse_include_exts(args.include_ext)

    if args.fix:
        run_fix(root, include_exts, verbose=args.verbose)

    if args.check:
        return run_check(root, include_exts, verbose=args.verbose)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
