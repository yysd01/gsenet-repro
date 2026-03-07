from __future__ import annotations

from pathlib import Path

from tools.sanitize_unicode import fix_text, run_check, run_fix, scan_text


def test_nonstandard_newline_is_reported_and_fixed(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.py"
    file_path.write_text("a = 1\u2028b = 2\n", encoding="utf-8")

    assert run_check(tmp_path, {".py"}) == 1

    run_fix(tmp_path, {".py"})
    fixed = file_path.read_text(encoding="utf-8")
    assert "\u2028" not in fixed
    assert "a = 1\nb = 2\n" == fixed
    assert run_check(tmp_path, {".py"}) == 0


def test_bidi_is_reported_and_removed(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.py"
    file_path.write_text("name = 'x\u202ey'\n", encoding="utf-8")

    assert run_check(tmp_path, {".py"}) == 1

    run_fix(tmp_path, {".py"})
    fixed = file_path.read_text(encoding="utf-8")
    assert "\u202e" not in fixed
    assert "name = 'xy'\n" == fixed
    assert run_check(tmp_path, {".py"}) == 0


def test_check_fails_on_cr(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.py"
    file_path.write_bytes(b"a\rb\n")

    issues = scan_text(file_path.read_bytes().decode("utf-8"))

    assert any(issue.codepoint == "U+000D" for issue in issues)
    assert run_check(tmp_path, {".py"}) == 1


def test_fix_normalizes_cr_to_lf(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.py"
    file_path.write_bytes(b"a\r\nb\r\nc\r")

    run_fix(tmp_path, {".py"})

    fixed = file_path.read_text(encoding="utf-8")
    assert "\r" not in fixed
    assert fixed.splitlines() == ["a", "b", "c"]
    assert len(fixed.splitlines()) > 1
    assert run_check(tmp_path, {".py"}) == 0


def test_fix_drops_disallowed_cc_control_char() -> None:
    fixed = fix_text("a\x07b\n")

    assert fixed == "ab\n"
