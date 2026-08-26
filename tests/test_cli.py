from __future__ import annotations

from pathlib import Path

import pytest

from security_monitor import __version__
from security_monitor.cli import main


def test_help_exits_zero() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_version_matches_package(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ == "0.2.0"
    assert __version__ in capsys.readouterr().out


def test_safe_mode_flags_in_help(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--safe-mode" in help_text
    assert "--no-safe-mode" in help_text


def test_check_and_init(tmp_path: Path, capsys) -> None:
    dest = tmp_path / "config.yaml"
    assert main(["init", "-o", str(dest)]) == 0
    assert dest.is_file()
    # Example cameras are placeholders; check should still parse.
    assert main(["check", "--config", str(dest)]) == 0
    out = capsys.readouterr().out
    assert "Front Door" in out
    assert "2x2" in out


def test_init_refuses_overwrite(tmp_path: Path) -> None:
    dest = tmp_path / "config.yaml"
    dest.write_text("already exists\n", encoding="utf-8")
    assert main(["init", "-o", str(dest)]) == 1
    assert dest.read_text(encoding="utf-8") == "already exists\n"
    assert main(["init", "--force", "-o", str(dest)]) == 0
    assert "cameras:" in dest.read_text(encoding="utf-8")
