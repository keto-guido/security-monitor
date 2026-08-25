"""Command-line entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from security_monitor import __version__
from security_monitor.config import (
    AppConfig,
    ConfigError,
    demo_config,
    example_config_text,
    load_config,
)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="security-monitor",
        description="Display multiple IP camera feeds (RTSP/RTP) in a grid window.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.set_defaults(handler=cmd_run)

    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="Open the camera mosaic (default)")
    _add_run_flags(run, suppress_defaults=True)
    run.set_defaults(handler=cmd_run)

    demo = sub.add_parser("demo", help="Open a synthetic 4-pane mosaic (no cameras required)")
    demo.add_argument("--columns", type=int, default=2)
    demo.add_argument("--rows", type=int, default=2)
    demo.add_argument("--fullscreen", action="store_true")
    demo.set_defaults(handler=cmd_demo)

    check = sub.add_parser("check", help="Validate config and print resolved cameras")
    check.add_argument("--config", "-c", default=argparse.SUPPRESS, help="Path to config.yaml")
    check.set_defaults(handler=cmd_check)

    init = sub.add_parser("init", help="Write config.yaml from the bundled example")
    init.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing config.yaml",
    )
    init.add_argument(
        "--output",
        "-o",
        default="config.yaml",
        help="Destination path (default: ./config.yaml)",
    )
    init.set_defaults(handler=cmd_init)

    # Top-level flags so `security-monitor --config x.yaml` works without `run`.
    _add_run_flags(parser)
    return parser


def _add_run_flags(parser: argparse.ArgumentParser, *, suppress_defaults: bool = False) -> None:
    extra = {"default": argparse.SUPPRESS} if suppress_defaults else {}
    parser.add_argument("--config", "-c", help="Path to config.yaml", **extra)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Ignore config and show synthetic demo feeds",
        **extra,
    )
    parser.add_argument(
        "--fullscreen",
        action="store_true",
        help="Start in fullscreen",
        **extra,
    )
    parser.add_argument("--columns", type=int, help="Override grid columns", **extra)
    parser.add_argument("--rows", type=int, help="Override grid rows", **extra)


def cmd_run(args: argparse.Namespace) -> int:
    if getattr(args, "demo", False):
        return cmd_demo(args)
    config = load_config(getattr(args, "config", None))
    _apply_overrides(config, args)
    from security_monitor.mosaic import run_monitor

    return run_monitor(config)


def cmd_demo(args: argparse.Namespace) -> int:
    columns = 2 if getattr(args, "columns", None) is None else args.columns
    rows = 2 if getattr(args, "rows", None) is None else args.rows
    if columns < 1 or rows < 1:
        raise ConfigError("columns and rows must be >= 1")
    config = demo_config(columns=columns, rows=rows)
    if getattr(args, "fullscreen", False):
        config.display.fullscreen = True
    from security_monitor.mosaic import run_monitor

    print("Demo mode — synthetic feeds, no network cameras.")
    return run_monitor(config)


def cmd_check(args: argparse.Namespace) -> int:
    config = load_config(getattr(args, "config", None))
    path = config.path
    print(f"Config: {path}")
    d = config.display
    print(
        f"Display: {d.columns}x{d.rows}  cell={d.cell_width}x{d.cell_height}  "
        f"scale={d.scale_mode}  transport={d.default_transport}"
    )
    visible = config.visible_cameras()
    print(f"Cameras: {len(visible)} shown / {len(config.cameras)} configured")
    for i, cam in enumerate(config.cameras, start=1):
        mark = "*" if cam in visible else " "
        state = "on " if cam.enabled else "off"
        print(f"  {mark} {i}. [{state}] {cam.name:20}  {cam.redacted_source()}")
    if not visible:
        print("warning: no enabled cameras fit in the grid", file=sys.stderr)
        return 1
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    dest = Path(args.output)
    if dest.exists() and not args.force:
        print(f"error: {dest} already exists (use --force to overwrite)", file=sys.stderr)
        return 1
    dest.write_text(example_config_text(), encoding="utf-8")
    print(f"Wrote {dest}")
    print("Edit camera URLs, then run:  security-monitor")
    return 0


def _apply_overrides(config: AppConfig, args: argparse.Namespace) -> None:
    if getattr(args, "fullscreen", False):
        config.display.fullscreen = True
    if getattr(args, "columns", None):
        if args.columns < 1:
            raise ConfigError("columns must be >= 1")
        config.display.columns = args.columns
    if getattr(args, "rows", None):
        if args.rows < 1:
            raise ConfigError("rows must be >= 1")
        config.display.rows = args.rows
