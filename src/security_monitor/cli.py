"""Command-line entry point."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from security_monitor import __version__
from security_monitor.autostart import AutostartError, AutostartOptions
from security_monitor.config import (
    AppConfig,
    ConfigError,
    demo_config,
    example_config_text,
    load_config,
    resolve_config_path,
    xdg_user_config_path,
)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except AutostartError as exc:
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
    demo.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Seconds to wait before opening the window (useful after login)",
    )
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
        default=None,
        help="Destination path (default: ./config.yaml)",
    )
    init.add_argument(
        "--user",
        action="store_true",
        help="Write to the per-user config path (~/.config/security-monitor/ on Linux)",
    )
    init.set_defaults(handler=cmd_init)

    reboot = sub.add_parser("reboot", help="Reboot cameras from config.yaml (SSH/HTTP)")
    reboot.add_argument("--config", "-c", default=argparse.SUPPRESS, help="Path to config.yaml")
    reboot.set_defaults(handler=cmd_reboot)

    autostart = sub.add_parser(
        "autostart",
        help="Install or remove login autostart (Ubuntu desktop / Windows Startup)",
    )
    autostart_sub = autostart.add_subparsers(dest="autostart_command", required=True)

    install = autostart_sub.add_parser("install", help="Start Security Monitor after login")
    install.add_argument("--config", "-c", default=None, help="Absolute path embedded in autostart")
    install.add_argument(
        "--fullscreen",
        action="store_true",
        help="Launch fullscreen after login",
    )
    install.add_argument(
        "--delay",
        type=float,
        default=15.0,
        help="Seconds to wait after login before opening (default: 15)",
    )
    install.add_argument(
        "--method",
        choices=("auto", "desktop", "systemd"),
        default="auto",
        help="Linux only: XDG .desktop (default) or systemd --user unit",
    )
    install.add_argument(
        "--workdir",
        default=None,
        help="Working directory for the launched process",
    )
    install.set_defaults(handler=cmd_autostart_install)

    remove = autostart_sub.add_parser("uninstall", help="Remove login autostart")
    remove.add_argument(
        "--method",
        choices=("auto", "desktop", "systemd"),
        default="auto",
        help="Linux only: which entry to remove (default: both if present)",
    )
    remove.set_defaults(handler=cmd_autostart_uninstall)

    status = autostart_sub.add_parser("status", help="Show whether autostart is installed")
    status.set_defaults(handler=cmd_autostart_status)

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
    parser.add_argument(
        "--delay",
        type=float,
        help="Seconds to wait before opening the window (useful after login)",
        **extra,
    )
    parser.add_argument("--columns", type=int, help="Override grid columns", **extra)
    parser.add_argument("--rows", type=int, help="Override grid rows", **extra)


def _maybe_delay(args: argparse.Namespace) -> None:
    delay = float(getattr(args, "delay", 0) or 0)
    if delay <= 0:
        return
    print(f"Waiting {delay:g}s before starting (network / desktop settle)...")
    time.sleep(delay)


def cmd_run(args: argparse.Namespace) -> int:
    if getattr(args, "demo", False):
        return cmd_demo(args)
    _maybe_delay(args)
    config = load_config(getattr(args, "config", None))
    _apply_overrides(config, args)
    from security_monitor.mosaic import run_monitor

    return run_monitor(config)


def cmd_demo(args: argparse.Namespace) -> int:
    _maybe_delay(args)
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
        kind = cam.kind or "-"
        print(f"  {mark} {i}. [{state}] {cam.name:20}  {kind}  {cam.redacted_source()}")
    if not visible:
        print("warning: no enabled cameras fit in the grid", file=sys.stderr)
        return 1
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    if args.output and args.user:
        print("error: use either --output or --user, not both", file=sys.stderr)
        return 1
    if args.user:
        dest = xdg_user_config_path()
        dest.parent.mkdir(parents=True, exist_ok=True)
    elif args.output:
        dest = Path(args.output)
    else:
        dest = Path("config.yaml")
    if dest.exists() and not args.force:
        print(f"error: {dest} already exists (use --force to overwrite)", file=sys.stderr)
        return 1
    dest.write_text(example_config_text(), encoding="utf-8")
    print(f"Wrote {dest}")
    print("Edit camera URLs, then run:  security-monitor")
    if args.user:
        print("This path is used automatically when no ./config.yaml is present.")
    return 0


def cmd_reboot(args: argparse.Namespace) -> int:
    config = load_config(getattr(args, "config", None))
    from security_monitor.reboot import reboot_targets, run_reboot_cli

    return run_reboot_cli(reboot_targets(config.cameras))


def cmd_autostart_install(args: argparse.Namespace) -> int:
    from security_monitor.autostart import install_autostart

    config_path: Path | None = None
    if args.config:
        config_path = Path(args.config).expanduser()
    else:
        # Prefer an already-resolved config so login launch does not depend on cwd.
        try:
            config_path = resolve_config_path(None)
        except ConfigError:
            config_path = None
            print(
                "warning: no config.yaml found yet; autostart will search default paths at login",
                file=sys.stderr,
            )

    workdir = Path(args.workdir).expanduser() if args.workdir else None
    status = install_autostart(
        AutostartOptions(
            config=config_path,
            fullscreen=bool(args.fullscreen),
            delay=float(args.delay),
            working_directory=workdir,
            method=args.method,
        )
    )
    print(f"Installed {status.method} autostart: {status.path}")
    if status.detail:
        print(status.detail)
    if config_path is not None:
        print(f"Config: {config_path}")
    print("Verify with:  security-monitor autostart status")
    return 0


def cmd_autostart_uninstall(args: argparse.Namespace) -> int:
    from security_monitor.autostart import uninstall_autostart

    results = uninstall_autostart(method=args.method)
    for status in results:
        print(f"{status.method}: {status.detail} ({status.path})")
    return 0


def cmd_autostart_status(args: argparse.Namespace) -> int:
    from security_monitor.autostart import status_autostart

    results = status_autostart()
    any_installed = False
    for status in results:
        state = "installed" if status.installed else "not installed"
        print(f"{status.platform}/{status.method}: {state}")
        if status.path is not None:
            print(f"  path: {status.path}")
        if status.installed:
            any_installed = True
            if status.path is not None and status.path.is_file():
                preview = status.path.read_text(encoding="utf-8").strip().splitlines()
                for line in preview[:12]:
                    print(f"  | {line}")
    if not any_installed:
        print("Tip: security-monitor autostart install --fullscreen")
        return 1
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
