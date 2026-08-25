"""Parallel camera reboot (SSH for Ubiquiti, HTTP for Reolink/Amcrest/Dahua)."""

from __future__ import annotations

import os
import socket
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable
from urllib.parse import urlparse

from security_monitor.config import CameraConfig

PING_DROP_TIMEOUT = 45
PING_RETURN_TIMEOUT = 150
HTTP_TIMEOUT = 10
SSH_TIMEOUT = 8
HTTP_KINDS = frozenset({"reolink", "amcrest", "dahua"})
SSH_KINDS = frozenset({"ubiquiti"})


@dataclass
class RebootDevice:
    name: str
    kind: str
    host: str
    username: str
    password: str
    ssh_port: int = 22
    http_port: int = 80
    scheme: str = "http"


@dataclass
class RebootRow:
    name: str
    host: str
    kind: str
    phase: str = "queued"
    elapsed: float = 0.0
    total: float = 0.0
    status: str = "waiting"
    color: str = "white"
    detail: str = ""
    done: bool = False


def ping_command(host: str) -> list[str]:
    if os.name == "nt":
        return ["ping", "-n", "1", "-w", "1000", host]
    return ["ping", "-c", "1", "-W", "1", host]


def camera_host(camera: CameraConfig) -> str | None:
    if camera.device is not None or not camera.url:
        return None
    host = urlparse(camera.url).hostname
    return host or None


def reboot_targets(cameras: list[CameraConfig]) -> list[RebootDevice]:
    devices: list[RebootDevice] = []
    for cam in cameras:
        if not cam.enabled or not cam.reboot:
            continue
        kind = (cam.kind or "").lower()
        if kind not in SSH_KINDS | HTTP_KINDS:
            continue
        host = camera_host(cam)
        if not host:
            continue
        if not cam.username or cam.password is None:
            continue
        devices.append(
            RebootDevice(
                name=cam.name,
                kind=kind,
                host=host,
                username=cam.username,
                password=cam.password,
                ssh_port=cam.ssh_port,
                http_port=cam.http_port,
            )
        )
    return devices


class RebootJob:
    """Run all camera reboots concurrently and expose live row state for the UI."""

    def __init__(self, devices: list[RebootDevice]) -> None:
        self.devices = devices
        self._lock = threading.Lock()
        self._rows: dict[str, RebootRow] = {
            dev.name: RebootRow(name=dev.name, host=dev.host, kind=dev.kind)
            for dev in devices
        }
        self._summaries: list[str] = []
        self._finished = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def finished(self) -> bool:
        return self._finished.is_set()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run_all, name="reboot-job", daemon=True)
        self._thread.start()

    def rows(self) -> list[RebootRow]:
        with self._lock:
            return [RebootRow(**vars(row)) for row in self._rows.values()]

    def summaries(self) -> list[str]:
        with self._lock:
            return list(self._summaries)

    def _set(self, name: str, **fields: object) -> None:
        with self._lock:
            row = self._rows[name]
            for key, value in fields.items():
                setattr(row, key, value)

    def _run_all(self) -> None:
        if not self.devices:
            self._finished.set()
            return
        with ThreadPoolExecutor(max_workers=len(self.devices)) as pool:
            futures = [pool.submit(self._worker, device) for device in self.devices]
            for fut in as_completed(futures):
                summary = fut.result()
                with self._lock:
                    self._summaries.append(summary)
        self._finished.set()

    def _worker(self, device: RebootDevice) -> str:
        self._set(device.name, phase="starting", status="initializing…", color="cyan")
        try:
            if device.kind in SSH_KINDS:
                return self._process_ubiquiti(device)
            if device.kind in HTTP_KINDS:
                return self._process_http(device)
            self._set(
                device.name,
                phase="done",
                status=f"unsupported type '{device.kind}'",
                color="yellow",
                done=True,
            )
            return f"[SKIP] {device.name} unsupported type '{device.kind}'."
        except Exception as exc:
            self._set(
                device.name,
                phase="done",
                status="fatal error",
                color="red",
                detail=str(exc),
                done=True,
            )
            return f"[ERR] {device.name} fatal: {exc}"

    def _process_ubiquiti(self, device: RebootDevice) -> str:
        import paramiko

        name, host, port = device.name, device.host, device.ssh_port
        self._set(name, phase="precheck", status="ping+ssh check", color="cyan")
        if not ping_once(host):
            self._set(name, status="unreachable (ICMP)", color="red", done=True)
            return f"[ERR] {name} ({host}) Unreachable (ICMP)."
        if not tcp_port_open(host, port, timeout=2.0):
            self._set(name, status=f"SSH {port} closed", color="red", done=True)
            return f"[ERR] {name} ({host}) SSH port {port} closed/filtered."

        self._set(name, phase="connect", status="SSH connecting…", color="cyan", detail="")
        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(
                hostname=host,
                port=port,
                username=device.username,
                password=device.password,
                auth_timeout=SSH_TIMEOUT,
                banner_timeout=SSH_TIMEOUT,
                timeout=SSH_TIMEOUT,
                look_for_keys=False,
                allow_agent=False,
            )
        except Exception as exc:
            self._set(name, status="SSH failed", color="red", detail=str(exc), done=True)
            return f"[ERR] {name} ({host}) SSH failed: {exc}"

        pre_uptime = _read_linux_uptime(client)
        self._set(
            name,
            status=f"pre-uptime: {int(pre_uptime) if pre_uptime else 'unknown'}s",
            color="cyan",
        )
        self._set(name, phase="command", status="sending reboot…", color="yellow")
        _send_ubnt_reboot(client)
        client.close()

        self._set(
            name,
            phase="going down",
            status="waiting for ping to drop",
            color="yellow",
            elapsed=0,
            total=PING_DROP_TIMEOUT,
        )
        if not self._wait_ping(name, host, PING_DROP_TIMEOUT, want_up=False):
            self._set(name, status="never dropped", color="yellow", done=True)
            return f"[WARN] {name} ({host}) Reboot sent; ping never dropped."

        self._set(
            name,
            phase="coming back",
            status="waiting for ping to return",
            color="yellow",
            elapsed=0,
            total=PING_RETURN_TIMEOUT,
        )
        if not self._wait_ping(name, host, PING_RETURN_TIMEOUT, want_up=True):
            self._set(name, status="down too long", color="red", done=True)
            return f"[ERR] {name} ({host}) Went down; not back within {PING_RETURN_TIMEOUT}s."

        self._set(name, phase="verify", status="SSH verify uptime…", color="cyan")
        try:
            client2 = paramiko.SSHClient()
            client2.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client2.connect(
                hostname=host,
                port=port,
                username=device.username,
                password=device.password,
                auth_timeout=SSH_TIMEOUT,
                banner_timeout=SSH_TIMEOUT,
                timeout=SSH_TIMEOUT,
                look_for_keys=False,
                allow_agent=False,
            )
            post_uptime = _read_linux_uptime(client2)
            client2.close()
            if post_uptime is not None and post_uptime < 120:
                self._set(
                    name,
                    phase="done",
                    status=f"OK (uptime {int(post_uptime)}s)",
                    color="green",
                    done=True,
                )
                return f"[OK] {name} ({host}) Reboot verified (uptime {int(post_uptime)}s)."
            self._set(
                name,
                phase="done",
                status=f"OK? (uptime {int(post_uptime) if post_uptime else 'unknown'}s)",
                color="yellow",
                done=True,
            )
            return (
                f"[OK?] {name} ({host}) Back online, "
                f"uptime={int(post_uptime) if post_uptime else 'unknown'}s."
            )
        except Exception as exc:
            self._set(
                name,
                phase="done",
                status="OK? (verify failed)",
                color="yellow",
                detail=str(exc),
                done=True,
            )
            return f"[OK?] {name} ({host}) Back online; SSH verify failed: {exc}"

    def _process_http(self, device: RebootDevice) -> str:
        import urllib3
        import requests
        from requests.auth import HTTPDigestAuth

        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        name, host = device.name, device.host
        port = device.http_port
        scheme = device.scheme.lower()
        self._set(name, phase="precheck", status="ping+port check", color="cyan")
        if not ping_once(host):
            self._set(name, status="unreachable (ICMP)", color="red", done=True)
            return f"[ERR] {name} ({host}) Unreachable (ICMP)."
        if not tcp_port_open(host, port, timeout=2.0):
            if (scheme, port) == ("http", 80) and tcp_port_open(host, 443, timeout=2.0):
                scheme, port = "https", 443
            else:
                self._set(name, status=f"{scheme.upper()}:{port} closed", color="red", done=True)
                return f"[ERR] {name} ({host}) {scheme.upper()}:{port} closed/filtered."

        self._set(
            name,
            phase="command",
            status=f"HTTP reboot → {scheme.upper()}:{port}",
            color="yellow",
        )
        url = f"{scheme}://{host}:{port}/cgi-bin/magicBox.cgi?action=reboot"
        try:
            response = requests.get(
                url,
                auth=HTTPDigestAuth(device.username, device.password),
                timeout=HTTP_TIMEOUT,
                verify=False,
            )
            if response.status_code != 200:
                self._set(name, status=f"HTTP {response.status_code}", color="red", done=True)
                return f"[ERR] {name} ({host}) Reboot request failed: HTTP {response.status_code}"
        except Exception as exc:
            self._set(name, status="HTTP error", color="red", detail=str(exc), done=True)
            return f"[ERR] {name} ({host}) Reboot request error: {exc}"

        self._set(
            name,
            phase="going down",
            status="waiting for ping to drop",
            color="yellow",
            elapsed=0,
            total=PING_DROP_TIMEOUT,
        )
        if not self._wait_ping(name, host, PING_DROP_TIMEOUT, want_up=False):
            self._set(name, status="never dropped", color="yellow", done=True)
            return f"[WARN] {name} ({host}) Reboot sent; ping never dropped."

        self._set(
            name,
            phase="coming back",
            status="waiting for ping to return",
            color="yellow",
            elapsed=0,
            total=PING_RETURN_TIMEOUT,
        )
        if not self._wait_ping(name, host, PING_RETURN_TIMEOUT, want_up=True):
            self._set(name, status="down too long", color="red", done=True)
            return f"[ERR] {name} ({host}) Went down; not back within {PING_RETURN_TIMEOUT}s."

        self._set(name, phase="verify", status="probing web UI…", color="cyan")
        try:
            probe = requests.get(
                f"{scheme}://{host}:{port}",
                timeout=5,
                verify=False,
            )
            if probe.status_code == 200:
                self._set(name, phase="done", status="OK (web 200)", color="green", done=True)
                return f"[OK] {name} ({host}) Reboot verified (ping dropped/returned, web UI 200)."
        except Exception:
            pass
        self._set(name, phase="done", status="OK (ping drop/return)", color="green", done=True)
        return f"[OK] {name} ({host}) Reboot verified (ping dropped/returned)."

    def _wait_ping(self, name: str, host: str, timeout_s: int, *, want_up: bool) -> bool:
        started = time.time()
        self._set(name, total=timeout_s)
        while True:
            elapsed = time.time() - started
            self._set(name, elapsed=min(elapsed, timeout_s))
            if ping_once(host) == want_up:
                return True
            if elapsed >= timeout_s:
                return False
            time.sleep(1)


def ping_once(host: str) -> bool:
    try:
        result = subprocess.run(
            ping_command(host),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return True


def tcp_port_open(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, TimeoutError, socket.timeout):
        return False


def _read_linux_uptime(ssh: object) -> float | None:
    try:
        _stdin, stdout, _stderr = ssh.exec_command("cat /proc/uptime", timeout=5)  # type: ignore[attr-defined]
        raw = stdout.read().decode("utf-8", "ignore")
        return float(raw.split()[0])
    except Exception:
        return None


def _send_ubnt_reboot(ssh: object) -> None:
    for cmd in ("reboot", "/sbin/reboot", "ubnt-systool reboot"):
        try:
            ssh.exec_command(cmd, timeout=3)  # type: ignore[attr-defined]
            return
        except Exception:
            continue


def run_reboot_cli(devices: list[RebootDevice], log: Callable[[str], None] = print) -> int:
    if not devices:
        log("No rebootable cameras. Set type: ubiquiti or reolink on each camera in config.yaml.")
        return 1
    log(f"Rebooting {len(devices)} camera(s) in parallel…")
    job = RebootJob(devices)
    job.start()
    seen: dict[str, str] = {}
    while not job.finished:
        for row in job.rows():
            stamp = f"{row.phase}|{row.status}"
            if seen.get(row.name) != stamp:
                seen[row.name] = stamp
                extra = f"  {row.detail}" if row.detail else ""
                log(f"  {row.name:16} {row.phase:12} {row.status}{extra}")
        time.sleep(0.25)
    log("\nSummary:")
    for line in job.summaries():
        log(line)
    log("\nNote: typical reboot window is 30–90s.")
    return 0
