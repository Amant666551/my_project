from __future__ import annotations

import argparse
import contextlib
import socket
import sys
import threading
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import requests
import uvicorn
import webview

from api import app, stop_pipeline_process_for_shutdown
from app_paths import runtime_dir

APP_TITLE = "Realtime Speech Translation"
HOST = "127.0.0.1"
HEALTH_PATH = "/health"
SERVER_READY_TIMEOUT_SEC = 15.0
WINDOW_MIN_WIDTH = 420
WINDOW_MIN_HEIGHT = 760
WINDOW_START_WIDTH = 460
WINDOW_START_HEIGHT = 920


def _pick_free_port(host: str = HOST) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


class _DesktopServer:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> None:
        config = uvicorn.Config(
            app=app,
            host=self.host,
            port=self.port,
            log_level="info",
            access_log=False,
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(
            target=self._server.run,
            daemon=True,
            name="desktop-web-server",
        )
        self._thread.start()
        self._wait_until_ready()

    def _wait_until_ready(self) -> None:
        deadline = time.time() + SERVER_READY_TIMEOUT_SEC
        health_url = f"{self.base_url}{HEALTH_PATH}"
        last_error: Exception | None = None

        while time.time() < deadline:
            if self._server is not None and self._server.started:
                try:
                    response = requests.get(health_url, timeout=1.0)
                    if response.ok:
                        return
                except requests.RequestException as exc:
                    last_error = exc
            time.sleep(0.15)

        raise RuntimeError(
            f"Desktop server failed to start within {SERVER_READY_TIMEOUT_SEC:.1f}s."
        ) from last_error

    def stop(self) -> None:
        stop_pipeline_process_for_shutdown()
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5.0)


def _run_desktop_window() -> None:
    port = _pick_free_port()
    server = _DesktopServer(HOST, port)
    server.start()

    window = webview.create_window(
        APP_TITLE,
        server.base_url,
        min_size=(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT),
        width=WINDOW_START_WIDTH,
        height=WINDOW_START_HEIGHT,
        text_select=True,
    )

    try:
        webview.start()
    finally:
        server.stop()
        with contextlib.suppress(Exception):
            window.destroy()


def _run_orchestrator_mode() -> None:
    import orchestrator

    with contextlib.suppress(Exception):
        import os

        os.chdir(runtime_dir())
    orchestrator.main()


def main() -> None:
    parser = argparse.ArgumentParser(description="Desktop wrapper for realtime speech translation.")
    parser.add_argument(
        "--run-orchestrator",
        action="store_true",
        help="Run the realtime orchestrator directly instead of opening the desktop window.",
    )
    args = parser.parse_args()

    if args.run_orchestrator:
        _run_orchestrator_mode()
        return

    _run_desktop_window()


if __name__ == "__main__":
    main()
