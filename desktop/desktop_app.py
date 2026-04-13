from __future__ import annotations

import argparse
import contextlib
import json
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

from app_paths import bundle_path, runtime_dir

APP_TITLE = "Realtime Speech Translation"
HOST = "127.0.0.1"
HEALTH_PATH = "/health"
SERVER_READY_TIMEOUT_SEC = 15.0
WINDOW_MIN_WIDTH = 420
WINDOW_MIN_HEIGHT = 760
WINDOW_START_WIDTH = 460
WINDOW_START_HEIGHT = 920
SPLASH_PATH = bundle_path("desktop", "assets", "splash.html")


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
        self._app = None
        self._shutdown_callback = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> None:
        if self._app is None:
            from api import app, stop_pipeline_process_for_shutdown

            self._app = app
            self._shutdown_callback = stop_pipeline_process_for_shutdown

        config = uvicorn.Config(
            app=self._app,
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
        if self._shutdown_callback is not None:
            self._shutdown_callback()
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5.0)


def _load_splash_html() -> str:
    return SPLASH_PATH.read_text(encoding="utf-8")


def _set_splash_state(
    window: webview.Window,
    *,
    step: str,
    badge: str,
    title: str,
    detail: str,
    mode: str = "booting",
    error: str = "",
) -> None:
    payload = {
        "step": step,
        "badge": badge,
        "title": title,
        "detail": detail,
        "mode": mode,
        "error": error,
    }
    script = f"window.updateSplashState({json.dumps(payload, ensure_ascii=False)});"
    with contextlib.suppress(Exception):
        window.evaluate_js(script)


def _bootstrap_desktop(window: webview.Window, server: _DesktopServer) -> None:
    try:
        _set_splash_state(
            window,
            step="bootstrap",
            badge="booting",
            title="正在初始化桌面环境",
            detail="窗口已创建，正在准备桌面启动序列。",
        )
        time.sleep(0.15)

        _set_splash_state(
            window,
            step="server",
            badge="starting",
            title="正在启动本地服务",
            detail="FastAPI 与桌面包装层正在启动，通常只需要几秒钟。",
        )
        server.start()

        _set_splash_state(
            window,
            step="health",
            badge="checking",
            title="正在检查服务状态",
            detail="本地服务已启动，正在验证前端与 API 健康状态。",
        )
        time.sleep(0.35)

        _set_splash_state(
            window,
            step="ready",
            badge="ready",
            title="准备完成，正在进入主界面",
            detail="启动检查已经通过，即将切换到实时语音翻译工作台。",
            mode="ready",
        )
        time.sleep(0.45)
        window.load_url(server.base_url)
    except Exception as exc:
        _set_splash_state(
            window,
            step="server",
            badge="failed",
            title="启动失败",
            detail="桌面窗口已保留。请查看下方错误信息，修复后重新打开应用。",
            mode="error",
            error=str(exc),
        )


def _run_desktop_window() -> None:
    port = _pick_free_port()
    server = _DesktopServer(HOST, port)

    window = webview.create_window(
        APP_TITLE,
        html=_load_splash_html(),
        min_size=(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT),
        width=WINDOW_START_WIDTH,
        height=WINDOW_START_HEIGHT,
        text_select=True,
        background_color="#EEF7F2",
    )

    try:
        webview.start(func=_bootstrap_desktop, args=(window, server))
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
