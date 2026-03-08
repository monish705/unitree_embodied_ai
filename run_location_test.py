import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parent
SERVER_URL = os.environ.get("SERVER_URL", "http://localhost:8000")


def _stream_process(process: subprocess.Popen, log_path: Path, label: str) -> int:
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        assert process.stdout is not None
        for line in process.stdout:
            log_file.write(f"[{label}] {line}")
    return process.wait()


def _wait_for_server(timeout_s: float = 60.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            response = requests.get(f"{SERVER_URL}/discover", timeout=2)
            if response.ok and response.json().get("tools"):
                return
        except requests.RequestException:
            pass
        time.sleep(1.0)
    raise TimeoutError("Telemetry server did not become ready in time.")


def _post_shutdown() -> None:
    try:
        requests.post(f"{SERVER_URL}/shutdown", timeout=5)
    except requests.RequestException:
        pass


def main() -> int:
    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY is required.", file=sys.stderr)
        return 2

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    artifact_dir = PROJECT_ROOT / "artifacts" / f"location_test_{timestamp}"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    video_path = artifact_dir / "location_test.mp4"
    summary_path = artifact_dir / "summary.json"
    server_log = artifact_dir / "server.log"
    navigator_log = artifact_dir / "navigator.log"

    env = os.environ.copy()
    env["LOCATION_TEST_RECORD"] = "1"
    env["RECORD_OUTPUT"] = str(video_path)
    env.setdefault("RECORD_CAMERA", "overview_cam")
    env.setdefault("RECORD_LAYOUT", "dual")
    env.setdefault("RECORD_FPS", "10")
    env["HEADLESS"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    server = subprocess.Popen(
        [sys.executable, "server.py"],
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    server_result = {}
    server_thread = threading.Thread(
        target=lambda: server_result.setdefault(
            "exit_code", _stream_process(server, server_log, "server")
        ),
        daemon=True,
    )
    server_thread.start()

    try:
        _wait_for_server()
    except Exception as exc:
        _post_shutdown()
        server.terminate()
        server_thread.join(timeout=10)
        print(f"Failed to start server: {exc}", file=sys.stderr)
        return 1

    navigator = subprocess.Popen(
        [sys.executable, "vlm_navigator.py"],
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    try:
        navigator_code = _stream_process(navigator, navigator_log, "navigator")
        try:
            state = requests.get(f"{SERVER_URL}/state", timeout=10).json()
        except requests.RequestException as exc:
            state = {"error": str(exc)}

        _post_shutdown()
        server_thread.join(timeout=30)
        server_code = server_result.get("exit_code", server.poll())
    finally:
        if navigator.poll() is None:
            navigator.terminate()
        if server.poll() is None:
            server.terminate()

    reached_target = (
        isinstance(state, dict)
        and state.get("stable") is True
        and float(state.get("distance_to_target_surface_xy", 999.0)) <= 0.10
    )

    summary = {
        "artifact_dir": str(artifact_dir),
        "video_path": str(video_path),
        "server_log": str(server_log),
        "navigator_log": str(navigator_log),
        "server_exit_code": server_code,
        "navigator_exit_code": navigator_code,
        "state": state,
        "passed": reached_target,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0 if reached_target else 1


if __name__ == "__main__":
    raise SystemExit(main())
