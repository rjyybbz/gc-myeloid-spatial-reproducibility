from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROJECT_PATHS = {
    "root": PROJECT_ROOT,
    "config": PROJECT_ROOT / "config",
    "data_raw": PROJECT_ROOT / "data_raw",
    "data_processed": PROJECT_ROOT / "data_processed",
    "docs": PROJECT_ROOT / "docs",
    "environment": PROJECT_ROOT / "environment",
    "figures": PROJECT_ROOT / "figures",
    "logs": PROJECT_ROOT / "logs",
    "metadata": PROJECT_ROOT / "metadata",
    "results": PROJECT_ROOT / "results",
    "inventory": PROJECT_ROOT / "results" / "inventory",
    "tables": PROJECT_ROOT / "results" / "tables",
    "qc": PROJECT_ROOT / "results" / "qc",
    "scripts": PROJECT_ROOT / "scripts",
    "tools": PROJECT_ROOT / "tools",
}


def ensure_directories() -> None:
    for path in PROJECT_PATHS.values():
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def safe_run(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            encoding="utf-8",
            errors="replace",
        )
    except Exception as exc:
        return f"ERROR: {exc}"
    if completed.returncode != 0:
        return completed.stderr.strip() or completed.stdout.strip() or f"exit_code={completed.returncode}"
    return completed.stdout.strip()


def build_runtime_snapshot() -> dict:
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cwd": str(PROJECT_ROOT),
        "pip_freeze_top": safe_run([sys.executable, "-m", "pip", "list", "--format=freeze"]).splitlines()[:80],
    }


def main() -> None:
    ensure_directories()
    snapshot = build_runtime_snapshot()
    write_json(PROJECT_PATHS["logs"] / "runtime_snapshot.json", snapshot)
    print(f"[setup] Project root ready: {PROJECT_ROOT}")
    print(f"[setup] Runtime snapshot written to: {PROJECT_PATHS['logs'] / 'runtime_snapshot.json'}")


if __name__ == "__main__":
    main()
