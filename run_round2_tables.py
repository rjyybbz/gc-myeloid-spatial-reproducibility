from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    ROOT / "revision_scripts" / "71_round2_sample_level_peritoneal_tables.py",
    ROOT / "revision_scripts" / "72_round2_build_supplementary_tables.py",
]


def main() -> None:
    for script in SCRIPTS:
        print(f"[run] {script.relative_to(ROOT)}")
        completed = subprocess.run([sys.executable, str(script)], cwd=ROOT, check=False)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)
    print("[done] round-2 supplementary tables regenerated")


if __name__ == "__main__":
    main()
