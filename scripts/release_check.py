"""One command for the pre-submission / pre-push checklist.

Usage:  python -m scripts.release_check

Runs, in order:
  1. ruff lint
  2. pytest (full suite)
  3. public-repo data audit of HEAD (and of the staged index if anything is staged)
  4. model smoke test (if models/router.joblib exists)
  5. predictions.csv validation (if predictions.csv and the assignment data exist)

Steps 4–5 are reported as SKIPPED when their inputs are absent (for example in a public clone).
Exit code 0 only if no step failed.
"""

from __future__ import annotations

import subprocess
import sys

from kestrel_router import config

PY = sys.executable


def run(name: str, cmd: list[str]) -> bool:
    print(f"\n=== {name} ===", flush=True)
    ok = subprocess.run(cmd, cwd=config.PROJECT_ROOT).returncode == 0
    print(f"--> {name}: {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def main() -> int:
    results: dict[str, str] = {}

    def record(name: str, cmd: list[str]) -> None:
        results[name] = "PASS" if run(name, cmd) else "FAIL"

    record("lint", [PY, "-m", "ruff", "check", "."])
    record("tests", [PY, "-m", "pytest"])
    record("data audit (HEAD)", [PY, "-m", "scripts.audit_public_repo", "--ref", "HEAD"])
    staged = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=config.PROJECT_ROOT,
                            capture_output=True, text=True).stdout.strip()
    if staged:
        record("data audit (staged)", [PY, "-m", "scripts.audit_public_repo"])

    if config.MODEL_PATH.exists():
        record("smoke test", [PY, "-m", "scripts.smoke_test"])
    else:
        results["smoke test"] = "SKIPPED (no models/router.joblib)"

    predictions = config.PROJECT_ROOT / "predictions.csv"
    if predictions.exists() and (config.DATA_DIR / config.TEST_FILE).exists():
        record("predictions.csv", [PY, "-m", "scripts.validate_submission", str(predictions)])
    else:
        results["predictions.csv"] = "SKIPPED (predictions or assignment data absent)"

    print("\n=== summary ===")
    for name, status in results.items():
        print(f"{name:22s} {status}")
    return 1 if "FAIL" in results.values() else 0


if __name__ == "__main__":
    sys.exit(main())
