"""Guard for the public repository: fail if restricted Kestrel data or secrets would be published.

Usage:
  python -m scripts.audit_public_repo            # audit files staged in the git index
  python -m scripts.audit_public_repo --ref HEAD # audit a commit

Checks every tracked/staged path for:
  * assignment source files, CSV/PDF/model/pickle files, env files
  * real request IDs (SR500000–SR512999) and order numbers (KO26xxxxx)
  * likely secrets (API keys, tokens, private keys)
  * verbatim customer request text (only when the assignment CSVs are present locally)

Exit code 0 = safe to publish, 1 = blocked.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RESTRICTED_NAMES = {
    "train.csv", "test_unlabelled.csv", "resolution_log.csv", "teams.csv", "sample_submission.csv",
    "ops-policy.pdf", "email-thread.txt", "README.txt", "predictions.csv",
}
RESTRICTED_SUFFIXES = {".csv", ".pdf", ".joblib", ".pkl", ".pickle", ".parquet", ".xlsx", ".npy", ".npz"}
ALLOWED_EXCEPTIONS = re.compile(r"^examples/[\w.-]+\.example\.csv$")
RESTRICTED_DIRS = {".venv", "venv", "__pycache__", ".claude", ".pytest_cache", ".ruff_cache"}

CONTENT_PATTERNS = {
    "real request id": re.compile(r"\bSR5(0\d|1[0-2])\d{3}\b"),
    "real order number": re.compile(r"\bKO26\d{5}\b"),
    "api key / token": re.compile(
        r"(sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{30,}|github_pat_\w{20,}|AIza[0-9A-Za-z_-]{30,}"
        r"|xox[baprs]-[A-Za-z0-9-]{10,})"
    ),
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "credential assignment": re.compile(
        r"(?i)\b(api[_-]?key|secret|password|token)\s*[:=]\s*['\"][^'\"\s]{8,}['\"]"
    ),
}


def tracked_files(ref: str | None) -> list[str]:
    cmd = ["git", "ls-tree", "-r", "--name-only", ref] if ref else ["git", "ls-files", "--cached"]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [line for line in out.splitlines() if line]


def read(path: str, ref: str | None) -> str:
    if ref:
        out = subprocess.run(["git", "show", f"{ref}:{path}"], cwd=ROOT, capture_output=True, check=True).stdout
    else:
        out = subprocess.run(["git", "show", f":{path}"], cwd=ROOT, capture_output=True, check=True).stdout
    return out.decode("utf-8", errors="ignore")


def source_texts() -> set[str]:
    """Customer messages from the local assignment files (if present), lower-cased, >= 30 chars."""
    texts: set[str] = set()
    for name in ("train.csv", "test_unlabelled.csv"):
        path = ROOT / name
        if path.exists():
            import pandas as pd

            df = pd.read_csv(path, dtype=str, keep_default_na=False)
            texts |= {t.lower() for t in df["request_text"] if len(t) >= 30}
    return texts


def audit(ref: str | None) -> list[str]:
    problems: list[str] = []
    files = tracked_files(ref)
    texts = source_texts()
    for path in files:
        name = Path(path).name
        suffix = Path(path).suffix.lower()
        if name in RESTRICTED_NAMES:
            problems.append(f"{path}: restricted assignment file")
            continue
        if suffix in RESTRICTED_SUFFIXES and not ALLOWED_EXCEPTIONS.match(path):
            problems.append(f"{path}: restricted file type {suffix}")
            continue
        parts = path.split("/")
        if name.startswith(".env") or path.startswith("reports/private/") or RESTRICTED_DIRS & set(parts[:-1]):
            problems.append(f"{path}: restricted path")
            continue
        content = read(path, ref)
        if path == "scripts/audit_public_repo.py":
            continue  # contains the patterns themselves
        for label, pattern in CONTENT_PATTERNS.items():
            m = pattern.search(content)
            if m:
                problems.append(f"{path}: contains {label} ({m.group(0)[:12]}...)")
        lowered = content.lower()
        hits = [t for t in texts if t in lowered]
        if hits:
            problems.append(f"{path}: contains {len(hits)} verbatim customer request text(s)")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref", help="audit this commit instead of the staged index")
    args = parser.parse_args()
    files = tracked_files(args.ref)
    problems = audit(args.ref)
    scope = args.ref or "staged index"
    if problems:
        print(f"BLOCKED ({scope}): {len(problems)} problem(s)")
        for p in problems:
            print("  -", p)
        return 1
    scanned = "with" if source_texts() else "without (data absent)"
    print(f"SAFE ({scope}): {len(files)} files checked, {scanned} verbatim-text scan")
    return 0


if __name__ == "__main__":
    sys.exit(main())
