"""Independent check of a predictions file against test_unlabelled.csv and sample_submission.csv.

Usage:  python -m scripts.validate_submission [predictions.csv]
Exit code 0 = valid, 1 = invalid.
"""

from __future__ import annotations

import sys
from pathlib import Path

from kestrel_router import config
from kestrel_router.data import read_csv
from kestrel_router.submission import validate_submission


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else config.PROJECT_ROOT / "predictions.csv"
    if not path.exists():
        print(f"INVALID: {path} does not exist", file=sys.stderr)
        return 1
    submission = read_csv(path)
    test = read_csv(config.DATA_DIR / config.TEST_FILE)
    sample = read_csv(config.DATA_DIR / config.SAMPLE_SUBMISSION_FILE)
    problems = validate_submission(submission, test["request_id"].tolist(), sample)

    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        problems.append("file starts with a UTF-8 BOM")
    if raw.splitlines()[0].decode() != ",".join(config.SUBMISSION_COLUMNS):
        problems.append(f"header line is {raw.splitlines()[0]!r}")

    if problems:
        print("INVALID:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    print(f"VALID: {path.name}: {len(submission)} rows, columns {list(submission.columns)}, "
          f"{submission['request_id'].nunique()} unique ids, "
          f"teams used: {sorted(submission['team'].unique())}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
