"""Predict every request in test_unlabelled.csv and write predictions.csv.

Usage:  python -m scripts.predict [--out predictions.csv]
The file is written only if it passes submission validation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from kestrel_router import config
from kestrel_router.data import load_test, read_csv
from kestrel_router.model import load_model
from kestrel_router.submission import validate_submission


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(config.PROJECT_ROOT / "predictions.csv"))
    args = parser.parse_args()

    test = load_test()
    model = load_model()
    submission = pd.DataFrame({"request_id": test["request_id"], "team": model.predict(test)})
    sample = read_csv(config.DATA_DIR / config.SAMPLE_SUBMISSION_FILE)

    problems = validate_submission(submission, test["request_id"].tolist(), sample)
    if problems:
        print("NOT WRITTEN - submission invalid:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    out = Path(args.out)
    submission.to_csv(out, index=False, lineterminator="\n")
    print(f"wrote {out} ({len(submission)} rows)")
    print(submission["team"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
