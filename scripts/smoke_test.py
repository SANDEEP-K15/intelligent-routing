"""Post-install smoke test. Needs only requirements.txt and the model artefact (no assignment data).

Usage:
  python -m scripts.smoke_test                         # load the model and route sample requests
  python -m scripts.smoke_test --url http://localhost:8000   # also exercise a running service

Exit code 0 = healthy, 1 = failed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from kestrel_router import config
from kestrel_router.model import ModelArtefactError, load_model

SAMPLES = [
    {"request_text": "my ceiling fan makes a grinding noise when it runs", "product_family": "Ceiling Fan"},
    {"request_text": "please send the gst invoice again, the company name is wrong"},
    {"request_text": "when will someone come to install the new cooktop", "product_family": "Induction Cooktop"},
    {"request_text": "how do I descale the purifier tank", "product_family": "Water Purifier"},
    {"request_text": "?"},
]


def check_result(result: dict) -> list[str]:
    problems = []
    if result.get("team") not in config.CURRENT_TEAMS:
        problems.append(f"team {result.get('team')!r} is not a current team")
    if not 0 <= float(result.get("confidence", -1)) <= 1:
        problems.append(f"confidence {result.get('confidence')!r} outside [0, 1]")
    if not result.get("reasons"):
        problems.append("no reasons returned")
    return problems


def check_local(model_path: Path) -> list[str]:
    try:
        model = load_model(model_path)
    except ModelArtefactError as exc:
        return [str(exc)]
    problems = []
    t0 = time.perf_counter()
    for sample in SAMPLES:
        result = model.route(**sample)
        problems += [f"local {sample['request_text'][:30]!r}: {p}" for p in check_result(result)]
    ms = (time.perf_counter() - t0) / len(SAMPLES) * 1000
    print(f"local model OK: {len(SAMPLES)} requests routed, {ms:.1f} ms each")
    return problems


def _request(url: str, payload: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def check_service(base: str) -> list[str]:
    base = base.rstrip("/")
    problems = []
    try:
        status, health = _request(f"{base}/api/health")
    except urllib.error.URLError as exc:
        return [f"service not reachable at {base}: {exc.reason}"]
    if status != 200 or health.get("status") != "ok":
        problems.append(f"/api/health returned {status} {health}")
    status, body = _request(f"{base}/api/route", SAMPLES[0])
    if status != 200:
        problems.append(f"/api/route returned {status}")
    else:
        problems += [f"service: {p}" for p in check_result(body)]
    status, _ = _request(f"{base}/api/route", {"request_text": "   "})
    if status != 422:
        problems.append(f"blank request returned {status}, expected 422")
    with urllib.request.urlopen(f"{base}/", timeout=10) as resp:
        if resp.status != 200 or b"/api/route" not in resp.read():
            problems.append("UI page not served")
    if not problems:
        print(f"service OK at {base}: health, route, validation and UI checked")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=config.MODEL_PATH)
    parser.add_argument("--url", help="base URL of a running service to check as well")
    args = parser.parse_args(argv)
    problems = check_local(args.model)
    if args.url:
        problems += check_service(args.url)
    if problems:
        print("SMOKE TEST FAILED:\n  " + "\n  ".join(problems))
        return 1
    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
