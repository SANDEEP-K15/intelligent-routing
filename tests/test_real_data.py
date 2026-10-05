"""Checks against the real assignment files. Skipped automatically when the data is absent
(for example in a public clone of the repository)."""

import pytest

from kestrel_router import config
from kestrel_router.data import load_test, load_train, read_csv
from kestrel_router.submission import validate_submission

pytestmark = pytest.mark.skipif(
    not (config.DATA_DIR / config.TRAIN_FILE).exists(), reason="assignment data not present"
)


def test_real_train_and_test_load():
    train = load_train()
    test = load_test()
    assert len(train) == 10822
    assert len(test) == 2178
    assert set(train["team"]) == set(config.CURRENT_TEAMS)
    assert train["text_clean"].str.contains("Ã|â€").sum() == 0
    assert test["created_at"].min() > train["created_at"].max()


@pytest.mark.skipif(not (config.PROJECT_ROOT / "predictions.csv").exists(), reason="no predictions yet")
def test_predictions_file_valid():
    sub = read_csv(config.PROJECT_ROOT / "predictions.csv")
    test = read_csv(config.DATA_DIR / config.TEST_FILE)
    sample = read_csv(config.DATA_DIR / config.SAMPLE_SUBMISSION_FILE)
    assert validate_submission(sub, test["request_id"].tolist(), sample) == []
    assert len(sub) == 2178
