"""The post-install smoke test runs against the synthetic model artefact."""

from scripts import smoke_test


def test_smoke_test_passes_with_valid_artefact(model_path, capsys):
    assert smoke_test.main(["--model", str(model_path)]) == 0
    assert "SMOKE TEST PASSED" in capsys.readouterr().out


def test_smoke_test_fails_without_artefact(tmp_path, capsys):
    assert smoke_test.main(["--model", str(tmp_path / "missing.joblib")]) == 1
    assert "not found" in capsys.readouterr().out


def test_smoke_test_reports_unreachable_service(model_path):
    problems = smoke_test.check_service("http://127.0.0.1:9")
    assert problems and "not reachable" in problems[0]


def test_check_result_flags_bad_output():
    assert smoke_test.check_result({"team": "Repairs", "confidence": 0.9, "reasons": ["x"]}) == []
    assert len(smoke_test.check_result({"team": "Installations", "confidence": 2, "reasons": []})) == 3
