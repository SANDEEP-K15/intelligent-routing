import pytest

from kestrel_router import config, metrics


def test_score_and_per_team():
    y = ["Repairs", "Billing", "Repairs", "Installs & Demo"]
    p = ["Repairs", "Billing", "Billing", "Installs & Demo"]
    s = metrics.score(y, p)
    assert s["accuracy"] == pytest.approx(0.75)
    assert s["n"] == 4
    pt = metrics.per_team(y, p)
    assert list(pt.index) == list(config.CURRENT_TEAMS)
    assert pt.loc["Billing", "precision"] == pytest.approx(0.5)
    assert pt.loc["Repairs", "recall"] == pytest.approx(0.5)


def test_confusion_matrix_shape_and_total():
    y = ["Repairs", "Billing"]
    cm = metrics.confusion(y, ["Repairs", "Repairs"])
    assert cm.shape == (7, 7)
    assert cm.to_numpy().sum() == 2


def test_misroute_cost_uses_policy_rates():
    # 10 misroutes x (1.5 transfers x Rs 305 + Rs 260 extra contact)
    assert metrics.misroute_cost(10, 1.5) == pytest.approx(10 * (1.5 * 305 + 260))


def test_monthly_model_cost_is_zero():
    c = metrics.monthly_model_cost(700)
    assert c["monthly_model_api_cost_rs"] == 0
    assert c["bot_licence_per_month_rs"] == pytest.approx(26666.67)
