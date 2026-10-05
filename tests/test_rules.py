from kestrel_router import config, rules
from kestrel_router.data import add_model_inputs
from tests.conftest import make_requests


def test_rules_baseline_outputs_current_teams():
    preds = rules.predict(add_model_inputs(make_requests()))
    assert set(preds) <= set(config.CURRENT_TEAMS)


def test_rules_encode_known_bot_habits():
    assert rules.predict_one("purifier leaking", "Water Purifier") == "Filters & Consumables"
    assert rules.predict_one("fan not turning on i paid extra", "Ceiling Fan") == "Billing"
    assert rules.predict_one("someone contact me about my fan", "Ceiling Fan") == "Repairs"
    assert rules.predict_one("how to clean the fan blades properly", "Ceiling Fan") in config.CURRENT_TEAMS
