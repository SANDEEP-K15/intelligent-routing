from kestrel_router.analysis import is_multi_intent, pattern_group, template


def test_template_hides_products_and_filler():
    assert template("good morning, water purifier dripping everywhere kindly resolve") == "<p> dripping everywhere"


def test_pattern_groups():
    assert pattern_group("fan stays off though i paid upfront", "Ceiling Fan") == "mentions payment"
    assert pattern_group("water purifier leaking since morning", "Water Purifier") == "purifier fault"
    assert pattern_group("someone contact me regarding the air fryer please", "Air Fryer") == "vague"
    assert pattern_group("hello team, help purifier today", "Water Purifier") == "vague"
    assert pattern_group("how to clean robot vacuum brushes properly", "Robot Vacuum") == "clear"


def test_multi_intent_ignores_greeting_comma():
    assert not is_multi_intent("hello team, emi still pending on my fan")
    assert is_multi_intent("emi still pending on my fan, coupon missing on my fan")
