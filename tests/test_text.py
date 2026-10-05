"""Text-cleaning tests. All strings are synthetic; IDs are fictitious (KO0000001, SR00001)."""

import pytest

from kestrel_router.text import clean_text, remove_identifiers, repair_mojibake


def test_repairs_zoho_mojibake():
    assert repair_mojibake("urgÃ©nt: my heater needs a look") == "urgént: my heater needs a look"
    assert repair_mojibake("still waiting â€¦") == "still waiting …"
    assert repair_mojibake("plÃ©ase reply â€“ thanks") == "pléase reply – thanks"


def test_clean_text_leaves_clean_text_alone():
    assert repair_mojibake("my kettle light stays off") == "my kettle light stays off"


def test_clean_text_full_pipeline():
    raw = "PlÃ©ase Check My Heater order KO0000001 reg no SR00001 â€¦"
    assert clean_text(raw) == "please check my heater order reg no"


def test_identifiers_removed_but_error_codes_kept():
    assert "e9" in clean_text("display shows error code E9")
    assert "ko0000001" not in clean_text("order KO0000001 still pending")
    assert remove_identifiers("SR00001 and KO0000002").split() == ["and"]


def test_clean_text_is_idempotent():
    s = "Hi, my cooktop shuts off on Ã©very use  asap"
    assert clean_text(clean_text(s)) == clean_text(s)


def test_crm_and_zoho_versions_clean_identically():
    assert clean_text("urgÃ©nt: the motor smÃ©lls odd") == clean_text("urgent: the motor smells odd")


def test_clean_text_rejects_non_strings():
    with pytest.raises(TypeError):
        clean_text(None)  # type: ignore[arg-type]


def test_undecodable_text_is_returned_unchanged():
    # Contains a marker but is not valid cp1252->utf8 mojibake.
    assert repair_mojibake("Ã alone") == "Ã alone"
