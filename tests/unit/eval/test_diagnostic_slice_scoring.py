from eval.longmemeval.diagnostic_slice import _gold_in_text


def test_short_amount_requires_same_numeric_identity() -> None:
    assert not _gold_in_text("$400,000", "$350,000")
    assert _gold_in_text("$400,000", "400,000")
    assert _gold_in_text("$400,000", "400000")


def test_short_duration_requires_same_number_and_unit() -> None:
    assert _gold_in_text("8 days", "it took 8 days")
    assert not _gold_in_text("8 days", "3 days")


def test_short_number_requires_token_boundary() -> None:
    assert _gold_in_text("2", "2")
    assert not _gold_in_text("2", "1")
    assert not _gold_in_text("2", "12")


def test_short_location_phrase_does_not_use_token_overlap() -> None:
    assert not _gold_in_text("Chicago", "the suburbs")


def test_short_phrase_exact_matches_pass() -> None:
    assert _gold_in_text("Business Administration", "Business Administration")
    assert _gold_in_text("Dr. Arati Prabhakar", "Dr. Arati Prabhakar")

