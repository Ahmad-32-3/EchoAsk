from app.conflict_guard import has_conflict


def test_no_conflict_for_true_paraphrase():
    assert not has_conflict(
        "How do I reset my password?", "I forgot my password, what do I do?"
    )


def test_conflict_on_differing_numbers():
    assert has_conflict("Is my order #4521 delayed?", "Is my order #7788 delayed?")


def test_conflict_on_billing_cadence():
    assert has_conflict(
        "What's the refund policy for monthly plans?",
        "What's the refund policy for annual plans?",
    )


def test_conflict_on_role_including_plural():
    assert has_conflict(
        "Do you offer a discount for students?", "Do you offer a discount for teachers?"
    )


def test_conflict_on_currency():
    assert has_conflict(
        "What is the price of the annual plan in USD?",
        "What is the price of the annual plan in EUR?",
    )


def test_conflict_on_negation_mismatch():
    assert has_conflict(
        "Why isn't my payment going through?", "Why is my payment going through twice?"
    )


def test_no_false_conflict_when_groups_dont_apply():
    assert not has_conflict(
        "Can you help me update my billing address?",
        "Could you help me change my billing address?",
    )


def test_known_gap_not_caught_by_design():
    """Documents a real, known limitation rather than hiding it: a novel
    entity swap outside the watched vocabulary is NOT caught. This test
    exists so a future change to the watchlist doesn't silently claim
    coverage it doesn't have without someone noticing this assertion flip."""
    assert not has_conflict(
        "Can I use the API without a paid plan?",
        "Can I use the API without a support contract?",
    )
