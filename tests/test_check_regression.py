from scripts.check_regression import compare_evaluations


def test_regression_comparison_allows_improvements_and_small_accuracy_rounding():
    baseline = {
        "matched": 5,
        "unexpected": 3,
        "accuracy": 0.5,
    }
    current = {
        "matched": 6,
        "unexpected": 2,
        "accuracy": 0.4991,
    }

    passed, failures = compare_evaluations(baseline, current)

    assert passed
    assert failures == []


def test_regression_comparison_fails_for_each_regression_condition():
    baseline = {
        "matched": 5,
        "unexpected": 3,
        "accuracy": 0.5,
    }
    current = {
        "matched": 4,
        "unexpected": 4,
        "accuracy": 0.4989,
    }

    passed, failures = compare_evaluations(baseline, current)

    assert not passed
    assert failures == [
        "matched decreased",
        "unexpected increased",
        "accuracy dropped beyond tolerance",
    ]


def test_regression_comparison_fails_when_current_evaluation_is_missing():
    passed, failures = compare_evaluations(
        {"matched": 5, "unexpected": 3, "accuracy": 0.5},
        None,
    )

    assert not passed
    assert failures == ["missing current evaluation"]
