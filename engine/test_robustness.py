
import pandas as pd
import pytest

from engine.robustness import evaluate_moving_average_parameters


def test_valid_moving_average_parameters():
    prices = pd.Series(
        [100, 102, 101, 105, 107, 104, 110, 112],
        index=pd.date_range("2024-01-01", periods=8, freq="D"),
    )

    results = evaluate_moving_average_parameters(
        prices=prices,
        parameter_pairs=[(2, 3), (2, 4)],
        transaction_cost=0.001,
    )

    assert len(results) == 2
    assert "Short_Window" in results.columns
    assert "Long_Window" in results.columns
    assert "Total Return" in results.columns
    assert results["Short_Window"].tolist() == [2, 2]
    assert results["Long_Window"].tolist() == [3, 4]


@pytest.mark.parametrize(
    "parameter_pairs",
    [
        [(0, 3)],
        [(-1, 3)],
        [(3, 3)],
        [(4, 3)],
    ],
)
def test_invalid_moving_average_parameters_raise_error(parameter_pairs):
    prices = pd.Series(
        [100, 102, 101, 105, 107, 104, 110, 112],
        index=pd.date_range("2024-01-01", periods=8, freq="D"),
    )

    with pytest.raises(ValueError):
        evaluate_moving_average_parameters(
            prices=prices,
            parameter_pairs=parameter_pairs,
        )
