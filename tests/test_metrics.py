"""Tests for the metric functions, with y = 0 cases made explicit."""

import numpy as np
import pandas as pd
import pytest

from metrics import TARGETS, evaluate_frame, mae, mase, rmse, smape, wape


def test_mae_known_value():
    assert mae([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0
    # errors 1, 1, 2 -> mean 4/3
    assert mae([1.0, 2.0, 3.0], [2.0, 1.0, 5.0]) == pytest.approx(4 / 3)


def test_rmse_known_value():
    # errors 3, 4 -> sqrt((9+16)/2) = sqrt(12.5)
    assert rmse([0.0, 0.0], [3.0, 4.0]) == pytest.approx(np.sqrt(12.5))


def test_metrics_ignore_nan_in_truth():
    # The NaN position must be dropped, not propagated: sensor outages are not
    # forecast errors. Remaining errors are 1 and 1 -> MAE 1.
    assert mae([1.0, np.nan, 3.0], [2.0, 999.0, 4.0]) == pytest.approx(1.0)
    assert not np.isnan(rmse([1.0, np.nan], [2.0, 999.0]))


def test_all_nan_truth_returns_nan():
    assert np.isnan(mae([np.nan, np.nan], [1.0, 2.0]))


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        mae([1.0, 2.0], [1.0])


# --- y = 0 handling, the part most likely to be silently wrong ---------------


def test_smape_zero_truth_and_zero_pred_is_zero_error():
    # Both exactly 0: the error is 0, not 0/0 and not 100%.
    assert smape([0.0], [0.0]) == 0.0
    assert smape([0.0, 0.0], [0.0, 0.0]) == 0.0


def test_smape_zero_truth_nonzero_pred_is_max_error():
    # |0 - 5| / ((0 + 5)/2) = 2, the maximum of this SMAPE convention.
    assert smape([0.0], [5.0]) == pytest.approx(2.0)


def test_smape_is_bounded_and_symmetric():
    a, b = [1.0, 10.0, 0.0], [2.0, 5.0, 3.0]
    assert 0.0 <= smape(a, b) <= 2.0
    # Swapping truth and prediction leaves SMAPE unchanged.
    assert smape(a, b) == pytest.approx(smape(b, a))


def test_smape_never_divides_by_zero():
    with np.errstate(divide="raise", invalid="raise"):
        assert smape([0.0, 0.0, 4.0], [0.0, 1.0, 4.0]) >= 0.0


def test_wape_survives_individual_zeros():
    # This is why WAPE is used instead of MAPE: a single y = 0 row would make
    # MAPE infinite, but WAPE normalises by the sum.
    y = [0.0, 10.0, 10.0]
    p = [1.0, 10.0, 10.0]
    assert wape(y, p) == pytest.approx(1.0 / 20.0)


def test_wape_all_zero_truth_is_nan_not_inf():
    result = wape([0.0, 0.0], [1.0, 2.0])
    assert np.isnan(result)


def test_wape_is_scale_free():
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([1.5, 2.5, 2.0])
    assert wape(y, p) == pytest.approx(wape(y * 1000, p * 1000))


# --- MASE -------------------------------------------------------------------


def test_mase_below_one_when_better_than_naive():
    y = [1.0, 2.0, 3.0, 4.0]
    good = [1.1, 2.1, 3.1, 4.1]
    naive = [2.0, 3.0, 4.0, 5.0]
    assert mase(y, good, naive) < 1.0


def test_mase_identical_to_naive_is_one():
    y = [1.0, 2.0, 3.0]
    naive = [2.0, 3.0, 4.0]
    assert mase(y, naive, naive) == pytest.approx(1.0)


def test_mase_perfect_naive_returns_nan_not_inf():
    y = [1.0, 2.0]
    assert np.isnan(mase(y, [5.0, 5.0], y))


# --- frame-level evaluation -------------------------------------------------


def _frame(values):
    return pd.DataFrame({c: values for c in TARGETS})


def test_evaluate_frame_has_per_pollutant_and_pooled_rows():
    truth = _frame([1.0, 2.0, 3.0])
    pred = _frame([1.0, 2.0, 3.0])
    res = evaluate_frame(truth, pred)
    assert list(res.index) == TARGETS + ["POOLED"]
    assert res.loc["POOLED", "mae"] == 0.0


def test_pooled_mae_is_the_cell_average_not_the_metric_average():
    # CO on a small scale, the rest large: the pooled MAE must be the mean over
    # all cells, so it is pulled toward the large-scale pollutants.
    truth = pd.DataFrame({c: [0.0, 0.0] for c in TARGETS})
    pred = pd.DataFrame({c: [1.0, 1.0] for c in TARGETS})
    pred["valeur_CO"] = [100.0, 100.0]
    res = evaluate_frame(truth, pred)
    expected = (100.0 * 2 + 1.0 * 8) / 10
    assert res.loc["POOLED", "mae"] == pytest.approx(expected)
