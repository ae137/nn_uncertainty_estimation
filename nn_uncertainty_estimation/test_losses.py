"""Unit tests for losses.py"""

import pytest
from typing import Tuple

import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore

import nn_uncertainty_estimation as nn_ue


class TestNegativeLogLikelihoodLossBadInputs:
    """Check that negative log likelihood function throws exception on bad inputs."""

    def test_scalar_inputs(self) -> None:
        """Expect that function fails with scalar inputs."""

        loss_function = nn_ue.losses.NegativeLogLikelihoodLoss()

        with pytest.raises(tf.errors.InvalidArgumentError):
            loss_function(y_true=1., y_pred=2.)

    def test_bad_shapes(self) -> None:
        """Expect that function fails with inputs of bad shapes."""

        loss_function = nn_ue.losses.NegativeLogLikelihoodLoss()

        with pytest.raises(tf.errors.InvalidArgumentError):
            # Wrong shape of y_pred
            loss_function(y_true=np.array([[1.]]), y_pred=np.array([[1.]]))

        with pytest.raises(tf.errors.InvalidArgumentError):
            # Wrong shape of y_true
            loss_function(y_true=np.array([1.]), y_pred=np.array([[1., 2.]]))

        with pytest.raises(tf.errors.InvalidArgumentError):
            # Wrong shape of y_true
            loss_function(y_true=np.array([[1., 1.]]), y_pred=np.array([[1., 1.]]))


@pytest.mark.parametrize("shape", [(1,), (5,), (1, 5,), (20, 5,), (8, 4, 2,)])
def test_equivalence_nll_mse_for_suitable_inputs_simple(shape: Tuple[int, ...]) -> None:
    """Check that negative log likelihood loss function yields results equal to MSE for suitable inputs.

        For log_var = y_pred[..., 1] == 0, NLL should be equivalent to MSE.
    """
    rnd = np.random.default_rng(seed=3)

    y_true = rnd.normal(size=(*shape, 1))
    y_pred = np.zeros((*shape, 2))
    y_pred[..., 0:1] = rnd.normal(size=(*shape, 1))

    # Check that non-reduced results coincide:
    nll = nn_ue.losses.NegativeLogLikelihoodLoss().call(y_true, y_pred)
    mse = 0.5 * tf.keras.losses.mean_squared_error(y_true, y_pred[..., 0:1])

    np.testing.assert_array_almost_equal(nll, mse)

    # Check that reduced results coincide
    nll_reduced = nn_ue.losses.NegativeLogLikelihoodLoss()(y_true, y_pred)
    mse_reduced = 0.5 * tf.keras.losses.MeanSquaredError()(y_true, y_pred[..., 0:1])

    np.testing.assert_array_almost_equal(nll_reduced.numpy(), mse_reduced.numpy())


class TestWithNontrivialInputs():
    """Check that negative log likelihood loss is correctly computed with non-trivial inputs."""

    def test_simple(self) -> None:
        """Simple test."""
        y_true = np.array([[1.]])
        y_pred = np.array([[2., 1.]])

        nll = nn_ue.losses.NegativeLogLikelihoodLoss().call(y_true, y_pred)
        nll_expected = 0.5 + 0.5 * np.exp(-1)

        assert nll == nll_expected

    def test_bigger(self) -> None:
        """Somewhat extended test."""
        y_true = np.array([[-1.], [3.]])
        y_pred = np.array([[1., 2.], [4., 3.]])

        nll = nn_ue.losses.NegativeLogLikelihoodLoss().call(y_true, y_pred)
        nll_expected = 0.5 * np.array([2 + 2**2 / np.exp(2), 3. + 1**2 / np.exp(3)])

        np.testing.assert_array_almost_equal(nll, nll_expected)
