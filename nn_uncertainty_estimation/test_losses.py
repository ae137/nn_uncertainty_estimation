"""Unit tests for losses.py"""

from typing import Tuple

import numpy as np
import pytest
import tensorflow as tf

from nn_uncertainty_estimation.losses import NegativeLogLikelihoodLoss


class TestNegativeLogLikelihoodLossBadInputs:
    """Check that negative log likelihood function throws exception on bad inputs."""

    def test_scalar_inputs(self) -> None:
        """Expect that function fails with scalar inputs."""

        loss_function = NegativeLogLikelihoodLoss()

        with pytest.raises(tf.errors.InvalidArgumentError):
            loss_function.call(y_true=tf.constant([1.0]), y_pred=tf.constant([2.0]))

    def test_bad_shapes(self) -> None:
        """Expect that function fails with inputs of bad shapes."""

        loss_function = NegativeLogLikelihoodLoss()

        with pytest.raises(tf.errors.InvalidArgumentError):
            # Wrong shape of y_pred
            loss_function.call(y_true=tf.constant([[1.0]]), y_pred=tf.constant([[1.0]]))

        with pytest.raises(tf.errors.InvalidArgumentError):
            # Wrong shape of y_true
            loss_function.call(
                y_true=tf.constant([1.0]), y_pred=tf.constant([[1.0, 2.0]])
            )

        with pytest.raises(tf.errors.InvalidArgumentError):
            # Wrong shape of y_true
            loss_function.call(
                y_true=tf.constant([[1.0, 1.0]]), y_pred=tf.constant([[1.0, 1.0]])
            )


@pytest.mark.parametrize(
    "shape",
    [
        (1,),
        (5,),
        (
            1,
            5,
        ),
        (
            20,
            5,
        ),
        (
            8,
            4,
            2,
        ),
    ],
)
def test_equivalence_nll_mse_for_suitable_inputs_simple(shape: Tuple[int, ...]) -> None:
    """Check that negative log likelihood loss function yields results equal to MSE for suitable inputs.

    For log_var = y_pred[..., 1] == 0, NLL should be equivalent to MSE.
    """
    rnd = np.random.default_rng(seed=3)

    y_true = rnd.normal(size=(*shape, 1))
    y_pred = np.zeros((*shape, 2))
    y_pred[..., 0:1] = rnd.normal(size=(*shape, 1))

    # Check that non-reduced results coincide:
    nll = NegativeLogLikelihoodLoss().call(tf.constant(y_true), tf.constant(y_pred))
    mse = 0.5 * tf.keras.losses.mean_squared_error(y_true, y_pred[..., 0:1])

    np.testing.assert_array_almost_equal(nll, mse)

    # Check that reduced results coincide
    nll_reduced = NegativeLogLikelihoodLoss()(tf.constant(y_true), tf.constant(y_pred))
    mse_reduced = 0.5 * tf.keras.losses.MeanSquaredError()(y_true, y_pred[..., 0:1])

    np.testing.assert_array_almost_equal(nll_reduced.numpy(), mse_reduced.numpy())


class TestWithNontrivialInputs:
    """Check that negative log likelihood loss is correctly computed with non-trivial inputs."""

    def test_simple(self) -> None:
        """Simple test."""
        y_true = np.array([[1.0]])
        y_pred = np.array([[2.0, 1.0]])

        nll = NegativeLogLikelihoodLoss().call(tf.constant(y_true), tf.constant(y_pred))
        nll_expected = 0.5 + 0.5 * np.exp(-1)

        assert nll == nll_expected

    def test_bigger(self) -> None:
        """Somewhat extended test."""
        y_true = np.array([[-1.0], [3.0]])
        y_pred = np.array([[1.0, 2.0], [4.0, 3.0]])

        nll = NegativeLogLikelihoodLoss().call(tf.constant(y_true), tf.constant(y_pred))
        nll_expected = 0.5 * np.array(
            [2 + 2**2 / np.exp(2), 3.0 + 1**2 / np.exp(3)]
        )

        np.testing.assert_array_almost_equal(nll, nll_expected)
