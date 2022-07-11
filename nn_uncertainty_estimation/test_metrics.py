"""Unit tests for metrics.py."""

import numpy as np
import pytest
import tensorflow as tf

from nn_uncertainty_estimation.metrics import (
    AdaptedMseForNllOutput,
    AdaptedMaeForNllOutput,
)


class TestAdaptedMseForNllOutput:
    """Check that negative log likelihood function throws exception on bad inputs."""

    def test_scalar_inputs(self) -> None:
        """Expect that function fails with scalar inputs."""

        loss_function = AdaptedMseForNllOutput()

        with pytest.raises(tf.errors.InvalidArgumentError):
            loss_function.update_state(y_true=1.0, y_pred=2.0)

    def test_bad_shapes(self) -> None:
        """Expect that function fails with inputs of bad shapes."""

        loss_function = AdaptedMseForNllOutput()

        with pytest.raises(tf.errors.InvalidArgumentError):
            # Wrong shape of y_pred
            loss_function.update_state(
                y_true=np.array([[1.0]]), y_pred=np.array([[1.0]])
            )

        with pytest.raises(tf.errors.InvalidArgumentError):
            # Wrong shape of y_true
            loss_function.update_state(
                y_true=np.array([1.0]), y_pred=np.array([[1.0, 2.0]])
            )

        with pytest.raises(tf.errors.InvalidArgumentError):
            # Wrong shape of y_true
            loss_function.update_state(
                y_true=np.array([[1.0, 1.0]]), y_pred=np.array([[1.0, 1.0]])
            )


class TestAdaptedMaeForNllOutput:
    """Check that negative log likelihood function throws exception on bad inputs."""

    def test_scalar_inputs(self) -> None:
        """Expect that function fails with scalar inputs."""

        loss_function = AdaptedMaeForNllOutput()

        with pytest.raises(tf.errors.InvalidArgumentError):
            loss_function.update_state(y_true=1.0, y_pred=2.0)

    def test_bad_shapes(self) -> None:
        """Expect that function fails with inputs of bad shapes."""

        loss_function = AdaptedMaeForNllOutput()

        with pytest.raises(tf.errors.InvalidArgumentError):
            # Wrong shape of y_pred
            loss_function.update_state(
                y_true=np.array([[1.0]]), y_pred=np.array([[1.0]])
            )

        with pytest.raises(tf.errors.InvalidArgumentError):
            # Wrong shape of y_true
            loss_function.update_state(
                y_true=np.array([1.0]), y_pred=np.array([[1.0, 2.0]])
            )

        with pytest.raises(tf.errors.InvalidArgumentError):
            # Wrong shape of y_true
            loss_function.update_state(
                y_true=np.array([[1.0, 1.0]]), y_pred=np.array([[1.0, 1.0]])
            )
