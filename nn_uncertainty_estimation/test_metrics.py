"""Unit tests for metrics.py."""

import pytest

import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore

from nn_uncertainty_estimation.metrics import (
    AdaptedMseForNllOutput,
    AdaptedMaeForNllOutput
)


class TestAdaptedMseForNllOutput:
    """Check that negative log likelihood function throws exception on bad inputs."""

    def test_scalar_inputs(self) -> None:
        """Expect that function fails with scalar inputs."""

        loss_function = AdaptedMseForNllOutput()

        with pytest.raises(tf.errors.InvalidArgumentError):
            loss_function(y_true=1., y_pred=2.)

    def test_bad_shapes(self) -> None:
        """Expect that function fails with inputs of bad shapes."""

        loss_function = AdaptedMseForNllOutput()

        with pytest.raises(tf.errors.InvalidArgumentError):
            # Wrong shape of y_pred
            loss_function(y_true=np.array([[1.]]), y_pred=np.array([[1.]]))

        with pytest.raises(tf.errors.InvalidArgumentError):
            # Wrong shape of y_true
            loss_function(y_true=np.array([1.]), y_pred=np.array([[1., 2.]]))

        with pytest.raises(tf.errors.InvalidArgumentError):
            # Wrong shape of y_true
            loss_function(y_true=np.array([[1., 1.]]), y_pred=np.array([[1., 1.]]))


class TestAdaptedMaeForNllOutput:
    """Check that negative log likelihood function throws exception on bad inputs."""

    def test_scalar_inputs(self) -> None:
        """Expect that function fails with scalar inputs."""

        loss_function = AdaptedMaeForNllOutput()

        with pytest.raises(tf.errors.InvalidArgumentError):
            loss_function(y_true=1., y_pred=2.)

    def test_bad_shapes(self) -> None:
        """Expect that function fails with inputs of bad shapes."""

        loss_function = AdaptedMaeForNllOutput()

        with pytest.raises(tf.errors.InvalidArgumentError):
            # Wrong shape of y_pred
            loss_function(y_true=np.array([[1.]]), y_pred=np.array([[1.]]))

        with pytest.raises(tf.errors.InvalidArgumentError):
            # Wrong shape of y_true
            loss_function(y_true=np.array([1.]), y_pred=np.array([[1., 2.]]))

        with pytest.raises(tf.errors.InvalidArgumentError):
            # Wrong shape of y_true
            loss_function(y_true=np.array([[1., 1.]]), y_pred=np.array([[1., 1.]]))
