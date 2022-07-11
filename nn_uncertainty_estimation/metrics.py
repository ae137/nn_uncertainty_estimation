import tensorflow as tf


class AdaptedMseForNllOutput(tf.keras.metrics.MeanSquaredError):
    """Computation of the mean-squared error metric for network that outputs means and logs of variances."""

    def __init__(self, name="adapted_mse_for_nll", dtype=None):
        super().__init__(name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """State update function for mean-squared error metric suitable for NLL style outputs.

        Args:
            y_true: Ground truth values as tensor of shape `[batch_size, d0, .. dN, 1]`.
            y_pred: Predicted values as tensor of shape `[batch_size, d0, .. dN, 2]`,
                    containing predicted mean values in y_pred[..., 0]
                    and predicted logarithms of variances in y_pred[..., 1]
        """

        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)

        tf.debugging.assert_equal(
            tf.rank(y_true),
            tf.rank(y_pred),
            message="Input tensors must have equal rank.",
        )
        tf.debugging.assert_equal(
            tf.shape(y_true)[-1], 1, message="Last dimension of y_true must be 1."
        )
        tf.debugging.assert_equal(
            tf.shape(y_pred)[-1], 2, message="Last dimension of y_pred must be 2."
        )

        y_pred_mu = y_pred[:, 0:1]
        super().update_state(y_true, y_pred_mu, sample_weight)


class AdaptedMaeForNllOutput(tf.keras.metrics.MeanAbsoluteError):
    """Computation of mean-absolute error metric for network that outputs means and logs of variances."""

    def __init__(self, name="adapted_mae_for_nll", dtype=None):
        super().__init__(name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """State update function for mean-average error metric suitable for NLL style outputs.

        Args:
            y_true: Ground truth values as tensor of shape `[batch_size, d0, .. dN, 1]`.
            y_pred: Predicted values as tensor of shape `[batch_size, d0, .. dN, 2]`,
                    containing predicted mean values in y_pred[..., 0]
                    and predicted logarithms of variances in y_pred[..., 1]
        """

        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)

        tf.debugging.assert_equal(
            tf.rank(y_true),
            tf.rank(y_pred),
            message="Input tensors must have equal rank.",
        )
        tf.debugging.assert_equal(
            tf.shape(y_true)[-1], 1, message="Last dimension of y_true must be 1."
        )
        tf.debugging.assert_equal(
            tf.shape(y_pred)[-1], 2, message="Last dimension of y_pred must be 2."
        )

        y_pred_mu = y_pred[:, 0:1]
        super().update_state(y_true, y_pred_mu, sample_weight)
