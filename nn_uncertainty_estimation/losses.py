"""Collection of custom loss functions for Tensorflow / Keras."""

import tensorflow as tf


class NegativeLogLikelihoodLoss(tf.keras.losses.Loss):
    """Negative log-likelihood loss function."""

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Call function for negative log likelihood loss.

        Args:
            y_true: Ground truth values as tensor of shape `[batch_size, d0, .. dN, 1]`.
            y_pred: Predicted values as tensor of shape `[batch_size, d0, .. dN, 2]`,
                    containing predicted mean values in y_pred[..., 0]
                    and predicted logarithms of variances in y_pred[..., 1]

        Returns:
            Negative log likelihood losses as tensor of shape `[batch_size, d0, .. dN]`.
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

        y_pred_mu = y_pred[..., 0:1]
        y_pred_log_var = y_pred[..., 1:2]
        inverse_y_pred_var = tf.exp(-y_pred_log_var)

        mse = tf.math.squared_difference(y_true, y_pred_mu)

        var_contribution = 0.5 * y_pred_log_var
        error_term = 0.5 * mse * inverse_y_pred_var

        return tf.reduce_mean(error_term + var_contribution, axis=-1)
