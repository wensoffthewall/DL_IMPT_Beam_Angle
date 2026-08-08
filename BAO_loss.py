import tensorflow as tf
import numpy as np

def sin_cos_mse_loss(y_true, y_pred):
    """
    Computes sine–cosine loss using circular distance for proper angle alignment.
    Converts y_true (normalised [0,1]) to degrees, to obtain its sine and cosine component
    """

    y_true = y_true * 360
    # compute sine and cosine component
    y_true_sincos = _angles_to_sin_cos(y_true)

    # Compute squared Euclidean distance in sine–cosine space
    loss = tf.reduce_mean((y_true_sincos - y_pred) ** 2)

    return loss

def angular_loss(y_true, y_pred, align=False):
    """
    Use cosine for angular loss for normalised angles [0,1]
    """
    # Convert normalised angles to radians
    y_true_rad = y_true * 2 * np.pi
    y_pred_rad = y_pred * 2 * np.pi

    # Align predictions to ground truth
    # Compute minimal angular difference in [-π, π)
    if align:
        y_pred_aligned = _align_angles(y_true_rad, y_pred_rad, mode='radian')
        angle_diff = tf.math.mod(y_pred_aligned - y_true_rad + np.pi, 2 * np.pi) - np.pi
    else:
        angle_diff = tf.math.mod(y_pred_rad - y_true_rad + np.pi, 2 * np.pi) - np.pi

    return tf.reduce_mean(1 - tf.cos(angle_diff))


def _angles_to_sin_cos(y):
    """ Converts angles in degrees to sine–cosine pairs. """
    y_rad = y * (np.pi / 180.0)
    sin_y = tf.sin(y_rad)
    cos_y = tf.cos(y_rad)
    return tf.concat([sin_y, cos_y], axis=-1)

def _align_angles(y_true, y_pred, mode):
    """
    Aligns predicted angles to ground truth based on minimal circular distance.
    """
    distance_matrix = _compute_circular_distance_matrix(y_true, y_pred, mode)
    indices = tf.argmin(distance_matrix, axis=-1)  # Find best matches

    y_pred_aligned = tf.gather(y_pred, indices, batch_dims=1)  # Reorder predictions properly

    return y_pred_aligned

def _compute_circular_distance_matrix(y_true, y_pred, mode):
    """
    Computes a pairwise circular distance matrix between ground truth and predicted angles.
    """
    y_true_exp = tf.expand_dims(y_true, axis=1)  # Shape [batch, 1, num_angles]
    y_pred_exp = tf.expand_dims(y_pred, axis=2)  # Shape [batch, num_angles, 1]

    raw_diff = tf.abs(y_true_exp - y_pred_exp)
    if mode == 'degree':
        circular_diff = tf.minimum(raw_diff, 360 - raw_diff)  # Account for wrap-around
    elif mode == 'radian':
        circular_diff = tf.minimum(raw_diff, 2 * np.pi - raw_diff)  # Account for wrap-around

    return circular_diff



