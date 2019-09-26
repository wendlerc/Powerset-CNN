"""Module containing utility functions specific to implementing models."""
import tensorflow as tf


def training_placeholder():
    """Either gets or creates the boolean placeholder `is_training`.

    The placeholder is initialized to have a default value of False,
    indicating that training is not taking place.
    Thus it is required to pass True to the placeholder
    to indicate training being active.

    Returns:
        tf.placeholder_with_default(False, name='is_training')

    """
    try:
        training = tf.get_default_graph().get_tensor_by_name('is_training:0')
    except KeyError:
        # We need to set this variable scope, otherwise the name of the
        # placeholder would be dependent on the variable scope of the caller
        cur_scope = tf.get_variable_scope().name
        if cur_scope == '':
            training = tf.placeholder_with_default(
                False, name='is_training', shape=[])
        else:
            with tf.variable_scope('/'):
                training = tf.placeholder_with_default(
                    False, name='init_mode', shape=[])
    return training
