"""Module with utitlity functions."""
import tensorflow as tf


def add_gradient_summaries(gradients):
    """Add summaries of variable gradients."""
    tf.summary.histogram(
        'gradients/global_norm',
        tf.global_norm(list(zip(*gradients))[0])
    )
    for gradient, variable in gradients:
        if isinstance(gradient, tf.IndexedSlices):
            grad_values = gradient.values
        else:
            grad_values = gradient

        if grad_values is not None:
            var_name = variable.name.replace(":", "_")
        tf.summary.histogram(f"gradients/{var_name}", grad_values)
        tf.summary.scalar(
            f"gradient_norm/{var_name}",
            tf.global_norm([grad_values])
        )


def add_variable_summaries(variables):
    """Add histogram and scalar norm summaries for variables."""
    for variable in variables:
        var_name = variable.name.replace(":", "_")
        tf.summary.histogram(f"variables/{var_name}", variable)
        tf.summary.scalar(f"variables/{var_name}", tf.norm(variable))


def print_call(fn):
    """Print function arguments and return values.

    Just add as a decorator to a function for
    which you want to analyse the behavior. It automatically prints inputs and
    outputs of each function call

    """
    def wrapping_fn(*args, **kwargs):
        print('Called', fn.__name__, 'with parameters:\n', args, kwargs)
        res = fn(*args, **kwargs)
        print('Returned:', res, '\n\n')
        return res
    return wrapping_fn


# Bottom code is borrowed from: https://stackoverflow.com/a/47884927
def py_func_decorator(output_types=None, output_shapes=None,
                      stateful=True, name=None):
    """Wrap a python function into a tensorflow py_func."""
    def decorator(func):
        def call(*args, **kwargs):
            return tf.contrib.framework.py_func(
                func=func,
                args=args, kwargs=kwargs,
                output_types=output_types, output_shapes=output_shapes,
                stateful=stateful, name=name
            )
        return call
    return decorator


def from_indexable(iterator, output_types, output_shapes=None,
                   num_parallel_calls=None, stateful=True, name=None):
    """Create tensorflow dataset from indexable python object."""
    ds = tf.data.Dataset.range(len(iterator))

    @py_func_decorator(output_types, output_shapes, stateful=stateful,
                       name=name)
    def index_to_entry(index):
        return iterator[index]

    return ds.map(index_to_entry, num_parallel_calls=num_parallel_calls)
