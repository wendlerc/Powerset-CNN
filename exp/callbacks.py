import os
import json
from tempfile import NamedTemporaryFile

import numpy as np
import tensorflow as tf


class LogTrainingLossCallback:
    """Logging of loss during training intop sacred run."""

    def __init__(self, run):
        """Create logger callback.

        Log the training loss using the sacred metrics API.

        Args:
            run: Sacred run
        """
        self.run = run

    def __call__(self, average_loss, std_loss, i, **kwargs):
        """Log average loss across batches and stddev."""
        self.run.log_scalar('training.loss', average_loss, i)
        self.run.log_scalar('training.loss.std', std_loss, i)
        return False


class TestCallback:
    def __init__(self, dataset, run, metrics={}):
        self.dataset = dataset
        self.run = run
        self.log = {k: [] for k in metrics.keys()}
        self.predictions = None
        self.metrics = metrics

    def __call__(self, sess, model, i, batch_size, **kwargs):
        if i%100 != 0:
            return False

        if self.predictions is None:
            tf_dataset = self.dataset.get_testing_data().prefetch(batch_size)
            tf_dataset = tf_dataset.batch(batch_size)
            self.iterator = tf_dataset.make_initializable_iterator()
            next_element = self.iterator.get_next()
            inputs, targets = next_element
            self.predictions = model.predict(inputs)
            self.labels = targets
        sess.run(self.iterator.initializer)
        cur_predictions = []
        labels = []
        while True:
            try:
                l, p = sess.run([self.labels, self.predictions])
                labels.append(l)
                cur_predictions.append(p)
            except tf.errors.OutOfRangeError:
                break

        cur_predictions = np.concatenate(cur_predictions, axis=0)
        labels = np.concatenate(labels, axis=0)

        for metric_name, metric_fn in self.metrics.items():
            score = metric_fn(labels, cur_predictions)
            print(f'Test {metric_name}: {score:5.5f}')
            self.run.log_scalar(f'test.{metric_name}', score, i)
            self.log[metric_name] += [score]

        return False



class ModelSaver:
    """Callback that saves the model.

    Callback that saves the model by storing its configuration values into a
    json file and its weights into a checkpoint file. The files are added as
    artifacts to a provided sacred run.

    The model is only saved if it's average loss improves.
    """

    def __init__(self, run):
        self.run = run
        self.saver = None
        try:
            self.savedir = self.run.observers[0].dir
        except Exception:
            self.savedir = None
        self.loss_last_saved = np.inf

    def __call__(self, model, sess, i, average_loss, **kwargs):
        if i%100 != 0:
            return False

        if self.savedir is None:
            # Nowhere to save to, skipping
            return False

        # Saver not initialized yet
        if self.saver is None:
            parameters, variables = model.save()
            with NamedTemporaryFile(mode='w', suffix='.json') as f:
                json.dump(parameters, f, cls=NumpyEncoder)
                f.flush()
                os.fsync(f)
                self.run.add_artifact(f.name, 'model_parameters.json')

            self.saver = tf.train.Saver(model.variables, sharded=True)

        if average_loss < self.loss_last_saved:
            self.saver.save(
                sess, os.path.join(self.savedir, 'model_checkpoint'))
            self.loss_last_saved = average_loss

        return False


class StopOnNaNCallback:
    """Callback to stop training in case of NaN, inf or -inf loss."""

    def __call__(self, cur_loss, **kwargs):
        return not np.isfinite(cur_loss)


# Useful piece of code based on:
# https://github.com/mpld3/mpld3/issues/434#issuecomment-340255689
# Using numpy type hierarchy for isinstance instead
# Allows numpy objects to be serialized with json.dump
class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.float):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

