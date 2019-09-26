"""Module containing convenience functions for training."""
import numpy as np
import tensorflow as tf

from .utils import add_gradient_summaries, add_variable_summaries
from .datasets import Dataset
from .models import BaseModel, training_placeholder


def train_model_with_dataset(
        model: BaseModel, dataset: Dataset, epochs: int, batch_size=32,
        lr=1e-3, decay_factor=1, print_progress=True, do_gradient_summary=False, callbacks=[], tf_logdir=None,
        load_model=None):
    """Train a model with a provided dataset.

    Args:
        model: Model instance
        dataset: Dataset instance
        epochs: Number of epochs to train
        batch_size: Batch size
        lr: Learning rate
        decay_factor: Factor for exponential learning rate decay
        print_progress: Output training loss after each batch
        do_gradient_summary: flag to toggle gradient summaries
        callbacks: List of callbacks to execute after each epoch
        tf_logdir: Path to tensorboard logging directory. Don't log if set to
            `None`
        load_model: Load model from provided path
    """
    dataset_size = dataset.size
    tf_dataset = dataset.get_tf_dataset()\
        .shuffle(dataset_size, reshuffle_each_iteration=True)\
        .prefetch(dataset_size)
    # Batch iterator
    tf_dataset = tf_dataset.batch(batch_size)
    iterator = tf_dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    # Loss
    loss = model.loss(*next_element)
    prediction = model.predict(next_element[0])
    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(next_element[1], 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float64))

    # training mode
    # We use this placeholder to indicate to the model if we are training or
    # not.
    # Default is not training (`is_training = False`), s.t. we should feed true
    # while training.
    is_training = training_placeholder()

    # Setup optimizer
    global_step = tf.train.get_or_create_global_step()
    lr = tf.train.exponential_decay(lr, global_step, dataset_size // batch_size, decay_factor, staircase=True)
    # Get update ops and control dependencies in case we are using batchnorm
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.name_scope('optimizer'), tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(lr)
        gradient_value_pairs = optimizer.compute_gradients(loss)
        if do_gradient_summary:
            add_gradient_summaries(gradient_value_pairs)
            add_variable_summaries(list(zip(*gradient_value_pairs))[1])
        gradients, variables = zip(*gradient_value_pairs)
        clipped_gradients, global_norm = tf.clip_by_global_norm(gradients, 0.5)
        tf.summary.scalar("gradients", global_norm)
        # In case gradient clipping is needed
        # train_op = optimizer.apply_gradients(
        #     zip(clipped_gradients, variables), global_step=global_step)
        train_op = optimizer.apply_gradients(
            gradient_value_pairs, global_step=global_step)
    summary = tf.summary.merge_all()

    sess = tf.Session()

    if tf_logdir:
        logger = tf.summary.FileWriter(tf_logdir, sess.graph)

    # Initialize variables
    sess.run(tf.global_variables_initializer())

    if load_model:
        saver = tf.train.Saver(model.variables)
        saver.restore(sess, load_model)
        print('Restored model from', load_model)

    average_losses = []
    std_losses = []
    average_accs = []
    std_accs = []
    stop_training = False

    for i in range(epochs):
        sess.run(iterator.initializer)
        if stop_training:
            print('Stopping training due to callback')
            break
        losses = []
        accs = []
        batch = 0
        while True:
            try:
                if tf_logdir:
                    cur_loss, cur_acc, summary_, step_, _ = sess.run(
                        [loss, accuracy, summary, global_step, train_op],
                        feed_dict={is_training: True}
                    )
                    logger.add_summary(summary_, global_step=step_)
                else:
                    cur_loss, cur_acc, _, = sess.run(
                        [loss, accuracy, train_op], feed_dict={is_training: True})
                if print_progress:
                    print(f'Training loss: {cur_loss:8.3f} acc: {cur_acc*100:8.3f}'\
                          f'(batch {batch}/{dataset_size // batch_size})',
                          end='\r')
                losses.append(cur_loss)
                accs.append(cur_acc)
                batch += 1
            except tf.errors.OutOfRangeError:
                average_loss = np.mean(losses)
                std_loss = np.std(losses)
                average_acc = np.mean(accs)
                std_acc = np.std(accs)
                average_losses.append(average_loss)
                std_losses.append(std_loss)
                average_accs.append(average_acc)
                std_accs.append(std_acc)
                print(f'End epoch {i}: mean train loss: {average_loss:4.3f} '\
                      f'+/- {std_loss:4.3f} mean train acc: {average_acc:4.3f} +/- {std_acc:4.3f}')
                for cb in callbacks:
                    try:
                        stop_training |= cb(**locals())
                    except Exception as e:
                        print(f'Got exception in callback {type(cb)}: {e}')
                    losses = []
                break
    sess.close()
    return average_losses, std_losses, average_accs, std_accs
