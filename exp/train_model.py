"""Module to train a model with a dataset configuration."""
import os

import numpy as np
import matplotlib.pyplot as plt

from sacred import Experiment
from tempfile import NamedTemporaryFile

from src.training import train_model_with_dataset

from .callbacks import LogTrainingLossCallback, StopOnNaNCallback, ModelSaver, TestCallback
from .ingredients import model
from .ingredients import dataset

experiment = Experiment(
    'training',
    ingredients=[model.ingredient, dataset.ingredient]
)


@experiment.config
def cfg():
    epochs = 101
    batch_size = 128
    lr = 1e-3
    decay_factor = 0.95
    early_stopping = 10
    print_progress = True
    save_tf_log = False
    min_measurements = None
    grad_summary = False
    fix_randomseed = False


@experiment.automain
def train(epochs, batch_size, lr, decay_factor, early_stopping,
          print_progress, save_tf_log, grad_summary, fix_randomseed, _run, _log):
    """Sacred wrapped function to run training of model."""
    if fix_randomseed:
        np.random.seed(42)
    if save_tf_log:
        try:
            rundir = _run.observers[0].dir
            tf_logdir = os.path.join(rundir, 'log')
        except IndexError:
            tf_logdir = None
    else:
        tf_logdir = None

    # Get data
    data = dataset.get_instance()
    # Get model
    mymodel = model.get_instance(data.n_classes, data.n_groundset)

    # Setup callbacks to track progress
    train_loss_cb = LogTrainingLossCallback(_run)
    metrics = {'accuracy': lambda target, pred: np.sum(np.argmax(target, axis=1) == np.argmax(pred, axis=1))/len(target)}#,
    test_cb = TestCallback(data, _run, metrics=metrics)

    def SaveFigure(fig, epoch):
        with NamedTemporaryFile(suffix='.pdf') as f:
            fig.savefig(f.name)
            _run.add_artifact(f.name, 'epoch_{}.pdf'.format(epoch))
            plt.close(fig)

    callbacks = [train_loss_cb, StopOnNaNCallback(), ModelSaver(_run), test_cb]

    train_losses, std_losses, train_accs, std_accs = train_model_with_dataset(
        mymodel, data, epochs, batch_size, lr, decay_factor, callbacks=callbacks,
        print_progress=print_progress, tf_logdir=tf_logdir, do_gradient_summary=grad_summary
    )

    with NamedTemporaryFile(suffix='.pdf') as f:
        plt.errorbar(
            range(1, len(train_losses)), train_losses[1:], yerr=std_losses[1:],
            label='training loss'
        )
        plt.savefig(f.name)
        plt.close()
        _run.add_artifact(f.name, 'losses.pdf')

    with NamedTemporaryFile(suffix='.pdf') as f:
        plt.errorbar(
            range(1, len(train_accs)), train_accs[1:], yerr=std_accs[1:],
            label='training acc'
        )
        plt.savefig(f.name)
        plt.close()
        _run.add_artifact(f.name, 'accs.pdf')

    with NamedTemporaryFile(suffix='.pdf') as f:
        for k in metrics.keys():
            plt.plot(test_cb.log[k], label='test '+k)
        plt.savefig(f.name)
        plt.close()
        _run.add_artifact(f.name, 'test_metrics.pdf')

    result = {
        'epochs_trained': len(train_losses),
        'end_train_loss': train_losses[-1],
        'best_train_loss': np.nanmin(train_losses),
        'end_test_acc': test_cb.log['accuracy'][-1],
        'best_test_acc': np.nanmax(test_cb.log['accuracy']),
        'end_train_acc': train_accs[-1]
    }
    return result
