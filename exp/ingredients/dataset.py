"""Module containing sacred functions for handling ML models."""
import inspect

from sacred import Ingredient

import functools

from src import datasets
from src.datasets.setfunction_dataset import artificial_hard, artificial_junta, \
    artificial_xmodular, congress10, coauth10, domain


ingredient = Ingredient('dataset')


@ingredient.config
def cfg():
    """Dataset configuration."""
    name = ''
    fname = None
    parameters = {
    }


@ingredient.named_config
def AH():
    """Aritificial Experiment"""
    name = 'SetfunctionDataset'
    parameters = {
        'N': 10,
        'data_generator': artificial_hard
    }


@ingredient.named_config
def AJ():
    """Aritificial Experiment"""
    name = 'SetfunctionDataset'
    parameters = {
        'N': 10,
        'data_generator': artificial_junta
    }


@ingredient.named_config
def AXM():
    """Aritificial Experiment"""
    name = 'SetfunctionDataset'
    parameters = {
        'N': 10,
        'data_generator': artificial_xmodular
    }



@ingredient.named_config
def CON10():
    name = 'SetfunctionDataset'
    parameters = {
        'N': 10,
        'data_generator': congress10
    }


@ingredient.named_config
def COAUTH10():
    name = 'SetfunctionDataset'
    parameters = {
        'N': 10,
        'data_generator': coauth10
    }


@ingredient.named_config
def DOM4():
    name = 'SetfunctionDataset'
    parameters = {
        'N': 10,
        'data_generator':  functools.partial(domain, n_classes=4)
    }


@ingredient.named_config
def DOM6():
    name = 'SetfunctionDataset'
    parameters = {
        'N': 10,
        'data_generator':  functools.partial(domain, n_classes=6)
    }


@ingredient.capture
def get_instance(name, parameters, _log):
    """Get an instance of a model according to parameters in the configuration.

    Also, check if the provided parameters fit to the signature of the model
    class and log default values if not defined via the configuration.

    """
    general_parameters = locals()
    # Get the mode class
    model_cls = getattr(datasets, name)

    # Inspect if the constructor specification fits with additional_parameters
    signature = inspect.signature(model_cls)
    available_parameters = signature.parameters
    for key in parameters.keys():
        if key not in available_parameters.keys():
            # If a parameter is defined which does not fit to the constructor
            # raise an error
            raise ValueError(
                f'{key} is not available in {name}\'s Constructor'
            )

    # Now check if optional parameters of the constructor are not defined
    optional_parameters = list(available_parameters.keys())[4:]
    for parameter_name in optional_parameters:
        if parameter_name not in parameters.keys():
            # If an optional parameter is not defined warn and run with default
            default = available_parameters[parameter_name].default
            _log.warning(f'Optional parameter {parameter_name} not explicitly '
                         f'defined, will run with {parameter_name}={default}')

    return model_cls(**parameters)
