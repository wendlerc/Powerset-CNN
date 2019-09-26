"""Module containing sacred functions for handling ML models."""
import inspect

from sacred import Ingredient


from src import models

ingredient = Ingredient('model')


@ingredient.config
def cfg():
    """Model configuration."""
    name = ''
    signal_model = None
    use_sum = False
    n_layers = 2
    arch = None
    pooling = None
    parameters = {
    }


@ingredient.named_config
def MLP():
    name = 'MLPModel'
    parameters = {
        'arch': [4096, 4096]
    }


@ingredient.named_config
def KIPF():
    name = 'PowersetConvNet'
    signal_model = None
    n_layers = 2
    pooling = None
    use_sum = False
    if pooling is None:
        arch = [('Kipf', 32)]*n_layers
    else:
        arch = [('Kipf', 32), pooling]*n_layers
    parameters = {'conv_arch': arch, 'mlp_arch': [512],
                  'model': signal_model, 'use_sum': use_sum}


@ingredient.named_config
def ADJ():
    name = 'PowersetConvNet'
    signal_model = None
    n_layers = 2
    pooling = None
    use_sum = False
    if pooling is None:
        arch = [('Adj', 32)]*n_layers
    else:
        arch = [('Adj', 32), pooling]*n_layers
    parameters = {'conv_arch': arch, 'mlp_arch': [512],
                  'model': signal_model, 'use_sum': use_sum}


@ingredient.named_config
def PCN():
    name = 'PowersetConvNet'
    signal_model = 3
    n_layers = 2
    pooling = None
    use_sum = False
    if pooling is None:
        arch = [('E', 32)]*n_layers
    else:
        arch = [('E', 32), pooling]*n_layers
    parameters = {'conv_arch': arch, 'mlp_arch': [512],
                  'model': signal_model, 'use_sum': use_sum}


@ingredient.capture
def get_instance(n_classes, n_groundset, name, parameters, _log):
    """Get an instance of a model according to parameters in the configuration.

    Also, check if the provided parameters fit to the signature of the model
    class and log default values if not defined via the configuration.

    """
    general_parameters = locals()
    # Get the mode class
    model_cls = getattr(models, name)

    # Inspect if the constructor specification fits with additional_parameters
    signature = inspect.signature(model_cls)
    available_parameters = signature.parameters
    if 'n_groundset' in available_parameters.keys():
        parameters['n_groundset'] = n_groundset

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

    return model_cls(n_classes=n_classes, **parameters)
