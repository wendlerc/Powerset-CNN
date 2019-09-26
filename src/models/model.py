"""Abstract base classes for models."""
import abc
import json


class SavableModel(abc.ABC):
    """Abstract base class allowing a model to be saved."""

    ignore_from_locals = ['self', '__class__']

    def __init__(self, local, tf_models):
        """Store initialization parameters and child models."""
        self.parameters = self.get_parameters_from_locals(local)
        self.tf_models = tf_models

    @property
    def variables(self):
        """Get variables using in all child models."""
        variables_for_saving = []
        for model in self.tf_models:
            variables_for_saving.extend(model.variables)
        return variables_for_saving

    def get_parameters_from_locals(self, local):
        """Filter locals that are not relevant for saving."""
        return {key: value for key, value in local.items()
                if key not in self.ignore_from_locals}

    def save(self):
        """Return initialization parameters and tf variables for saving."""
        return self.parameters, self.variables

    @classmethod
    def from_file(cls, json_file):
        """Create model instance using json file."""
        with open(json_file, 'r') as f:
            parameters = json.load(f)
        return cls._restore(parameters)

    @classmethod
    def from_string(cls, json_string):
        """Create model instance using json string."""
        parameters = json.loads(json_string)
        return cls._restore(parameters)

    @classmethod
    def from_dict(cls, parameters):
        """Create mdoel instance from dict of parameters."""
        return cls._restore(parameters)

    @classmethod
    @abc.abstractmethod
    def _restore(cls, parameters):
        raise NotImplementedError()


class BaseModel(SavableModel, abc.ABC):
    """Abstract base class for a trainable model."""

    def __init__(self, *args, **kwargs):
        """Initialize model."""
        super().__init__(*args, **kwargs)

    @abc.abstractmethod
    def loss(self, inputs, target):
        """Define loss."""
        raise NotImplementedError()

    @abc.abstractmethod
    def predict(self, inputs):
        raise NotImplementedError()
