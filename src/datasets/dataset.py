"""Module containing basic dataset functionality."""
import abc
import os
import urllib.request


cache_directory = os.path.join(
    os.path.basename(os.path.abspath(__file__)),
    '.cache'
)


class Dataset(abc.ABC):
    """Abstract base class of a dataset.

    We use this class to define which functions a dataset should have.
    Right now these are:
     - get_tf_dataset()
     - dataset_size

    """

    def __init__(self, cache_dir=cache_directory):
        """Create Dataset instance."""
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _download_cached(self, filename, url):
        """Download file or retrieve from cache."""
        cached_path = self._get_cached_path(filename)
        if not os.path.exists(cached_path):
            urllib.request.urlretrieve(url, cached_path)
        return cached_path

    def _get_cached_path(self, filename):
        return os.path.join(self.cache_dir, filename)

    @abc.abstractmethod
    def get_tf_dataset(self, *args, **kwargs):
        """Get tensorflow dataset for this dataset."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_testing_data(self, *args, **kwargs):
        """Get testing data."""
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def size(self):
        """Get size of dataset."""
        raise NotImplementedError()
