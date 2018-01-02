import numpy as np
from tfsnippet.utils import DocInherit

__all__ = ['DataAugmentation', 'MissingDataInjection']


@DocInherit
class DataAugmentation(object):
    """
    Base class for data augmentation in training.

    Args:
        mean (float): Mean of the training data.
        std (float): Standard deviation of the training data.
    """

    def __init__(self, mean, std):
        if std <= 0.:
            raise ValueError('`std` must be positive')
        self._mean = mean
        self._std = std

    def augment(self, values, labels, missing):
        """
        Generate augmented data.

        Args:
            values (np.ndarray): 1-D float32 array of shape `(data_length,)`,
                the standardized KPI values.
            labels (np.ndarray): 1-D int32 array of shape `(data_length,)`,
                the anomaly labels for `values`.
            missing (np.ndarray): 1-D int32 array of shape `(data_length,)`,
                the indicator of missing points.

        Returns:
            np.ndarray: The augmented KPI values.
            np.ndarray: The augmented labels.
            np.ndarray: The augmented indicators of missing points.
        """
        if len(values.shape) != 1:
            raise ValueError('`values` must be a 1-D array')
        if labels.shape != values.shape:
            raise ValueError('The shape of `labels` does not agree with the '
                             'shape of `values` ({} vs {})'.
                             format(labels.shape, values.shape))
        if missing.shape != values.shape:
            raise ValueError('The shape of `missing` does not agree with the '
                             'shape of `values` ({} vs {})'.
                             format(missing.shape, values.shape))
        return self._augment(values, labels, missing)

    def _augment(self, values, labels, missing):
        """
        Derived classes should override this to actually implement the
        data augmentation algorithm.
        """
        raise NotImplementedError()

    @property
    def mean(self):
        """Get the mean of the training data."""
        return self._mean

    @property
    def std(self):
        """Get the standard deviation of training data."""
        return self._std


class MissingDataInjection(DataAugmentation):
    """
    Data augmentation by injecting missing points into training data.

    Args:
        mean (float): Mean of the training data.
        std (float): Standard deviation of the training data.
        missing_rate (float): The ratio of missing points to inject.
    """

    def __init__(self, mean, std, missing_rate):
        super(MissingDataInjection, self).__init__(mean, std)
        self._missing_rate = missing_rate

    @property
    def missing_rate(self):
        """Get the ratio of missing points to inject."""
        return self._missing_rate

    def _augment(self, values, labels, missing):
        inject_y = np.random.binomial(1, self.missing_rate, size=values.shape)
        inject_idx = np.where(inject_y.astype(np.bool))[0]
        values = np.copy(values)
        values[inject_idx] = -self.mean / self.std
        missing = np.copy(missing)
        missing[inject_idx] = 1
        return values, labels, missing
