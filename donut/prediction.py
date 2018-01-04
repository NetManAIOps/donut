import six
import numpy as np
import tensorflow as tf
from tfsnippet.utils import (VarScopeObject, get_default_session_or_error,
                             reopen_variable_scope)

from .model import Donut
from .utils import BatchSlidingWindow

__all__ = ['DonutPredictor']


class DonutPredictor(VarScopeObject):
    """
    Donut predictor.

    Args:
        model (Donut): The :class:`Donut` model instance.
        n_z (int or None): Number of `z` samples to take for each `x`.
            If :obj:`None`, one sample without explicit sampling dimension.
            (default 1024)
        mcmc_iteration: (int or tf.Tensor): Iteration count for MCMC
            missing data imputation. (default 10)
        batch_size (int): Size of each mini-batch for prediction.
            (default 32)
        feed_dict (dict[tf.Tensor, any]): User provided feed dict for
            prediction. (default :obj:`None`)
        last_point_only (bool): Whether to obtain the reconstruction
            probability of only the last point in each window?
            (default :obj:`True`)
        name (str): Optional name of this predictor
            (argument of :class:`tfsnippet.utils.VarScopeObject`).
        scope (str): Optional scope of this predictor
            (argument of :class:`tfsnippet.utils.VarScopeObject`).
    """

    def __init__(self, model, n_z=1024, mcmc_iteration=10, batch_size=32,
                 feed_dict=None, last_point_only=True, name=None, scope=None):
        super(DonutPredictor, self).__init__(name=name, scope=scope)
        self._model = model
        self._n_z = n_z
        self._mcmc_iteration = mcmc_iteration
        self._batch_size = batch_size
        if feed_dict is not None:
            self._feed_dict = dict(six.iteritems(feed_dict))
        else:
            self._feed_dict = {}
        self._last_point_only = last_point_only

        with reopen_variable_scope(self.variable_scope):
            # input placeholders
            self._input_x = tf.placeholder(
                dtype=tf.float32, shape=[None, model.x_dims], name='input_x')
            self._input_y = tf.placeholder(
                dtype=tf.int32, shape=[None, model.x_dims], name='input_y')

            # outputs of interest
            self._score = self._score_without_y = None

    def _get_score(self):
        if self._score is None:
            with reopen_variable_scope(self.variable_scope), \
                    tf.name_scope('score'):
                self._score = self.model.get_score(
                    x=self._input_x,
                    y=self._input_y,
                    n_z=self._n_z,
                    mcmc_iteration=self._mcmc_iteration,
                    last_point_only=self._last_point_only
                )
        return self._score

    def _get_score_without_y(self):
        if self._score_without_y is None:
            with reopen_variable_scope(self.variable_scope), \
                    tf.name_scope('score_without_y'):
                self._score_without_y = self.model.get_score(
                    x=self._input_x,
                    n_z=self._n_z,
                    last_point_only=self._last_point_only
                )
        return self._score_without_y

    @property
    def model(self):
        """
        Get the :class:`Donut` model instance.

        Returns:
            Donut: The :class:`Donut` model instance.
        """
        return self._model

    def get_score(self, values, missing=None):
        """
        Get the `reconstruction probability` of specified KPI observations.

        The larger `reconstruction probability`, the less likely a point
        is anomaly.  You may take the negative of the score, if you want
        something to directly indicate the severity of anomaly.

        Args:
            values (np.ndarray): 1-D float32 array, the KPI observations.
            missing (np.ndarray): 1-D int32 array, the indicator of missing
                points.  If :obj:`None`, the MCMC missing data imputation
                will be disabled. (default :obj:`None`)

        Returns:
            np.ndarray: The `reconstruction probability`,
                1-D array if `last_point_only` is :obj:`True`,
                or 2-D array if `last_point_only` is :obj:`False`.
        """
        with tf.name_scope('DonutPredictor.get_score'):
            sess = get_default_session_or_error()
            collector = []

            # validate the arguments
            values = np.asarray(values, dtype=np.float32)
            if len(values.shape) != 1:
                raise ValueError('`values` must be a 1-D array')

            # run the prediction in mini-batches
            sliding_window = BatchSlidingWindow(
                array_size=len(values),
                window_size=self.model.x_dims,
                batch_size=self._batch_size,
            )
            if missing is not None:
                missing = np.asarray(missing, dtype=np.int32)
                if missing.shape != values.shape:
                    raise ValueError('The shape of `missing` does not agree '
                                     'with the shape of `values` ({} vs {})'.
                                     format(missing.shape, values.shape))
                for b_x, b_y in sliding_window.get_iterator([values, missing]):
                    feed_dict = dict(six.iteritems(self._feed_dict))
                    feed_dict[self._input_x] = b_x
                    feed_dict[self._input_y] = b_y
                    b_r = sess.run(self._get_score(), feed_dict=feed_dict)
                    collector.append(b_r)
            else:
                for b_x, in sliding_window.get_iterator([values]):
                    feed_dict = dict(six.iteritems(self._feed_dict))
                    feed_dict[self._input_x] = b_x
                    b_r = sess.run(self._get_score_without_y(),
                                   feed_dict=feed_dict)
                    collector.append(b_r)

            # merge the results of mini-batches
            result = np.concatenate(collector, axis=0)
            return result
