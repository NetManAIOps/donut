import pytest
import numpy as np
import tensorflow as tf
from tfsnippet.utils import ensure_variables_initialized

from donut import Donut, DonutPredictor


class DonutPredictorTestCase(tf.test.TestCase):
    """
    Tests for :class:`DonutPredictor`.

    It is very difficult to check the outputs of a :class:`DonutPredictor`,
    since it involves random samples of `z` internally.  Fortunately, we
    have thoroughly tested :class:`Donut`, thus we just check whether or not
    the :class:`DonutPredictor` can produce outputs successfully, without
    testing the values of its outputs.
    """

    def test_prediction(self):
        np.random.seed(1234)
        tf.set_random_seed(1234)

        # test last_point_only == True
        donut = Donut(h_for_p_x=lambda x: x, h_for_q_z=lambda x: x, x_dims=5,
                      z_dims=3)
        _ = donut.get_score(
            tf.zeros([4, 5], dtype=tf.float32),
            tf.zeros([4, 5], dtype=tf.int32))  # ensure variables created
        pred = DonutPredictor(donut, n_z=2, batch_size=4)
        self.assertIs(pred.model, donut)

        with self.test_session():
            ensure_variables_initialized()

            # test without missing
            res = pred.get_score(values=np.arange(5, dtype=np.float32))
            self.assertEqual(res.shape, (1,))

            res = pred.get_score(values=np.arange(8, dtype=np.float32))
            self.assertEqual(res.shape, (4,))

            res = pred.get_score(values=np.arange(10, dtype=np.float32))
            self.assertEqual(res.shape, (6,))

            # test with missing
            res = pred.get_score(values=np.arange(10, dtype=np.float32),
                                 missing=np.random.binomial(1, .5, size=10))
            self.assertEqual(res.shape, (6,))

        # test errors
        with self.test_session():
            with pytest.raises(
                    ValueError, match='`values` must be a 1-D array'):
                _ = pred.get_score(np.arange(10, dtype=np.float32).
                                   reshape([-1, 1]))
            with pytest.raises(
                    ValueError, match='The shape of `missing` does not agree '
                                      'with the shape of `values`'):
                _ = pred.get_score(np.arange(10, dtype=np.float32),
                                   np.arange(9, dtype=np.int32))

        # test last_point_only == False
        pred = DonutPredictor(donut, n_z=2, batch_size=4, last_point_only=False)

        with self.test_session():
            ensure_variables_initialized()

            # test without missing
            res = pred.get_score(values=np.arange(10, dtype=np.float32))
            self.assertEqual(res.shape, (6, 5))

            # test with missing
            res = pred.get_score(values=np.arange(10, dtype=np.float32),
                                 missing=np.random.binomial(1, .5, size=10))
            self.assertEqual(res.shape, (6, 5))

        # test set feed_dict
        is_training = tf.placeholder(tf.bool, shape=())
        pred = DonutPredictor(donut, n_z=2, batch_size=4,
                              feed_dict={is_training: False})

        with self.test_session():
            ensure_variables_initialized()
            _ = pred.get_score(values=np.arange(10, dtype=np.float32))
