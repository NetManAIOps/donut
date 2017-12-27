import numpy as np
import pytest
import tensorflow as tf
from tfsnippet.utils import TemporaryDirectory

from donut import Donut, DonutTrainer


class DonutTrainerTestCase(tf.test.TestCase):
    """
    Tests for :class:`DonutTrainer`.

    It is very difficult to test a :class:`DonutTrainer`.
    We just check whether or not the :class:`DonutTrainer` can run
    successfully.
    """

    def _payload(self):
        def mklabel(p=.01, n=1000):
            ret = np.random.binomial(1, p, size=[n]).astype(np.int32)
            self.assertGreater(ret.sum(), 0)
            return ret

        np.random.seed(1234)
        labels = mklabel()
        missing = mklabel()
        excludes = mklabel()
        values = np.random.normal(
            0., 1., size=labels.shape).astype(np.float32)

        return values, labels, missing, excludes

    def test_fit(self):
        values, labels, missing, excludes = self._payload()

        with TemporaryDirectory() as tmpdir:
            tf.set_random_seed(1234)
            donut = Donut(
                h_for_p_x=lambda x: x, h_for_q_z=lambda x: x, x_dims=5,
                z_dims=3
            )
            trainer = DonutTrainer(
                donut, max_epoch=3, batch_size=7, valid_step_freq=50,
                lr_anneal_epochs=2
            )

            with self.test_session():
                trainer.fit(
                    values=values, labels=labels, missing=missing, mean=1.,
                    std=2., excludes=excludes, summary_dir=tmpdir
                )

    def test_construction_args(self):
        values, labels, missing, excludes = self._payload()

        tf.set_random_seed(1234)
        donut = Donut(
            h_for_p_x=lambda x: x, h_for_q_z=lambda x: x, x_dims=5,
            z_dims=3
        )

        # test feed_dict
        is_training = tf.placeholder(tf.bool, ())
        trainer = DonutTrainer(
            donut, max_epoch=1, feed_dict={is_training: True}
        )
        with self.test_session():
            trainer.fit(values=values, labels=labels, missing=missing,
                        mean=1., std=2., excludes=excludes)

        # test valid_feed_dict
        trainer = DonutTrainer(
            donut, max_epoch=1, valid_feed_dict={is_training: True}
        )
        with self.test_session():
            trainer.fit(values=values, labels=labels, missing=missing,
                        mean=1., std=2., excludes=excludes)

        # test max_epoch is None and max_step is None
        with pytest.raises(
                ValueError, match='At least one of `max_epoch` and `max_step` '
                                  'should be specified'):
            _ = DonutTrainer(donut, max_epoch=None, max_step=None)

        # test optimizer and optimizer_params
        trainer = DonutTrainer(
            donut, max_epoch=1, optimizer=tf.train.MomentumOptimizer,
            optimizer_params={'momentum': 0.01}
        )
        with self.test_session():
            trainer.fit(values=values, labels=labels, missing=missing,
                        mean=1., std=2., excludes=excludes)

    def test_fit_args(self):
        values, labels, missing, excludes = self._payload()

        tf.set_random_seed(1234)
        donut = Donut(
            h_for_p_x=lambda x: x, h_for_q_z=lambda x: x, x_dims=5,
            z_dims=3
        )
        trainer = DonutTrainer(donut, max_epoch=1)

        with self.test_session():
            # test no exclude
            trainer.fit(values=values, labels=labels, missing=missing,
                        mean=1., std=2.)

            # test shape error
            with pytest.raises(
                    ValueError, match='`values` must be a 1-D array'):
                trainer.fit(values=np.array([[1.]]), labels=labels,
                            missing=missing, mean=1., std=2.)
            with pytest.raises(
                    ValueError, match='The shape of `labels` does not agree '
                                      'with the shape of `values`'):
                trainer.fit(values=values, labels=labels[:-1],
                            missing=missing, mean=1., std=2.)
            with pytest.raises(
                    ValueError, match='The shape of `missing` does not agree '
                                      'with the shape of `values`'):
                trainer.fit(values=values, labels=labels,
                            missing=missing[:-1], mean=1., std=2.)
