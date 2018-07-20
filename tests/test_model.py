import numpy as np
import pytest
import tensorflow as tf
from mock import Mock
from tfsnippet.modules import VAE
from tfsnippet.utils import (global_reuse, ensure_variables_initialized,
                             get_variables_as_dict)

from donut import Donut


class ModelTestCase(tf.test.TestCase):

    def test_props(self):
        donut = Donut(
            h_for_p_x=Mock(wraps=lambda x: x),
            h_for_q_z=Mock(wraps=lambda x: x),
            x_dims=5,
            z_dims=3,
            std_epsilon=0.125,
        )
        self.assertEqual(donut.x_dims, 5)
        self.assertEqual(donut.z_dims, 3)
        self.assertIsInstance(donut.vae, VAE)

    def test_variable_reuse(self):
        @global_reuse
        def get_donut():
            return Donut(
                h_for_p_x=lambda x: x,
                h_for_q_z=lambda x: x,
                x_dims=5,
                z_dims=3,
            )
        tf.set_random_seed(1234)
        donut1 = get_donut()
        donut2 = get_donut()
        self.assertEqual(donut1.variable_scope.name,
                         donut2.variable_scope.name)

        x = tf.reshape(tf.range(20, dtype=tf.float32), [4, 5])
        _ = donut1.get_score(x)
        _ = donut1.get_score(x)
        _ = donut2.get_score(x)
        _ = donut2.get_score(x)
        self.assertListEqual(
            sorted(get_variables_as_dict()),
            ['get_donut/donut/p_x_given_z/x_mean/bias',
             'get_donut/donut/p_x_given_z/x_mean/kernel',
             'get_donut/donut/p_x_given_z/x_std/bias',
             'get_donut/donut/p_x_given_z/x_std/kernel',
             'get_donut/donut/q_z_given_x/z_mean/bias',
             'get_donut/donut/q_z_given_x/z_mean/kernel',
             'get_donut/donut/q_z_given_x/z_std/bias',
             'get_donut/donut/q_z_given_x/z_std/kernel']
        )

    def test_error_construction(self):
        with pytest.raises(
                ValueError, match='`x_dims` must be a positive integer'):
            _ = Donut(lambda x: x, lambda x: x, x_dims=-1, z_dims=1)
        with pytest.raises(
                ValueError, match='`x_dims` must be a positive integer'):
            _ = Donut(lambda x: x, lambda x: x, x_dims=object(), z_dims=1)
        with pytest.raises(
                ValueError, match='`z_dims` must be a positive integer'):
            _ = Donut(lambda x: x, lambda x: x, x_dims=1, z_dims=0)
        with pytest.raises(
                ValueError, match='`z_dims` must be a positive integer'):
            _ = Donut(lambda x: x, lambda x: x, x_dims=1, z_dims=object())

    def test_training_loss(self):
        class Capture(object):
            def __init__(self, vae):
                self._vae = vae
                self._vae_variational = vae.variational
                vae.variational = self._variational
                self.q_net = None

            def _variational(self, x, z=None, n_z=None):
                assert(z is None)
                if n_z is None:
                    z = tf.reshape(tf.range(12, dtype=tf.float32), [4, 3])
                else:
                    z = tf.reshape(tf.range(n_z * 12, dtype=tf.float32),
                                   [n_z, 4, 3])
                self.q_net = self._vae_variational(x, z=z, n_z=n_z)
                return self.q_net

        # payloads
        x = tf.reshape(tf.range(20, dtype=tf.float32), [4, 5])
        y = tf.constant(
            [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0],
             [0, 0, 0, 1, 0]],
            dtype=tf.int32
        )
        alpha = tf.cast(1 - y, dtype=tf.float32)
        beta = tf.reduce_mean(alpha, axis=-1)

        # create and patch donut model
        tf.set_random_seed(1234)
        donut = Donut(
            h_for_p_x=Mock(wraps=lambda x: x),
            h_for_q_z=Mock(wraps=lambda x: x),
            x_dims=5,
            z_dims=3,
            std_epsilon=0.125,
        )
        capture = Capture(donut.vae)
        _ = donut.get_training_loss(x, y)  # ensure model is built

        # training loss with n_z is None
        with self.test_session() as sess:
            ensure_variables_initialized()

            loss = donut.get_training_loss(x, y)
            np.testing.assert_equal(capture.q_net['z'].eval(),
                                    np.arange(12).reshape([4, 3]))
            p_net = donut.vae.model(z=capture.q_net['z'], x=x)
            sgvb = (
                tf.reduce_sum(p_net['x'].log_prob(group_ndims=0) * alpha,
                              axis=-1) +
                p_net['z'].log_prob() * beta -
                capture.q_net['z'].log_prob()
            )
            self.assertEqual(sgvb.get_shape(), tf.TensorShape([4]))
            loss2 = -tf.reduce_mean(sgvb)
            np.testing.assert_allclose(*sess.run([loss, loss2]))

        # training loss with n_z > 1
        with self.test_session() as sess:
            ensure_variables_initialized()

            loss = donut.get_training_loss(x, y, n_z=7)
            np.testing.assert_equal(capture.q_net['z'].eval(),
                                    np.arange(84).reshape([7, 4, 3]))
            p_net = donut.vae.model(z=capture.q_net['z'], x=x, n_z=7)
            sgvb = (
                tf.reduce_sum(p_net['x'].log_prob(group_ndims=0) * alpha,
                              axis=-1) +
                p_net['z'].log_prob() * beta -
                capture.q_net['z'].log_prob()
            )
            self.assertEqual(sgvb.get_shape(), tf.TensorShape([7, 4]))
            loss2 = -tf.reduce_mean(sgvb)
            np.testing.assert_allclose(*sess.run([loss, loss2]))

    def test_get_score(self):
        class Capture(object):
            def __init__(self, vae):
                self._vae = vae
                self._vae_variational = vae.variational
                vae.variational = self._variational
                self.q_net = None

            def _variational(self, x, z=None, n_z=None):
                self.q_net = self._vae_variational(x, z=z, n_z=n_z)
                return self.q_net

        tf.set_random_seed(1234)
        donut = Donut(
            h_for_p_x=lambda x: x,
            h_for_q_z=lambda x: x,
            x_dims=5,
            z_dims=3,
        )
        capture = Capture(donut.vae)
        donut.vae.reconstruct = Mock(
            wraps=lambda x: x + tf.reduce_sum(x)  # only called by MCMC
        )
        x = tf.reshape(tf.range(20, dtype=tf.float32), [4, 5])
        y = tf.constant(
            [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0],
             [0, 0, 0, 1, 0]],
            dtype=tf.int32
        )
        _ = donut.get_score(x, y)  # ensure variables created

        def r_prob(x, z, n_z=None, x_in=None):
            if x_in is None:
                x_in = x
            q_net = donut.vae.variational(x_in, z=z, n_z=n_z)
            p_net = donut.vae.model(z=q_net['z'], x=x, n_z=n_z)
            p = p_net['x'].log_prob(group_ndims=0)
            if n_z is not None:
                p = tf.reduce_mean(p, axis=0)
            return p

        with self.test_session() as sess:
            ensure_variables_initialized()

            # test y is None
            donut.vae.reconstruct.reset_mock()
            np.testing.assert_allclose(*sess.run([
                donut.get_score(x, y=None, mcmc_iteration=1,
                                last_point_only=False),
                r_prob(x, z=capture.q_net['z'])
            ]))
            self.assertEqual(donut.vae.reconstruct.call_count, 0)

            # test mcmc_iteration is None
            donut.vae.reconstruct.reset_mock()
            np.testing.assert_allclose(*sess.run([
                donut.get_score(x, y=y, mcmc_iteration=None,
                                last_point_only=False),
                r_prob(x, z=capture.q_net['z'])
            ]))
            self.assertEqual(donut.vae.reconstruct.call_count, 0)

            # test mcmc once
            x2 = tf.where(
                tf.cast(y, dtype=tf.bool),
                x,
                x + tf.reduce_sum(x),
            )
            donut.vae.reconstruct.reset_mock()
            np.testing.assert_allclose(*sess.run([
                donut.get_score(x, y=y, mcmc_iteration=1,
                                last_point_only=False),
                r_prob(x, z=capture.q_net['z'], x_in=x2)
            ]))
            self.assertEqual(donut.vae.reconstruct.call_count, 1)

            # test mcmc with n_z > 1
            donut.vae.reconstruct.reset_mock()
            np.testing.assert_allclose(*sess.run([
                donut.get_score(x, y=y, n_z=7, mcmc_iteration=1,
                                last_point_only=False),
                r_prob(x, z=capture.q_net['z'], x_in=x2, n_z=7)
            ]))
            self.assertEqual(capture.q_net['z'].get_shape(),
                             tf.TensorShape([7, 4, 3]))
            self.assertEqual(donut.vae.reconstruct.call_count, 1)
