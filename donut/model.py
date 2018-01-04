import tensorflow as tf
from tensorflow import keras as K
from tfsnippet.distributions import Normal
from tfsnippet.modules import VAE, Sequential, DictMapper, Module
from tfsnippet.stochastic import validate_n_samples
from tfsnippet.utils import (VarScopeObject,
                             reopen_variable_scope,
                             is_integer)
from tfsnippet.variational import VariationalInference

from .reconstruction import iterative_masked_reconstruct

__all__ = ['Donut']


class Donut(VarScopeObject):
    """
    Class for constructing Donut model.

    This class provides :meth:`get_training_objective` for deriving the
    training loss :class:`tf.Tensor`, and :meth:`get_score` for obtaining
    the reconstruction probability :class:`tf.Tensor`.

    Note:
        :class:`Donut` instances will not build the computation graph
        until :meth:`get_training_objective` or :meth:`get_score` is
        called.  This suggests that a :class:`donut.DonutTrainer` or
        a :class:`donut.DonutPredictor` must have been constructed
        before saving or restoring the model parameters.

    Args:
        h_for_p_x (Module or (tf.Tensor) -> tf.Tensor):
            The hidden network for :math:`p(x|z)`.
        h_for_q_z (Module or (tf.Tensor) -> tf.Tensor):
            The hidden network for :math:`q(z|x)`.
        x_dims (int): The number of `x` dimensions.
        z_dims (int): The number of `z` dimensions.
        std_epsilon (float): The minimum value of std for `x` and `z`.
        name (str): Optional name of this module
            (argument of :class:`tfsnippet.utils.VarScopeObject`).
        scope (str): Optional scope of this module
            (argument of :class:`tfsnippet.utils.VarScopeObject`).
    """
    def __init__(self, h_for_p_x, h_for_q_z, x_dims, z_dims, std_epsilon=1e-4,
                 name=None, scope=None):
        if not is_integer(x_dims) or x_dims <= 0:
            raise ValueError('`x_dims` must be a positive integer')
        if not is_integer(z_dims) or z_dims <= 0:
            raise ValueError('`z_dims` must be a positive integer')

        super(Donut, self).__init__(name=name, scope=scope)
        with reopen_variable_scope(self.variable_scope):
            self._vae = VAE(
                p_z=Normal(mean=tf.zeros([z_dims]), std=tf.ones([z_dims])),
                p_x_given_z=Normal,
                q_z_given_x=Normal,
                h_for_p_x=Sequential([
                    h_for_p_x,
                    DictMapper(
                        {
                            'mean': K.layers.Dense(x_dims),
                            'std': lambda x: (
                                std_epsilon + K.layers.Dense(
                                    x_dims,
                                    activation=tf.nn.softplus
                                )(x)
                            )
                        },
                        name='p_x_given_z'
                    )
                ]),
                h_for_q_z=Sequential([
                    h_for_q_z,
                    DictMapper(
                        {
                            'mean': K.layers.Dense(z_dims),
                            'std': lambda z: (
                                std_epsilon + K.layers.Dense(
                                    z_dims,
                                    activation=tf.nn.softplus
                                )(z)
                            )
                        },
                        name='q_z_given_x'
                    )
                ]),
            )
        self._x_dims = x_dims
        self._z_dims = z_dims

    @property
    def x_dims(self):
        """Get the number of `x` dimensions."""
        return self._x_dims

    @property
    def z_dims(self):
        """Get the number of `z` dimensions."""
        return self._z_dims

    @property
    def vae(self):
        """
        Get the VAE object of this :class:`Donut` model.

        Returns:
            VAE: The VAE object of this model.
        """
        return self._vae

    def get_training_objective(self, x, y, n_z=None):
        """
        Get the training objective for `x` and `y`.

        Args:
            x (tf.Tensor): 2-D `float32` :class:`tf.Tensor`, the windows of
                KPI observations in a mini-batch.
            y (tf.Tensor): 2-D `int32` :class:`tf.Tensor`, the windows of
                ``(label | missing)`` in a mini-batch.
            n_z (int or None): Number of `z` samples to take for each `x`.
                (default :obj:`None`, one sample without explicit sampling
                dimension)

        Returns:
            tf.Tensor: The training objective, which can be optimized by
                gradient descent algorithms.
        """
        with tf.name_scope('Donut.training_objective'):
            chain = self.vae.chain(x, n_z=n_z)
            x_log_prob = chain.model['x'].log_prob(group_ndims=0)
            alpha = tf.cast(1 - y, dtype=tf.float32)
            beta = tf.reduce_mean(alpha, axis=-1)
            log_joint = (
                tf.reduce_sum(alpha * x_log_prob, axis=-1) +
                beta * chain.model['z'].log_prob()
            )
            vi = VariationalInference(
                log_joint=log_joint,
                latent_log_probs=chain.vi.latent_log_probs,
                axis=chain.vi.axis
            )
            loss = tf.reduce_mean(vi.training.sgvb())
            return loss

    def get_score(self, x, y=None, n_z=None, mcmc_iteration=None,
                  last_point_only=True):
        """
        Get the reconstruction probability for `x` and `y`.

        The larger `reconstruction probability`, the less likely a point
        is anomaly.  You may take the negative of the score, if you want
        something to directly indicate the severity of anomaly.

        Args:
            x (tf.Tensor): 2-D `float32` :class:`tf.Tensor`, the windows of
                KPI observations in a mini-batch.
            y (tf.Tensor): 2-D `int32` :class:`tf.Tensor`, the windows of
                missing point indicators in a mini-batch.
            n_z (int or None): Number of `z` samples to take for each `x`.
                (default :obj:`None`, one sample without explicit sampling
                dimension)
            mcmc_iteration (int or tf.Tensor): Iteration count for MCMC
                missing data imputation. (default :obj:`None`, no iteration)
            last_point_only (bool): Whether to obtain the reconstruction
                probability of only the last point in each window?
                (default :obj:`True`)

        Returns:
            tf.Tensor: The reconstruction probability, with the shape
                ``(len(x) - self.x_dims + 1,)`` if `last_point_only` is
                :obj:`True`, or ``(len(x) - self.x_dims + 1, self.x_dims)``
                if `last_point_only` is :obj:`False`.  This is because the
                first ``self.x_dims - 1`` points are not the last point of
                any window.
        """
        with tf.name_scope('Donut.get_score'):
            # MCMC missing data imputation
            if y is not None and mcmc_iteration:
                x_r = iterative_masked_reconstruct(
                    reconstruct=self.vae.reconstruct,
                    x=x,
                    mask=y,
                    iter_count=mcmc_iteration,
                    back_prop=False,
                )
            else:
                x_r = x

            # get the reconstruction probability
            q_net = self.vae.variational(x=x_r, n_z=n_z)  # notice: x=x_r
            p_net = self.vae.model(z=q_net['z'], x=x, n_z=n_z)  # notice: x=x
            r_prob = p_net['x'].log_prob(group_ndims=0)
            if n_z is not None:
                n_z = validate_n_samples(n_z, 'n_z')
                assert_shape_op = tf.assert_equal(
                    tf.shape(r_prob),
                    tf.stack([n_z, tf.shape(x)[0], self.x_dims]),
                    message='Unexpected shape of reconstruction prob'
                )
                with tf.control_dependencies([assert_shape_op]):
                    r_prob = tf.reduce_mean(r_prob, axis=0)
            if last_point_only:
                r_prob = r_prob[:, -1]
            return r_prob
