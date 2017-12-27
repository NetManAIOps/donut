import numpy as np
import pytest
import tensorflow as tf

from donut.reconstruction import (masked_reconstruct,
                                  iterative_masked_reconstruct)


class MaskedReconstructTestCase(tf.test.TestCase):

    def test_static(self):
        with self.test_session():
            x = tf.reshape(tf.range(6), [2, 3])
            reconstruct = lambda v: v + 10

            # test full mask
            mask = tf.ones_like(x, dtype=tf.int32)
            x_r = masked_reconstruct(reconstruct, x, mask)
            np.testing.assert_equal(x_r.eval(), [[10, 11, 12], [13, 14, 15]])

            # test empty mask
            mask = tf.zeros_like(x, dtype=tf.int32)
            x_r = masked_reconstruct(reconstruct, x, mask)
            np.testing.assert_equal(x_r.eval(), [[0, 1, 2], [3, 4, 5]])

            # test partial mask
            mask = tf.constant([[0, 1, 0], [1, 1, 0]], dtype=tf.int32)
            x_r = masked_reconstruct(reconstruct, x, mask)
            np.testing.assert_equal(x_r.eval(), [[0, 11, 2], [13, 14, 5]])

            # test broadcast mask
            mask = tf.constant([0, 1, 1], dtype=tf.int32)
            x_r = masked_reconstruct(reconstruct, x, mask)
            np.testing.assert_equal(x_r.eval(), [[0, 11, 12], [3, 14, 15]])

            mask = tf.constant([[0], [1]], dtype=tf.int32)
            x_r = masked_reconstruct(reconstruct, x, mask)
            np.testing.assert_equal(x_r.eval(), [[0, 1, 2], [13, 14, 15]])

            # test non-broadcastable
            with pytest.raises(
                    ValueError, match='Shape of `mask` cannot broadcast '
                                      'into the shape of `x`'):
                mask = tf.constant([1, 0, 0, 0])
                _ = masked_reconstruct(reconstruct, x, mask)

            with pytest.raises(
                    ValueError, match='Shape of `mask` cannot broadcast '
                                      'into the shape of `x`'):
                mask = tf.constant([[[0]]])
                _ = masked_reconstruct(reconstruct, x, mask)

    def test_dynamic(self):
        with self.test_session():
            x = tf.reshape(tf.range(6), [2, 3])
            mask_ph = tf.placeholder(dtype=tf.int32, shape=None)
            reconstruct = lambda v: v + 10

            # test broadcast mask
            mask = np.asarray([0, 1, 1], dtype=np.int32)
            x_r = masked_reconstruct(reconstruct, x, mask_ph)
            np.testing.assert_equal(
                x_r.eval({mask_ph: mask}), [[0, 11, 12], [3, 14, 15]])

            mask = np.asarray([[0], [1]], dtype=np.int32)
            x_r = masked_reconstruct(reconstruct, x, mask_ph)
            np.testing.assert_equal(
                x_r.eval({mask_ph: mask}), [[0, 1, 2], [13, 14, 15]])

            # test non-broadcastable
            with pytest.raises(
                    Exception, match='Shape of `mask` cannot broadcast into '
                                     'the shape of `x`'):
                mask = np.asarray([[[0]]])
                x_r = masked_reconstruct(reconstruct, x, mask_ph)
                _ = x_r.eval({mask_ph: mask})


class IterativeMaskedReconstructTestCase(tf.test.TestCase):

    def test_iterative_masked_reconstruct(self):
        with self.test_session():
            # test success call
            x = tf.reshape(tf.range(6), [2, 3])
            reconstruct = lambda v: v + 10
            mask = tf.constant([[0, 1, 0], [1, 1, 0]], dtype=tf.int32)
            x_r = iterative_masked_reconstruct(
                reconstruct, x, mask, iter_count=tf.constant(7))
            np.testing.assert_equal(x_r.eval(), [[0, 71, 2], [73, 74, 5]])

            # test error of static call
            with pytest.raises(ValueError, match='iter_count must be positive'):
                _ = iterative_masked_reconstruct(
                    reconstruct, x, mask, iter_count=0)

            # test error of dynamic call
            with pytest.raises(Exception, match='iter_count must be positive'):
                _ = iterative_masked_reconstruct(
                    reconstruct, x, mask, iter_count=tf.constant(0)).eval()

