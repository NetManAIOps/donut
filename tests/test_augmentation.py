# -*- coding: utf-8 -*-
import unittest

import pytest
import numpy as np
from mock import Mock

from donut.augmentation import DataAugmentation, MissingDataInjection


class AugmentationTestCase(unittest.TestCase):

    def test_props(self):
        aug = DataAugmentation(-1., 2.)
        self.assertEqual(aug.mean, -1.)
        self.assertEqual(aug.std, 2.)

        with pytest.raises(ValueError, match='`std` must be positive'):
            _ = DataAugmentation(-1., 0.)
        with pytest.raises(ValueError, match='`std` must be positive'):
            _ = DataAugmentation(-1., -1.)

    def test_augment(self):
        aug = DataAugmentation(-1, 2.)
        aug._augment = Mock(wraps=lambda a, b, c: (a, b, c))
        values, labels, missing = aug.augment(
            values=np.arange(7, dtype=np.float32),
            labels=np.asarray([0, 1, 0, 0, 1, 0, 1], dtype=np.int32),
            missing=np.asarray([0, 0, 0, 1, 0, 0, 1], dtype=np.int32)
        )
        np.testing.assert_equal(values, np.arange(7, dtype=np.float32))
        np.testing.assert_equal(labels, [0, 1, 0, 0, 1, 0, 1])
        np.testing.assert_equal(missing, [0, 0, 0, 1, 0, 0, 1])
        self.assertTrue(aug._augment.called)

        with pytest.raises(ValueError, match='`values` must be a 1-D array'):
            _ = aug.augment(np.zeros([1, 2]), np.zeros([1]), np.zeros([1]))
        with pytest.raises(
                ValueError, match='The shape of `labels` does not agree '
                                  'with the shape of `values`'):
            _ = aug.augment(np.zeros([1]), np.zeros([2]), np.zeros([1]))
        with pytest.raises(
                ValueError, match='The shape of `missing` does not agree '
                                  'with the shape of `values`'):
            _ = aug.augment(np.zeros([1]), np.zeros([1]), np.zeros([2]))


class MissingDataInjectionTestCase(unittest.TestCase):

    def test_props(self):
        mdi = MissingDataInjection(1., 2., .25)
        self.assertEqual(mdi.missing_rate, .25)

    def test_augment(self):
        np.random.seed(1234)
        aug = MissingDataInjection(1., 2., .5)
        values, labels, missing = aug.augment(
            values=np.arange(100, dtype=np.float32),
            labels=np.zeros([100], dtype=np.int32),
            missing=np.zeros([100], dtype=np.int32)
        )
        np.testing.assert_equal(labels, np.zeros([100], dtype=np.int32))
        # big number law: 3-σ, p ≈ 99.7%
        self.assertLess(np.abs(missing.sum() / 100. - .5), .15)
        x = np.arange(100, dtype=np.float32)
        x[missing.astype(np.bool)] = -.5
        np.testing.assert_allclose(values, values)
