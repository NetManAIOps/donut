import unittest

import pytest
import numpy as np

from donut.preprocessing import complete_timestamp, standardize_kpi


class CompleteTimestampTestCase(unittest.TestCase):

    def test_timestamp(self):
        # test pure sort of `timestamp`
        ts, y = complete_timestamp(np.array([0, 1, 6, 5, 3, 4, 2]))
        self.assertEqual(ts.dtype, np.int64)
        self.assertEqual(y.dtype, np.int32)
        np.testing.assert_equal(ts, np.arange(7, dtype=np.int64))
        np.testing.assert_equal(y, np.zeros([7], dtype=np.int32))

        # test fill the missing
        ts, y = complete_timestamp(np.array([0, 1, 3, 6, 5]))
        np.testing.assert_equal(ts, np.arange(7, dtype=np.int64))
        np.testing.assert_equal(y, [0, 0, 1, 0, 1, 0, 0])

        # test shape validation on `timestamp`
        with pytest.raises(
                ValueError, match='`timestamp` must be a 1-D array'):
            _ = complete_timestamp(np.array([[0]]))

        # test duplication detection in `timestamp`
        with pytest.raises(
                ValueError, match='Duplicated values in `timestamp`'):
            _ = complete_timestamp(np.array([0, 0, 1]))

        # test interval homogeneous validation
        with pytest.raises(
                ValueError, match='Not all intervals in `timestamp` are '
                                  'multiples of the minimum interval'):
            _ = complete_timestamp(np.array([0, 5, 2]))

    def test_arrays(self):
        # test pure sort w.r.t. `timestamp`
        ts, y, arrays = complete_timestamp(
            timestamp=np.array([0, 1, 6, 5, 3, 4, 2]),
            arrays=[np.array([7, 6, 1, 2, 4, 3, 5], dtype=np.int32),
                    np.arange(1, 8, dtype=np.float32)]
        )
        self.assertEqual(arrays[0].dtype, np.int32)
        self.assertEqual(arrays[1].dtype, np.float32)
        np.testing.assert_equal(arrays[0], [7, 6, 5, 4, 3, 2, 1])
        np.testing.assert_equal(arrays[1], [1, 2, 7, 5, 6, 4, 3])

        # test fill the missing
        ts, y, arrays = complete_timestamp(
            timestamp=np.array([0, 1, 3, 6, 5]),
            arrays=[np.array([5, 4, 3, 1, 2], dtype=np.int32),
                    np.arange(1, 6, dtype=np.float32)]
        )
        np.testing.assert_equal(arrays[0], [5, 4, 0, 3, 0, 2, 1])
        np.testing.assert_equal(arrays[1], [1, 2, 0, 3, 0, 5, 4])

        # test empty array
        _, _, arrays = complete_timestamp(np.arange(7), [])
        self.assertListEqual(arrays, [])

        # test shape validation
        with pytest.raises(
                ValueError, match=r'The shape of ``arrays\[1\]`` does not '
                                  r'agree with the shape of `timestamp`'):
            _ = complete_timestamp(np.arange(7), [np.arange(7), np.arange(6)])


class StandardizeKPITestCase(unittest.TestCase):

    def test_compute_mean_std(self):
        # test without excludes
        values, mean, std = standardize_kpi(np.arange(10))
        np.testing.assert_allclose(mean, 4.5)
        np.testing.assert_allclose(std, 2.8722813232690143)
        np.testing.assert_allclose(values, (np.arange(10) - mean) / std)

        # test with excludes
        excludes = np.array([0, 1, 0, 0, 1, 1, 0, 0, 0, 1])
        values, mean, std = standardize_kpi(np.arange(10), excludes=excludes)
        np.testing.assert_allclose(mean, 4.333333333333333)
        np.testing.assert_allclose(std, 2.8674417556808756)
        np.testing.assert_allclose(values, (np.arange(10) - mean) / std)

    def test_apply_mean_std(self):
        values, mean, std = standardize_kpi(np.arange(10), mean=5., std=2.)
        np.testing.assert_allclose(mean, 5.)
        np.testing.assert_allclose(std, 2.)
        np.testing.assert_allclose(values, (np.arange(10) - mean) / std)

    def test_errors(self):
        with pytest.raises(
                ValueError, match='`values` must be a 1-D array'):
            _ = standardize_kpi(np.array([[0]]))
        with pytest.raises(
                ValueError, match='`mean` and `std` must be both None or not '
                                  'None'):
            _ = standardize_kpi(np.arange(10), mean=1.)
        with pytest.raises(
                ValueError, match='`mean` and `std` must be both None or not '
                                  'None'):
            _ = standardize_kpi(np.arange(10), std=2.)
        with pytest.raises(
                ValueError, match='The shape of `excludes` does not agree with '
                                  'the shape of `values`'):
            _ = standardize_kpi(
                np.arange(10), excludes=np.zeros([11], dtype=np.int32))
