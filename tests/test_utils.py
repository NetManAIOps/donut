import unittest

import pytest
import numpy as np

from donut.utils import minibatch_slices_iterator, BatchSlidingWindow


class MiniBatchSlicesIteratorTestCase(unittest.TestCase):

    def test_minibatch_slices_iterator(self):
        self.assertEqual(
            list(minibatch_slices_iterator(0, 10, False)),
            []
        )
        self.assertEqual(
            list(minibatch_slices_iterator(9, 10, False)),
            [slice(0, 9, 1)]
        )
        self.assertEqual(
            list(minibatch_slices_iterator(10, 10, False)),
            [slice(0, 10, 1)]
        )
        self.assertEqual(
            list(minibatch_slices_iterator(10, 9, False)),
            [slice(0, 9, 1), slice(9, 10, 1)]
        )
        self.assertEqual(
            list(minibatch_slices_iterator(10, 9, True)),
            [slice(0, 9, 1)]
        )


class BatchSlidingWindowTestCase(unittest.TestCase):

    def _check_output(self, iterator, answer):
        result = list(iterator)
        self.assertEqual(len(result), len(answer))
        for res, ans in zip(result, answer):
            self.assertEqual(len(res), len(ans))
            for r, a in zip(res, ans):
                np.testing.assert_equal(r, np.asarray(a))

    def test_construction(self):
        with pytest.raises(
                ValueError, match='`window_size` must be at least 1'):
            _ = BatchSlidingWindow(10, 0, 3)
        with pytest.raises(
                ValueError, match='`array_size` must be at least as large as '
                                  '`window_size`'):
            _ = BatchSlidingWindow(4, 5, 3)
        with pytest.raises(
                ValueError, match=r'The shape of `excludes` is expected to '
                                  r'be \(10,\), but got \(9,\)'):
            _ = BatchSlidingWindow(10, 5, 3, excludes=np.arange(9))

    def test_validate_arrays(self):
        with pytest.raises(ValueError, match='`arrays` must not be empty'):
            _ = next(BatchSlidingWindow(10, 5, 3).get_iterator([]))
        with pytest.raises(
                ValueError, match=r'The shape of `arrays\[1\]` is expected '
                                  r'to be \(10,\), but got \(10, 1\)'):
            _ = next(BatchSlidingWindow(10, 5, 3).get_iterator(
                [np.arange(10), np.arange(10).reshape([-1, 1])]))

    def test_basic(self):
        self._check_output(
            BatchSlidingWindow(5, 5, 3).get_iterator(
                [np.arange(5), np.arange(-1, -6, -1)]),
            [
                ([[0, 1, 2, 3, 4]], [[-1, -2, -3, -4, -5]],)
            ]
        )
        self._check_output(
            BatchSlidingWindow(7, 5, 3).get_iterator(
                [np.arange(7), np.arange(-1, -8, -1)]),
            [
                ([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5], [2, 3, 4, 5, 6]],
                 [[-1, -2, -3, -4, -5], [-2, -3, -4, -5, -6],
                  [-3, -4, -5, -6, -7]])
            ]
        )
        self._check_output(
            BatchSlidingWindow(9, 5, 3).get_iterator(
                [np.arange(9), np.arange(-1, -10, -1)]),
            [
                ([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5], [2, 3, 4, 5, 6]],
                 [[-1, -2, -3, -4, -5], [-2, -3, -4, -5, -6],
                  [-3, -4, -5, -6, -7]]),
                ([[3, 4, 5, 6, 7], [4, 5, 6, 7, 8]],
                 [[-4, -5, -6, -7, -8], [-5, -6, -7, -8, -9]])
            ]
        )
        self._check_output(
            BatchSlidingWindow(10, 5, 3).get_iterator(
                [np.arange(10), np.arange(-1, -11, -1)]),
            [
                ([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5], [2, 3, 4, 5, 6]],
                 [[-1, -2, -3, -4, -5], [-2, -3, -4, -5, -6],
                  [-3, -4, -5, -6, -7]]),
                ([[3, 4, 5, 6, 7], [4, 5, 6, 7, 8], [5, 6, 7, 8, 9]],
                 [[-4, -5, -6, -7, -8], [-5, -6, -7, -8, -9],
                  [-6, -7, -8, -9, -10]])
            ]
        )

    def test_ignore_incomplete(self):
        self._check_output(
            BatchSlidingWindow(5, 5, 3,
                               ignore_incomplete_batch=True).get_iterator(
                [np.arange(5), np.arange(-1, -6, -1)]
            ),
            []
        )
        self._check_output(
            BatchSlidingWindow(7, 5, 3,
                               ignore_incomplete_batch=True).get_iterator(
                [np.arange(7), np.arange(-1, -8, -1)]
            ),
            [
                ([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5], [2, 3, 4, 5, 6]],
                 [[-1, -2, -3, -4, -5], [-2, -3, -4, -5, -6],
                  [-3, -4, -5, -6, -7]])
            ]
        )
        self._check_output(
            BatchSlidingWindow(9, 5, 3,
                               ignore_incomplete_batch=True).get_iterator(
                [np.arange(9), np.arange(-1, -10, -1)]
            ),
            [
                ([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5], [2, 3, 4, 5, 6]],
                 [[-1, -2, -3, -4, -5], [-2, -3, -4, -5, -6],
                  [-3, -4, -5, -6, -7]])
            ]
        )
        self._check_output(
            BatchSlidingWindow(10, 5, 3,
                               ignore_incomplete_batch=True).get_iterator(
                [np.arange(10), np.arange(-1, -11, -1)]
            ),
            [
                ([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5], [2, 3, 4, 5, 6]],
                 [[-1, -2, -3, -4, -5], [-2, -3, -4, -5, -6],
                  [-3, -4, -5, -6, -7]]),
                ([[3, 4, 5, 6, 7], [4, 5, 6, 7, 8], [5, 6, 7, 8, 9]],
                 [[-4, -5, -6, -7, -8], [-5, -6, -7, -8, -9],
                  [-6, -7, -8, -9, -10]])
            ]
        )

    def test_shuffle(self):
        a_collector = []
        b_collector = []
        for a, b in BatchSlidingWindow(10, 5, 3, shuffle=True). \
                get_iterator([np.arange(10), np.arange(-1, -11, -1)]):
            for a_row in a:
                a_collector.append(a_row)
            for b_row in b:
                b_collector.append(b_row)
        a_collector = np.asarray(a_collector)
        b_collector = np.asarray(b_collector)
        idx = np.argsort(a_collector[:, 0])
        a = a_collector[idx, :]
        b = b_collector[idx, :]
        np.testing.assert_equal(
            a, [[0, 1, 2, 3, 4], [1, 2, 3, 4, 5], [2, 3, 4, 5, 6],
                [3, 4, 5, 6, 7], [4, 5, 6, 7, 8], [5, 6, 7, 8, 9]])
        np.testing.assert_equal(
            b, [[-1, -2, -3, -4, -5], [-2, -3, -4, -5, -6],
                [-3, -4, -5, -6, -7], [-4, -5, -6, -7, -8],
                [-5, -6, -7, -8, -9], [-6, -7, -8, -9, -10]])

    def test_excludes(self):
        excludes = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 1], dtype=np.bool)
        self._check_output(
            BatchSlidingWindow(10, 3, 2, excludes=excludes).get_iterator(
                [np.arange(10), np.arange(-1, -11, -1)]),
            [
                ([[1, 2, 3], [5, 6, 7]],
                 [[-2, -3, -4], [-6, -7, -8]]),
                ([[6, 7, 8]],
                 [[-7, -8, -9]])
            ]
        )
