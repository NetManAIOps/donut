import numpy as np

__all__ = ['minibatch_slices_iterator', 'BatchSlidingWindow']


def minibatch_slices_iterator(length, batch_size,
                              ignore_incomplete_batch=False):
    """
    Iterate through all the mini-batch slices.

    Args:
        length (int): Total length of data in an epoch.
        batch_size (int): Size of each mini-batch.
        ignore_incomplete_batch (bool): If :obj:`True`, discard the final
            batch if it contains less than `batch_size` number of items.
            (default :obj:`False`)

    Yields
        slice: Slices of each mini-batch.  The last mini-batch may contain
               less indices than `batch_size`.
    """
    start = 0
    stop1 = (length // batch_size) * batch_size
    while start < stop1:
        yield slice(start, start + batch_size, 1)
        start += batch_size
    if not ignore_incomplete_batch and start < length:
        yield slice(start, length, 1)


class BatchSlidingWindow(object):
    """
    Class for obtaining mini-batch iterators of sliding windows.

    Each mini-batch will have `batch_size` windows.  If the final batch
    contains less than `batch_size` windows, it will be discarded if
    `ignore_incomplete_batch` is :obj:`True`.

    Args:
        array_size (int): Size of the arrays to be iterated.
        window_size (int): The size of the windows.
        batch_size (int): Size of each mini-batch.
        excludes (np.ndarray): 1-D `bool` array, indicators of whether
            or not to totally exclude a point.  If a point is excluded,
            any window which contains that point is excluded.
            (default :obj:`None`, no point is totally excluded)
        shuffle (bool): If :obj:`True`, the windows will be iterated in
            shuffled order. (default :obj:`False`)
        ignore_incomplete_batch (bool): If :obj:`True`, discard the final
            batch if it contains less than `batch_size` number of windows.
            (default :obj:`False`)
    """

    def __init__(self, array_size, window_size, batch_size, excludes=None,
                 shuffle=False, ignore_incomplete_batch=False):
        # check the parameters
        if window_size < 1:
            raise ValueError('`window_size` must be at least 1')
        if array_size < window_size:
            raise ValueError('`array_size` must be at least as large as '
                             '`window_size`')
        if excludes is not None:
            excludes = np.asarray(excludes, dtype=np.bool)
            expected_shape = (array_size,)
            if excludes.shape != expected_shape:
                raise ValueError('The shape of `excludes` is expected to be '
                                 '{}, but got {}'.
                                 format(expected_shape, excludes.shape))

        # compute which points are not excluded
        if excludes is not None:
            mask = np.logical_not(excludes)
        else:
            mask = np.ones([array_size], dtype=np.bool)
        mask[: window_size - 1] = False
        where_excludes = np.where(excludes)[0]
        for k in range(1, window_size):
            also_excludes = where_excludes + k
            also_excludes = also_excludes[also_excludes < array_size]
            mask[also_excludes] = False

        # generate the indices of window endings
        indices = np.arange(array_size)[mask]
        self._indices = indices.reshape([-1, 1])

        # the offset array to generate the windows
        self._offsets = np.arange(-window_size + 1, 1)

        # memorize arguments
        self._array_size = array_size
        self._window_size = window_size
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._ignore_incomplete_batch = ignore_incomplete_batch

    def get_iterator(self, arrays):
        """
        Iterate through the sliding windows of each array in `arrays`.

        This method is not re-entrant, i.e., calling :meth:`get_iterator`
        would invalidate any previous obtained iterator.

        Args:
            arrays (Iterable[np.ndarray]): 1-D arrays to be iterated.

        Yields:
            tuple[np.ndarray]: The windows of arrays of each mini-batch.
        """
        # check the parameters
        arrays = tuple(np.asarray(a) for a in arrays)
        if not arrays:
            raise ValueError('`arrays` must not be empty')
        expected_shape = (self._array_size,)
        for i, a in enumerate(arrays):
            if a.shape != expected_shape:
                raise ValueError('The shape of `arrays[{}]` is expected to '
                                 'be {}, but got {}'.
                                 format(i, expected_shape, a.shape))

        # shuffle if required
        if self._shuffle:
            np.random.shuffle(self._indices)

        # iterate through the mini-batches
        for s in minibatch_slices_iterator(
                length=len(self._indices),
                batch_size=self._batch_size,
                ignore_incomplete_batch=self._ignore_incomplete_batch):
            idx = self._indices[s] + self._offsets
            yield tuple(a[idx] for a in arrays)
