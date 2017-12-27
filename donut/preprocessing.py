import numpy as np

__all__ = ['complete_timestamp', 'standardize_kpi']


def complete_timestamp(timestamp, arrays=None):
    """
    Complete `timestamp` such that the time interval is homogeneous.

    Zeros will be inserted into each array in `arrays`, at missing points.
    Also, an indicator array will be returned to indicate whether each
    point is missing or not.

    Args:
        timestamp (np.ndarray): 1-D int64 array, the timestamp values.
            It can be unsorted.
        arrays (Iterable[np.ndarray]): The 1-D arrays to be filled with zeros
            according to `timestamp`.

    Returns:
        np.ndarray: A 1-D int64 array, the completed timestamp.
        np.ndarray: A 1-D int32 array, indicating whether each point is missing.
        list[np.ndarray]: The arrays, missing points filled with zeros.
            (optional, return only if `arrays` is specified)
    """
    timestamp = np.asarray(timestamp, np.int64)
    if len(timestamp.shape) != 1:
        raise ValueError('`timestamp` must be a 1-D array')

    has_arrays = arrays is not None
    arrays = [np.asarray(array) for array in (arrays or ())]
    for i, array in enumerate(arrays):
        if array.shape != timestamp.shape:
            raise ValueError('The shape of ``arrays[{}]`` does not agree with '
                             'the shape of `timestamp` ({} vs {})'.
                             format(i, array.shape, timestamp.shape))

    # sort the timestamp, and check the intervals
    src_index = np.argsort(timestamp)
    timestamp_sorted = timestamp[src_index]
    intervals = np.unique(np.diff(timestamp_sorted))
    interval = np.min(intervals)
    if interval == 0:
        raise ValueError('Duplicated values in `timestamp`')
    for itv in intervals:
        if itv % interval != 0:
            raise ValueError('Not all intervals in `timestamp` are multiples '
                             'of the minimum interval')

    # prepare for the return arrays
    length = (timestamp_sorted[-1] - timestamp_sorted[0]) // interval + 1
    ret_timestamp = np.arange(timestamp_sorted[0],
                              timestamp_sorted[-1] + interval,
                              interval,
                              dtype=np.int64)
    ret_missing = np.ones([length], dtype=np.int32)
    ret_arrays = [np.zeros([length], dtype=array.dtype) for array in arrays]

    # copy values to the return arrays
    dst_index = np.asarray((timestamp_sorted - timestamp_sorted[0]) // interval,
                           dtype=np.int)
    ret_missing[dst_index] = 0
    for ret_array, array in zip(ret_arrays, arrays):
        ret_array[dst_index] = array[src_index]

    if has_arrays:
        return ret_timestamp, ret_missing, ret_arrays
    else:
        return ret_timestamp, ret_missing


def standardize_kpi(values, mean=None, std=None, excludes=None):
    """
    Standardize a
    Args:
        values (np.ndarray): 1-D `float32` array, the KPI observations.
        mean (float): If not :obj:`None`, will use this `mean` to standardize
            `values`. If :obj:`None`, `mean` will be computed from `values`.
            Note `mean` and `std` must be both :obj:`None` or not :obj:`None`.
            (default :obj:`None`)
        std (float): If not :obj:`None`, will use this `std` to standardize
            `values`. If :obj:`None`, `std` will be computed from `values`.
            Note `mean` and `std` must be both :obj:`None` or not :obj:`None`.
            (default :obj:`None`)
        excludes (np.ndarray): Optional, 1-D `int32` or `bool` array, the
            indicators of whether each point should be excluded for computing
            `mean` and `std`. Ignored if `mean` and `std` are not :obj:`None`.
            (default :obj:`None`)

    Returns:
        np.ndarray: The standardized `values`.
        float: The computed `mean` or the given `mean`.
        float: The computed `std` or the given `std`.
    """
    values = np.asarray(values, dtype=np.float32)
    if len(values.shape) != 1:
        raise ValueError('`values` must be a 1-D array')
    if (mean is None) != (std is None):
        raise ValueError('`mean` and `std` must be both None or not None')
    if excludes is not None:
        excludes = np.asarray(excludes, dtype=np.bool)
        if excludes.shape != values.shape:
            raise ValueError('The shape of `excludes` does not agree with '
                             'the shape of `values` ({} vs {})'.
                             format(excludes.shape, values.shape))

    if mean is None:
        if excludes is not None:
            val = values[np.logical_not(excludes)]
        else:
            val = values
        mean = val.mean()
        std = val.std()

    return (values - mean) / std, mean, std
