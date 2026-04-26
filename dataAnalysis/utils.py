
import numpy as np


def find_idx(array, values, sorting=True):
    """Find index/indices in array closest to given value(s).
    
    Parameters
    ----------
    array : array-like
        Array to search in.
    values : scalar or array-like
        Value(s) to find closest indices for.
    sorting : bool, optional
        If True and values is array-like, sort returned indices.
        Default is True.
    
    Returns
    -------
    int or ndarray
        Index or indices of closest values in array.
    """
    try: iter(values)
    except TypeError:
        idx = (np.abs(array - values)).argmin()
    else:
        idx = np.zeros(len(values))
        for ii in range(len(values)):
            idx[ii] = (np.abs(array - values[ii])).argmin()
        if sorting:
            idx.sort()
    return idx.astype(int)

def find_slice(array, values=None):
    """Create a slice object for array indexing based on closest values.
    
    Parameters
    ----------
    array : array-like
        Array to search in.
    values : array-like of length 2, optional
        Min and max values to find closest indices for. If None, returns
        a slice spanning the entire array. Default is None.
    
    Returns
    -------
    slice
        Slice object with indices corresponding to closest values in array.
        When values is provided, returns slice(idx_min, idx_max+1) where
        idx_max is incremented by 1 to include the upper bound.
    
    Examples
    --------
    >>> arr = np.array([1, 2, 3, 4, 5])
    >>> find_slice(arr, [2.1, 4.1])
    slice(1, 4, None)
    """
    if values is None:
        idx_min = 0
        idx_max = len(array)
    else:
        if values[0] is None:
            idx_min = 0
        else:
            idx_min = (np.abs(array - values[0])).argmin()
        if values[1] is None:
            idx_max = len(array)
        else:
            idx_max = (np.abs(array - values[1])).argmin() + 1
    return slice(idx_min, idx_max)

def filter_array(main_array:np.ndarray, lower_value=None, upper_value=None, other_arrays=None):
    """Filter array based on value range and apply slice to other arrays.
    
    Parameters
    ----------
    main_array : np.ndarray
        Array to search in and filter based on value range.
    lower_value : scalar, optional
        Minimum value to find closest index for. If None, starts from index 0.
        Default is None.
    upper_value : scalar, optional
        Maximum value to find closest index for. If None, goes to end of array.
        Default is None.
    other_arrays : list of np.ndarray, optional
        Other arrays to apply the same slice to. Default is None.
    
    Returns
    -------
    tuple
        Tuple of (filtered_main_array, sliced_other_arrays).
    """
    slice_obj = find_slice(main_array, [lower_value, upper_value])
    main_array = main_array[slice_obj]
    if other_arrays is not None:
        for other_array in other_arrays:
            other_array = other_array[slice_obj]
    return main_array, other_arrays

def slice_array(array, slice_length):
    """Slice a 2D array into chunks along the first dimension.
    
    Parameters
    ----------
    array : ndarray
        2D array to slice.
    slice_length : int
        Length of each slice along the first dimension.
    
    Returns
    -------
    ndarray
        3D array of shape (n_slices, slice_length, n_cols) containing the slices.
    """
    slice_length = int(slice_length)
    n = array.shape
    array_out = np.zeros([int(n[0]/slice_length), slice_length, n[1]])
    for ii in range(int(n[0]/slice_length)):
        array_out[ii] = array[ii:(ii+slice_length)]
    return array_out

def power_watts_to_dBm(power_watts):
    """Convert power from watts to dBm."""
    return 10 * np.log10(power_watts / 1e-3)

def power_dBm_to_watts(power_dBm):
    """Convert power from dBm to watts."""
    return 1e-3 * 10**(power_dBm / 10)



