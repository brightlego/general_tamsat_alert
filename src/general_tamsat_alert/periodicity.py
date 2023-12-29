import numpy as np
from scipy.fft import fft
from scipy.stats import linregress, mode
import xarray as xr
from typing import List, Hashable, Dict
from itertools import product

AxisLabelType = str


def standardise(data):
    data -= np.mean(data)
    data /= np.std(data)
    return data


def smooth(data, max_offset_proportion=0.75):
    smoothed_data = np.zeros(int(len(data) * max_offset_proportion) - 1)
    for index in range(1, int(len(data) * max_offset_proportion) - 1):
        smoothed_data[index - 1] = np.mean((data[:-index] - data[index:]) ** 2)
    return smoothed_data


def get_axis_periodicity(data):
    data = np.array(data)
    out = smooth(data - np.mean(data))
    result = linregress(np.arange(len(out)), out)
    trend = np.arange(len(out)) * result.slope + result.intercept
    out -= trend
    fourier = np.abs(fft(out))
    freq = np.linspace(0, 1, len(fourier), endpoint=False)
    fourier = fourier[(freq <= 0.5)]
    freq = freq[(freq <= 0.5)]
    return 1 / freq[np.argmax(fourier)]


def get_non_nan_indices(
    ds: xr.Dataset,
    field: Hashable,
    time_label: AxisLabelType = "time",
    step: int = 10,
) -> List[Dict[str, int]]:
    """Gets all the indices in `ds[field]`, sampling every `step` in x
    and y that have no NaN values in all the other (usually only time)
    Axes.

    :param ds: The dataset of the data to check for NaNs
    :param field: The field to check for NaNs in
    :param time_label: The label for the time axis
    :param step: The step to use to sample the grid
    :return: A list of lat, lon indices that contain no NaN values
    """
    out = []
    # Exclude the boundaries as they may contain wierd points, and it
    # is safer to just ignore them
    indices = []
    dimension_names = []
    for dim in ds.dims:
        if dim != time_label:
            if ds.dims[dim] >= 3:
                indices.append(range(1, ds.dims[dim]-1, step))
            else:
                indices.append(range(ds.dims[dim]))
            dimension_names.append(dim)

    for index in product(*indices):
        labeled_index = {i: j for i, j in zip(dimension_names, index)}
        # Turn the data into a numpy array so that nansum is *not* used
        data = np.array(ds.isel(labeled_index)[field])

        # Check if there are any nans as nans propagate
        if not np.isnan(np.sum(data)):
            out.append(labeled_index)

    # If no nans are found, raise an error
    if len(out) == 0:
        raise ValueError(f"All points have NaNs at some time in field {field}")
    return out


def get_periodicity(
    ds: xr.Dataset,
    field: Hashable,
    point: Dict[str, int] = None,
    time_label: AxisLabelType = "time",
    step: int = 10,
) -> int:
    """Gets the periodicity in indices of the data in `ds[field]`.

    Uses some preprocessing + a fourier transform on every `step`th
    lon/lat in the dataset to get the most significant peak in each
    sample and then return the modal sample. Also plots a histogram
    of the sample data.

    If only a singular point is needed to be checked, set `point` to
    a tuple pair of integers.

    :param ds: The dataset to find the periodicity with
    :param field: The field to find the periodicity across
    :param point: (Optional) The point to use if you are impatient
    :param time_label: The label for the time axis
    :param step: The lon/lat step to use in sampling
    :param debug_plots: Whether to plot debug plots
    :return: The periodicity in indices
    """
    if point is None:
        try:
            search_co_ordinates = get_non_nan_indices(ds, field, time_label, step)
        except ValueError:
            m = input(
                "No periodicity found. \nPlease enter a periodicity in "
                "time steps (e.g. 12 for an annual period for monthly data): "
            )
            while True:
                try:
                    m = int(m)
                except ValueError:
                    m = input(f"{m} is not a valid integer. Please input again: ")
                else:
                    return m
        out = []
        for index in search_co_ordinates:
            data = np.array(ds.isel(index)[field])
            out.append(round(get_axis_periodicity(data)))

        m = mode(out)[0][0]
        if m == 1:
            m = input(
                "No periodicity found. \nPlease enter a periodicity in "
                "time steps (e.g. 12 for an annual period for monthly data): "
            )
            while True:
                try:
                    m = int(m)
                except ValueError:
                    m = input(f"{m} is not a valid integer. Please input again: ")
                else:
                    return m
        return m
    else:
        return round(get_axis_periodicity(np.array(ds.isel(point)[field])))
