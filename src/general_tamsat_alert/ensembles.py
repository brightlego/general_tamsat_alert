import xarray as xr
import numpy as np
from typing import List, Tuple, Hashable

from . import weighting_functions as wfs


def get_ensembles(
        da: xr.DataArray,
        period: int,
        ensemble_length: int,
        initiation_index: int,
        look_back: int = 0,
        wf: wfs.WeightingFunctionType = wfs.no_weights,
        time_label: str = "time",
        do_increments: int = 1
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Get an ensemble of the data in the data array.

    Given the most significant period of the data (usually annual),
    the index of the initiation date and the length of the ensemble in
    indices.

    The lookback is the number of indices of observation data required

    TODO: Document inputs

    :param do_increments:
    :param da:
    :param period:
    :param ensemble_length:
    :param initiation_index:
    :param look_back:
    :param wf:
    :param time_label:
    :return:
    """
    time = da[time_label].values[
        initiation_index - look_back : initiation_index + ensemble_length
    ]

    start_times = np.arange(
        initiation_index % period, len(da[time_label]) - ensemble_length, period
    )
    start_times = start_times[(start_times != initiation_index)]
    start_times = np.insert(start_times, 0, initiation_index)

    ensemble_count = len(start_times)
    ensemble_indices = np.arange(0, ensemble_count)

    coords = [da[i].values for i in da.dims if i != time_label] + [ensemble_indices]
    dims = [i for i in da.dims if i != time_label] + ["ensemble"]

    data = np.empty([len(i) for i in [time] + coords])
    ensembles = xr.DataArray(
        data, coords=[time] + coords, dims=[time_label] + dims, name="data",
    )

    weight_data = np.empty([len(i) for i in coords])
    weights = xr.DataArray(weight_data, coords=coords, dims=dims, name="weights",)
    for index, start_time in enumerate(start_times):
        ensembles[look_back:, ..., index] = da.isel(
            {time_label: slice(start_time, start_time + ensemble_length)}
        ).values
        weights[..., index] = wf(start_time)

    if do_increments == 1:
        ensembles[...] -= ensembles[look_back, ...]
        ensembles[...] += da.isel({time_label: initiation_index})

    ensembles[:look_back+1, ...] = da.isel(
        {time_label: slice(initiation_index - look_back, initiation_index+1)}
    ).values[..., np.newaxis]
    weights.values[np.isnan(weights.values)] = 0
    return ensembles, weights


def get_ensemble_indices(
    da: xr.DataArray,
    prediction_date: str,
    start_dates: List[str],
    time_label: Hashable = "time",
):
    indices = np.arange(len(da[time_label]))
    start_indices = indices[
        da[time_label].isin(
            da[time_label].sel({time_label: start_dates}, method="nearest")
        )
    ]
    end_index = indices[
        da[time_label]
        == da[time_label].sel({time_label: prediction_date}, method="nearest")
    ][0]
    ensemble_lengths = (
        end_index - start_indices + 1
    )  # Start and end inclusive so it needs an extra timestep

    if np.sum(da[time_label].isin(np.array(start_dates, dtype=np.datetime64))) < len(
        start_dates
    ):
        print(
            "Warning: not all prediction times are in the dataset. "
            "Using nearest neighbours."
        )
    if np.datetime64(prediction_date) not in da[time_label]:
        print(
            f"Warning: prediction date is not in the dataset. "
            f"Using nearest neighbour {da[end_index]}."
        )
    return start_indices, ensemble_lengths, end_index


def get_mean_data(
    da: xr.DataArray, weights: xr.DataArray = None
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    if weights is not None:
        parray = da[..., 1:].weighted(weights[..., 1:])
    else:
        parray = da[..., 1:]

    mean = parray.mean(dim="ensemble")
    bias = mean - da[..., 0]
    rel_bias = bias / da[..., 0]
    return mean, bias, rel_bias


def get_hindcasts_observed(
    da: xr.DataArray,
    ensemble_lengths: List[int],
    start_indices: List[int],
    period: int,
    wf: wfs.WeightingFunctionIntermediateType = wfs.no_weights_intermediate,
    time_label: str = "time",
):

    mean = []
    stddev = []
    observed = []

    non_time_dims = [da[i] for i in da.dims if i != time_label]
    non_time_labels = [i for i in da.dims if i != time_label]

    for i, start_index in enumerate(start_indices):
        hindcast_indices = np.arange(
            start_index % period, len(da[time_label]) - ensemble_lengths[i], period
        )
        mean.append(
            xr.DataArray(
                np.empty([len(i) for i in non_time_dims] + [len(hindcast_indices)]),
                non_time_dims + [hindcast_indices],
                non_time_labels + ["hindcast"],
            )
        )
        observed.append(
            xr.DataArray(
                np.empty([len(i) for i in non_time_dims] + [len(hindcast_indices)]),
                non_time_dims + [hindcast_indices],
                non_time_labels + ["hindcast"],
            )
        )
        stddev.append(
            xr.DataArray(
                np.empty([len(i) for i in non_time_dims] + [len(hindcast_indices)]),
                non_time_dims + [hindcast_indices],
                non_time_labels + ["hindcast"],
            )
        )
        for j, index in enumerate(hindcast_indices):
            ensembles, weights = get_ensembles(
                da,
                period,
                ensemble_lengths[i],
                index,
                wf=wf(index),
                time_label=time_label,
            )
            data, _, _ = get_mean_data(ensembles[-1, ...], weights)
            mean[-1][..., j] = data.values
            stddev[-1][..., j] = ensembles[-1, :, :, 1:].std(dim="ensemble")
            observed[-1][..., j] = da.isel({time_label: index + ensemble_lengths[i] - 1}).values
    return mean, stddev, observed
