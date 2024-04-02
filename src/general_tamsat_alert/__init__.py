import numpy as np
import xarray as xr

from .periodicity import get_periodicity
from . import weighting_functions as wfs
from .ensembles import get_ensembles
from . import misc


def get_index(da, axis_label, value):
    indices = np.arange(len(da[axis_label]), dtype=np.int64)
    nearest_value = da[axis_label].sel({axis_label: value}, method="nearest")
    return indices[da[axis_label].isin(nearest_value)][0]


def do_forecast(
    datafile,
    field_name,
    init_date,
    poi_start,
    poi_end,
    time_label="time",
    period=12,
    weights_flag=0,
    weighting_data_file=None,
    weighting_strength=1,
    do_increments=1,
):
    ds = xr.open_dataset(datafile)
    da = ds[field_name]

    if weighting_data_file is not None:
        weighting_data = misc.read_noaa_data_file(
            weighting_data_file, da[time_label], time_label
        )
    else:
        weighting_data = np.ones(len(da[time_label]))

    weighting_functions = {
        0: wfs.no_weights_builder(),
        1: wfs.weight_time_builder(period, weighting_strength),
        2: wfs.weight_value_builder(weighting_data, weighting_strength),
    }

    init_index = get_index(da, time_label, init_date)
    poi_start_index = get_index(da, time_label, poi_start)
    poi_end_index = get_index(da, time_label, poi_end)

    print(init_index, poi_start_index, poi_end_index)

    # Calculate inputs for get_ensembles
    ensemble_start = init_index

    ensemble_length = poi_end_index - init_index + 1

    if poi_start_index < init_index:
        look_back = init_index - poi_start_index
    else:
        look_back = 0

    print(ensemble_start, ensemble_length, look_back)

    # check what happens if poi_end_index = init_index
    if poi_end_index < init_index:
        raise ValueError(f"POI end {poi_end} is before the initiation date {init_date}")

    ensemble_out, weights = get_ensembles(
        ds[field_name],
        period=int(period),
        ensemble_length=ensemble_length,
        initiation_index=ensemble_start,
        look_back=look_back,
        wf=weighting_functions[weights_flag](init_index),
        do_increments=do_increments,
    )
    if poi_start_index > init_index:
        poi_offset = poi_start_index - init_index
    else:
        poi_offset = 0

    tmpout_xr = ensemble_out.to_dataset()
    poi_mean = ensemble_out[poi_offset:, ..., 1:].mean(dim=time_label)
    ens_mean = np.average(poi_mean.values, weights=weights.values[..., 1:], axis=-1)
    ens_stddev = np.sqrt(
        np.average(
            (poi_mean.values - ens_mean[..., np.newaxis]) ** 2,
            weights=weights.values[..., 1:],
            axis=-1,
        )
    )

    dims = [i for i in da.dims if i != time_label] + ["ensemble"]

    tmpout_xr["ens_mean"] = (dims[:-1], ens_mean)
    tmpout_xr["ens_std"] = (dims[:-1], ens_stddev)
    tmpout_xr["weights"] = (dims, weights.values)
    return tmpout_xr
