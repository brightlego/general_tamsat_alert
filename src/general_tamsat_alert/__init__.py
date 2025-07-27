import numpy as np
import xarray as xr
import datetime

from .periodicity import get_periodicity
from . import weighting_functions as wfs
from .ensembles import get_ensembles
from . import misc


def get_index(da, axis_label, value):
    indices = np.arange(len(da[axis_label]), dtype=np.int64)
    nearest_value = da[axis_label].sel({axis_label: value}, method="nearest")

    # Assume the values on the time axis label are increasing
    max_value = da[axis_label].values[-1]
    min_value = da[axis_label].values[0]
    if value > max_value:
        delta = value - max_value
        max_index = get_index(da, axis_label, max_value)
        if max_value - delta >= min_value:
            opposite = get_index(da, axis_label, max_value - delta)
            return (max_index - opposite) + max_index
        else:
            try:
                max_delta = max_value - da[axis_label].values.min()
                max_index_delta = da[axis_label].values.shape[0]
                return round(max_index_delta * delta / max_delta + max_index)
            except TypeError:
                raise TypeError("Unable to extrapolate time index from non-numeric data")

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
    suppress_datetime_conversion=False
):

    '''
    Function that ingests time series data and produces ensemble forecasts using the TAMSAT-ALERT method
    
    Input parameters
    ----------------
    :param datafile: netcdf file containing the time series data on which to base the forecasts. The datafile must
                     include a time axis, but the format is otherwise flexible
    :param field_name: name of the variable to be forecast
    :param init_date: initiation date of the forecast (datetime object)
    :param poi_start: date of the start of the period of interest (datetime object)
    :param poi_end: date of the end of the period of interest (datetime object)
    :param time_label: [default 'time'] time axis label in the netcdf file
    :param period: [default 12] period of the data to be used for deriving the climatology
    :param weights_flag: [default 0] type of ensemble weighting to be used:
        0: No weighting
        1: Weighting using the proximity of the ensemble member year to the initiation date
        2: Weighting using a monthly data included in weighting_data_file
    :param weighting_data_file: [default 'None'] text file containing the data to be used for weighting. The data are
                                in the format used for the NOAA composite and correlation site (format described here:
                                https://psl.noaa.gov/data/composites/createtime.html)
    :param weighting_strength: [default 1] coefficient specifying the strength of the weighting used when weights_flag
                                is set to 1 or 2. 0 indicates no weighting; floats >0 indicates weighting is applied.
                                Users should experiment to find the most appropriate weighting strength
    :param do_increments: [default 1] flag specifying whether or not the ensemble members should be incremented from
                                the initial state. Set do_increments to 0 for no incrementing; 1 for incrementing
    :param suppress_datetime_conversion: [default False] whether to suppress the conversion of datetime.datetimes to
                                np.datetime64 for poi_start, poi_end, time_label

    Returns
    -------
    xarray dataset on the same grid and using the same dimensions as datafile, with an additional dimension 'ensemble'
    specifying the ensemble number. The dataset includes the following variables:
        ensemble_out: array containing the full forecast ensemble (dimensions <datafile geographical dimensions>,
                      <datafile time dimension>, ensemble)
        weights: array containing the the weights applied to each ensemble member at each point in space (dimensions
                 <datafile geographical dimensions>, ensemble). Note that in the current version of the code,
                 weights is constant over the geographical domain
        ens_mean: weighted ensemble mean (dimensions <datafile geographic dimensions>)
        ens_std: weighted ensemble standard deviation (dimensions <datafile geographic dimensions>)
        clim: climatology of the data in datafile (based on the user specified periodicity)
    
    Example function call:
    ---------------------
    import datetime as dtmod
    from general_tamsat_alert import do_forecast
    
    
    field_name='precip'
    time_label='time'
    datafile='pr_gpcc_africa.nc'
    init_date=dtmod.datetime(1997,9,1)
    poi_start=dtmod.datetime(1997,10,1)
    poi_end=dtmod.datetime(1997,10,1)
    period=12
    weights_flag=2
    weighting_data_file='oni.data'
    do_increments=0
    weighting_strength=1

    tmpout=do_forecast(datafile,field_name,init_date,poi_start,poi_end,
                    time_label,period,weights_flag,weighting_data_file,
                    weighting_strength, do_increments)
    
    The example function call uses regridded and subset GPCC precipiation data, and the Oceanic Nino Index provided by NOAA. Convenience copies of these datasets can be found in https://gws-access.jasmin.ac.uk/public/tamsat/tamsat_alert/example_data/
    
    '''
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

    if not suppress_datetime_conversion:
        if isinstance(init_date, datetime.datetime):
            init_date = np.datetime64(init_date)

        if isinstance(poi_start, datetime.datetime):
            poi_start = np.datetime64(poi_start)

        if isinstance(poi_end, datetime.datetime):
            poi_end = np.datetime64(poi_end)

    init_index = get_index(da, time_label, init_date)
    poi_start_index = get_index(da, time_label, poi_start)
    poi_end_index = get_index(da, time_label, poi_end)


    # Calculate inputs for get_ensembles
    ensemble_start = init_index

    ensemble_length = poi_end_index - init_index + 1

    if poi_start_index < init_index:
        look_back = init_index - poi_start_index
    else:
        look_back = 0


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
    nanless_poi_mean = np.ma.masked_array(poi_mean.values, np.isnan(poi_mean.values))
    ens_mean = np.ma.average(nanless_poi_mean, weights=weights.values[..., 1:], axis=-1)
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
    tmpout_xr["clim"] = (["time_clim"] + dims[:-1], get_climatology(datafile, period, field_name))
    return tmpout_xr


def get_climatology(datafile: str, period: int, field_name: str):
    datain = xr.open_dataset(datafile)[field_name].values
    datain = datain[0:period*(datain.shape[0]//period), ...]

    newshape = ((datain.shape[0]//period, period), datain.shape[1:])

    # https://stackoverflow.com/questions/3204245/how-do-i-convert-a-tuple-of-tuples-to-a-one-dimensional-list-using-list-comprehe
    newshape = [element for tupl in newshape for element in tupl]

    datain = np.reshape(datain, newshape=newshape)
    clim = np.nanmean(datain, axis=0)

    return clim
