import itertools
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode, norm
import cartopy.crs as ccrs
from cartopy.feature import BORDERS
from typing import List, Tuple, Hashable, Iterable, Union, Callable
import pandas as pd

from progress_bar import progress_bar
import main
import weighting_functions
import fastroc

def get_non_nan_coords(ds: xr.Dataset,
                       field: Hashable,
                       lon_label: Hashable = "lon",
                       lat_label: Hashable = "lat",
                       step: int = 10) -> List[Tuple[int, int]]:
    out = []
    # Exclude the boundaries as they may contain wierd points and it is safer to just ignore them
    for lat, lon in itertools.product(range(1, ds.dims[lat_label]-1, step), range(1, ds.dims[lon_label]-1, step)):
        data = np.array(ds.isel(lat=lat, lon=lon)[field])
        if not np.isnan(np.sum(data)):
            out.append((lat, lon))
    if len(out) == 0:
        raise ValueError(f"All points have NaNs at some time in field {field}")
    return out


def get_periodicity(ds: xr.Dataset,
                    field: Hashable,
                    point: Tuple[int, int] = None,
                    lon_label: Hashable = "lon",
                    lat_label: Hashable = "lat",
                    step: int = 10) -> int:
    if point is None:
        search_co_ordinates = get_non_nan_coords(ds, field, lon_label, lat_label, step)
        out = []
        for i, (lat, lon) in progress_bar(
                search_co_ordinates, "Finding Periodicity",
                lambda _1, _2, coords: f"({ds.coords[lat_label].isel({lat_label: coords[0]}):.2f}, "
                                       + f"{ds.coords[lon_label].isel({lon_label: coords[1]}):.2f})"):
            data = np.array(ds.isel({lat_label: lat, lon_label: lon})[field])
            out.append(round(main.get_periodicity(data)))
        print()
        plt.figure()
        plt.hist(out, bins=np.arange(min(out), max(out)), density=True)
        plt.yscale("log")
        m = mode(out)[0][0]
        if m == 1:
            m = input("No periodicity found. \nPlease enter a periodicity in time steps "
                      "(e.g. 12 for an annual period for monthly data): ")
            while True:
                try:
                    return int(m)
                except ValueError:
                    m = input(f"{m} is not a valid integer. Please input again: ")
        return m
    else:
        return round(main.get_periodicity(np.array(ds.isel(lat=point[1], lon=point[0])[field])))


# Ensembles are labeled as positive integers. Ensemble 0 is the true data and ensembles 1+ are ordered from end time
# to start sequentially
def get_ensembles(da: xr.DataArray,
                  period: int,
                  ensemble_length: int,
                  ensemble_start: int,
                  look_back: int = 0,
                  weighting_function: Callable[[int, float], float] = weighting_functions.no_weights,
                  in_progress_bar: bool = False,
                  lon_label: Hashable = "lon",
                  lat_label: Hashable = "lat",
                  time_label: Hashable = "time") -> Tuple[xr.DataArray, xr.DataArray]:
    lat = da[lat_label].values
    lon = da[lon_label].values
    time = da[time_label].values[ensemble_start-look_back:ensemble_start+ensemble_length]

    start_times = np.arange(ensemble_start % period, len(da[time_label]) - ensemble_length, period)
    start_times = start_times[(start_times != ensemble_start)]
    start_times = np.insert(start_times, 0, ensemble_start)

    ensemble_count = len(start_times)
    ensemble_indices = np.arange(0, ensemble_count)

    data = np.empty((len(time), len(lat), len(lon), len(ensemble_indices)))
    ensembles = xr.DataArray(data, coords=[time, lat, lon, ensemble_indices], dims=[time_label, lat_label, lon_label, "ensemble"],
                             name="data")

    weight_data = np.empty((len(lat), len(lon), len(ensemble_indices)))
    weights = xr.DataArray(weight_data, coords=[lat, lon, ensemble_indices], dims=[lat_label, lon_label, "ensemble"],
                           name="weights")
    for index, start_time in progress_bar(start_times, "Getting ensemble",
                                          lambda i, imax, val: f"{da[time_label][val].values}",
                                          in_progress_bar=in_progress_bar):
        ensembles[look_back:, :, :, index] = da[start_time:start_time+ensemble_length, :, :].values
        weights[:, :, index] = weighting_function(start_time, 1)
    ensembles[:,:,:,:] -= ensembles[look_back,:,:,:]
    ensembles[:,:,:,:] += da[ensemble_start,:,:]
    ensembles[:look_back, :, :, :] = da[ensemble_start-look_back:ensemble_start, :, :].values[:,:,:,np.newaxis]

    return ensembles, weights


def get_mean_data(da: xr.DataArray,
                  weights: xr.DataArray = None) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    if weights is not None:
        parray = da.isel(ensemble=slice(1,None)).weighted(weights.isel(ensemble=slice(1,None)))
    else:
        parray = da.isel(ensemble=slice(1,None))

    mean = parray.mean(dim="ensemble")
    bias = mean - da.sel(ensemble=0)
    rel_bias = bias/da.sel(ensemble=0)
    return mean, bias, rel_bias


def plot_ensembles(da: xr.DataArray,
                   weights: xr.DataArray = None,
                   quantiles: Iterable[int] = (0.25, 0.5, 0.75),
                   data_min: float = None, data_max: float = None,
                   robust: bool = True,
                   plot_value: bool = True, plot_abs_bias: bool = True, plot_rel_bias: bool = True,
                   only_mean: bool = False, subplot_index: Union[Tuple[int, int, int], Tuple[int]] = None,
                   lon_label: Hashable = "lon", lat_label: Hashable = "lat"
                   ) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    if weights is not None:
        toqarray = da.weighted(weights)
    else:
        toqarray = da

    if only_mean:
        to_plot, bias_array, rel_bias_array = get_mean_data(da, weights)
        if subplot_index is not None:
            plt.subplot(*subplot_index)
        if plot_value:
            to_plot.plot(x=lon_label, y=lat_label, cmap="viridis", vmin=data_min, vmax=data_max, robust=robust)
            plt.suptitle("Values")
        if plot_abs_bias:
            bias_array.plot(x=lon_label, y=lat_label, robust=robust)
            plt.suptitle("Absolute bias")
        if plot_rel_bias:
            rel_bias_array.plot(x=lon_label, y=lat_label, robust=robust)
            plt.suptitle("Relative bias")
        return to_plot, bias_array, rel_bias_array

    blank_data = np.empty((len(da.coords[lat_label]), len(da.coords[lon_label]), len(quantiles)))
    quantile_array = xr.DataArray(blank_data,
                                  coords=[da.coords[lat_label], da.coords[lon_label],list(quantiles)],
                                  dims=[lat_label, lon_label,"quantile"])

    for i, q in progress_bar(quantiles, "Getting quantile", lambda i, imax, val : f"{val}"):
        quantile_array[:,:,i] = toqarray.quantile(q, dim="ensemble")

    if plot_value:
        quantile_array.plot(x=lon_label, y=lat_label, col="quantile", cmap="viridis", vmin=data_min, vmax=data_max, robust=robust)
        plt.suptitle("Values")
    bias_array = quantile_array - da.sel(ensemble=0)
    if plot_abs_bias:
        bias_array.plot(x=lon_label, y=lat_label, col="quantile", robust=robust)
        plt.suptitle("Absolute bias")
    rel_bias_array = bias_array/da.sel(ensemble=0)
    if plot_rel_bias:
        rel_bias_array.plot(x=lon_label, y=lat_label, col="quantile", robust=robust)
        plt.suptitle("Relative bias")
    return quantile_array, bias_array, rel_bias_array


def plot_changing_lookback(da: xr.DataArray,
                           period: int,
                           ensemble_length: int,
                           ensemble_start: int,
                           wf: Callable[[int, float], float] = weighting_functions.no_weights,
                           point: Tuple[int, int]=(150, 150),
                           plot_mean: bool = True, plot_bias: bool = True, plot_rel_bias: bool = True,
                           projection: ccrs.Projection = ccrs.PlateCarree(),
                           lon_label: Hashable = "lon", lat_label: Hashable = "lat", time_label: Hashable = "time"
                           ) -> None:
    offset = np.arange(0, ensemble_length)
    mean_array = xr.DataArray(np.empty((len(da[lat_label]), len(da[lon_label]), len(offset),3)),
                              [da[lat_label], da[lon_label], offset, ["mean", "bias", "relative bias"]],
                              [lat_label, lon_label, "offset", "variable"])
    ensembles, _ = get_ensembles(da, period, ensemble_length, ensemble_start)

    spaghetti_array = xr.DataArray(np.empty((len(ensembles.coords[time_label]), len(ensembles.coords["ensemble"]), len(offset))),
                                   [ensembles.coords[time_label], ensembles.coords["ensemble"], offset], [time_label, "ensemble", "offset"])
    # plt.figure()
    for _, i in progress_bar(offset, "Getting Offset"):
        ensembles, weights = get_ensembles(da, period, ensemble_length-i, ensemble_start+i, look_back=i, in_progress_bar=True,
                                           weighting_function=wf, lon_label=lon_label, lat_label=lat_label, time_label=time_label)
        mean, bias, rel_bias = get_mean_data(ensembles[-1,:,:,:], weights)

        mean_array[:,:,i,0] = mean.values
        mean_array[:,:,i,1] = bias.values
        mean_array[:,:,i,2] = rel_bias.values
        if ensembles.shape[-1] > spaghetti_array.shape[1]:
            spaghetti_array[:,:,i] = ensembles.isel(lat=point[1], lon=point[0]).values[:, :spaghetti_array.shape[1]]
        else:
            spaghetti_array[:, :, i] = ensembles.isel(lat=point[1], lon=point[0]).values

    proj_args = dict(projection=projection)

    if plot_mean:
        i = mean_array[:,:,:,0].plot.imshow(transform=ccrs.PlateCarree(), x=lon_label, y=lat_label, col="offset", robust=True, col_wrap=3, subplot_kws=proj_args)
        for ax in plt.gcf().axes[:-1]:
            ax.coastlines()
    if plot_bias:
        i = mean_array[:,:,:,1].plot.imshow(transform=ccrs.PlateCarree(), x=lon_label, y=lat_label, col="offset", robust=True, col_wrap=3, center=0, subplot_kws=proj_args)
        for ax in plt.gcf().axes[:-1]:
            ax.coastlines()
    if plot_rel_bias:
        i = mean_array[:,:,:,2].plot.imshow(transform=ccrs.PlateCarree(), x=lon_label, y=lat_label, col="offset", robust=True, col_wrap=3, center=0, subplot_kws=proj_args)
        for ax in plt.gcf().axes[:-1]:
            ax.coastlines()

    spaghetti_array.plot.line(x=time_label, col="offset", add_legend=False, col_wrap=3)

def get_ensemble_indices(da: xr.DataArray, prediction_date: str, start_dates: List[str], time_label: Hashable = 'time'):
    indices = np.arange(len(da[time_label]))
    start_indices = indices[da[time_label].isin(da[time_label].sel({time_label: start_dates}, method="nearest"))]
    end_index = indices[da[time_label] == da[time_label].sel({time_label: prediction_date}, method="nearest")][0]
    ensemble_lengths = (end_index - start_indices + 1)  # Start and end inclusive so it needs an extra timestep

    if np.sum(da[time_label].isin(np.array(start_dates, dtype=np.datetime64))) < len(start_dates):
        print("Warning: not all prediction times are in the dataset. Using nearest neighbours.")
    if np.datetime64(prediction_date) not in da[time_label]:
        print(f"Warning: prediction date is not in the dataset. Using nearest neighbour {da[end_index]}.")
    return start_indices, ensemble_lengths, end_index

def plot_predictions(da: xr.DataArray, prediction_date: str, start_dates: List[str], period: int,
                     wf: Callable[[int, xr.DataArray], Callable[[int, float], int]] = lambda _1, _2: weighting_functions.no_weights,
                     lat_label: Hashable = 'lat', lon_label: Hashable = 'lon', time_label: Hashable = 'time', data_label = "data",
                     mean_kwargs={}, bias_kwargs={}, mean_robust=True, bias_robust=True):
    start_indices, ensemble_lengths, end_index = get_ensemble_indices(da, prediction_date, start_dates, time_label=time_label)

    mean_data = xr.DataArray(np.empty((len(da[lat_label]), len(da[lon_label]), len(start_dates))),
                             [da[lat_label], da[lon_label], da[time_label][start_indices]],
                             [lat_label, lon_label, "start date"])
    bias_data = xr.DataArray(np.empty((len(da[lat_label]), len(da[lon_label]), len(start_dates))),
                             [da[lat_label], da[lon_label], da[time_label][start_indices]],
                             [lat_label, lon_label, "start date"])
    climate_mean = da[end_index%period::period, :, :].mean(time_label).values[:,:,np.newaxis]
    for i,_ in progress_bar(start_dates, "Getting ensembles", lambda _1,_2,val: val):
        ensembles, weights = get_ensembles(da, period, ensemble_lengths[i], start_indices[i], 0, wf(start_indices[i], da), True,
                                           lon_label=lon_label, lat_label=lat_label, time_label=time_label)
        mean, bias, _ = get_mean_data(ensembles[-1, :, :, :], weights)
        mean_data[:,:,i] = mean.values
        bias_data[:,:,i] = bias.values
    proj_args = dict(projection=ccrs.PlateCarree())
    mean_data.plot.imshow(x=lon_label, y=lat_label, col="start date", robust=mean_robust, transform=ccrs.PlateCarree(), subplot_kws=proj_args, vmin=0, cmap='cividis', **mean_kwargs)
    plt.suptitle(f"Mean {data_label} for {da[time_label][end_index].values}", y=1)
    for ax in plt.gcf().axes[:-1]:
        ax.coastlines()
        ax.add_feature(BORDERS)


    (mean_data - climate_mean).plot.imshow(
        x=lon_label,
        y=lat_label,
        col="start date",
        robust=mean_robust,
        transform=ccrs.PlateCarree(),
        subplot_kws=proj_args,
        cmap="BrBG",
        **bias_kwargs,
    )
    plt.suptitle(f"Anomaly {data_label} from climate mean for {da[time_label][end_index].values}", y=1)
    for ax in plt.gcf().axes[:-1]:
        ax.coastlines()
        ax.add_feature(BORDERS)

    bias_data.plot.imshow(x=lon_label, y=lat_label, col="start date", robust=bias_robust, cmap="BrBG", transform=ccrs.PlateCarree(), subplot_kws=proj_args, **bias_kwargs)
    plt.suptitle(f"Anomaly {data_label} from observed for {da[time_label][end_index].values}", y=1)
    for ax in plt.gcf().axes[:-1]:
        ax.coastlines()
        ax.add_feature(BORDERS)


def get_hindcasts_observed(da: xr.DataArray, ensemble_lengths: List[int], start_indices: List[int], period: int,
               wf: Callable[[int, xr.DataArray], Callable[[int, float], int]] = lambda _1, _2: weighting_functions.no_weights,
               lat_label: Hashable = 'lat', lon_label: Hashable = 'lon', time_label: Hashable = 'time', in_progress_bar=False):

    mean = []
    stddev = []
    observed = []
    for i, start_index in enumerate(start_indices):
        hindcast_indices = np.arange(start_index%period, len(da[time_label])-ensemble_lengths[i], period)
        mean.append(xr.DataArray(np.empty((len(da[lat_label]), len(da[lon_label]), len(hindcast_indices))),
                         [da[lat_label], da[lon_label], hindcast_indices],
                         [lat_label, lon_label, "hindcast"]))
        observed.append(xr.DataArray(np.empty((len(da[lat_label]), len(da[lon_label]), len(hindcast_indices))),
                        [da[lat_label], da[lon_label], hindcast_indices],
                        [lat_label, lon_label, "hindcast"]))
        stddev.append(xr.DataArray(np.empty((len(da[lat_label]), len(da[lon_label]), len(hindcast_indices))),
                        [da[lat_label], da[lon_label], hindcast_indices],
                        [lat_label, lon_label, "hindcast"]))
        for j, index in progress_bar(hindcast_indices, f"Calculating hindcasts ({i+1}/{len(start_indices)})", in_progress_bar=in_progress_bar):
            ensembles, weights = get_ensembles(da, period, ensemble_lengths[i], index, weighting_function=wf(index, da),
                                               lat_label=lat_label, lon_label=lon_label, time_label=time_label, in_progress_bar=True)
            data, _, _ = get_mean_data(ensembles[-1,:,:,:], weights)
            mean[-1][:,:,j] = data.values
            stddev[-1][:,:,j] = ensembles[-1,:,:,1:].std(dim="ensemble")
            observed[-1][:,:,j] = da[index+ensemble_lengths[i]-1,:,:].values
    return mean, stddev, observed


def plot_ppmcc(da: xr.DataArray, prediction_date: str, start_dates: List[str], period: int,
               wf: Callable[[int, xr.DataArray], Callable[[int, float], int]] = lambda _1, _2: weighting_functions.no_weights,
               lat_label: Hashable = 'lat', lon_label: Hashable = 'lon', time_label: Hashable = 'time'):
    start_indices, ensemble_lengths, _ = get_ensemble_indices(da, prediction_date, start_dates, time_label=time_label)
    ppmcc = xr.DataArray(
        np.empty((len(da[lat_label]), len(da[lon_label]), len(start_dates))),
        [da[lat_label], da[lon_label], da[time_label][start_indices]],
        [lat_label, lon_label, "start dates"],
    )
    rmse = xr.DataArray(
        np.empty((len(da[lat_label]), len(da[lon_label]), len(start_dates))),
        [da[lat_label], da[lon_label], da[time_label][start_indices]],
        [lat_label, lon_label, "start dates"],
    )
    hindcasts, _, observed = get_hindcasts_observed(da, ensemble_lengths, start_indices, period, wf, lat_label, lon_label, time_label)
    for i, start_index in enumerate(start_indices):
        ppmcc[:,:,i] = xr.corr(hindcasts[i], observed[i], "hindcast")
        rmse[:,:,i] = ((hindcasts[i] - observed[i]) ** 2).mean(dim="hindcast")**0.5
    (ppmcc**2).plot.imshow(x=lon_label,y=lat_label,col="start dates", vmin=0, vmax=1, cmap="viridis")
    rmse.plot.imshow(x=lon_label,y=lat_label,col="start dates", robust=True, cmap="viridis")


def get_roc_auc(da: xr.DataArray, prediction_date: str, start_dates: List[str], period: int, threshold_value: float = 0.2,
                 wf: Callable[[int, xr.DataArray], Callable[[int, float], int]] = lambda _1, _2: weighting_functions.no_weights,
                integration_steps=50,
                 lat_label: Hashable = 'lat', lon_label: Hashable = 'lon', time_label: Hashable = 'time'):
    start_indices, ensemble_lengths, end_index = get_ensemble_indices(
        da, prediction_date, start_dates, time_label=time_label
    )
    means, stddev, observed = get_hindcasts_observed(
        da, ensemble_lengths, start_indices, period, wf, lat_label, lon_label, time_label
    )
    roc_auc = xr.DataArray(
        np.empty((len(da[lat_label]), len(da[lon_label]), len(start_dates))),
        [da[lat_label], da[lon_label], da[time_label][start_indices]],
        [lat_label, lon_label, "start dates"],
    )
    climate_mean = da[end_index % period :: period, :, :].mean(dim=time_label)
    climate_std = da[end_index % period :: period, :, :].std(dim=time_label)
    threshold = norm.ppf(threshold_value) * climate_std + climate_mean
    for i, start_index in progress_bar(start_indices, "Calculating ROC AUC"):
        events = xr.DataArray(
            np.array(observed[i] < threshold, dtype=bool), means[i].coords
        )
    
        standardised = (threshold - means[i]) / stddev[i]
        percentiles = xr.DataArray(norm.cdf(standardised), means[i].coords)
    
        roc_auc[:, :, i] = fastroc.calc_roc_auc(
            events.values, percentiles.values, thread_count=16, integral_precision=integration_steps
        )
    return roc_auc

def plot_roc_auc(da: xr.DataArray, prediction_date: str, start_dates: List[str], period: int, threshold_value: float = 0.2,
                 wf: Callable[[int, xr.DataArray], Callable[[int, float], int]] = lambda _1, _2: weighting_functions.no_weights,
                 lat_label: Hashable = 'lat', lon_label: Hashable = 'lon', time_label: Hashable = 'time'):
    roc_auc = get_roc_auc(da, prediction_date, start_dates, period, threshold_value, wf, 50, lat_label, lon_label, time_label)
    roc_auc.plot(x=lon_label,y=lat_label,col="start dates", cmap="plasma", vmin=0.5)


def read_noaa_data_file(fname: str, time_axis: xr.DataArray = None, time_label: str = 'time', replace_given_nan_value=True):
    """

    Data format (BNF for anyone that can read it):
    <ws-char> ::= " " | "\t"
    <ws> ::= <ws-char> | <ws_char> <ws>
    <ws-opt> ::= "" | <ws>
    <digit> ::= "0"|"1"|"2"|"3"|"4"|"5"|"6"|"7"|"8"|"9"
    <year> ::= <digit> <digit> <digit> <digit>
    <natural> ::= <digit> | <digit> <natural>
    <integer> ::= "-" <natural> | "+" <natural>
    <real> ::= <integer> "." <natural> | <real> "E" <integer> | <real> "e" <integer>
    <line-end> ::= <ws-opt> "\\n" | <ws-opt> "\\r\\n"

    <real-3> ::= <real> <ws> <real> <ws> <real>
    <real-12> ::= <real-3> <ws> <real-3> <ws> <real-3> <ws> <real-3>

    <any-str> ::= "" | <any-str> <*>

    <header> ::= <ws-opt> <year> <ws> <year>
    <line> ::= <ws-opt> <year> <ws> <real-12> | <ws-opt> <year> <ws> <real-12> <ws> <any-str>
    <data-matrix> ::= <line> | <line> <line-end> <data>

    <nan-value> ::= <real>

    <footer> ::= <any-str>

    <file-format> ::= <header> <line-end> <data-matrix> <line-end> <nan-value> <line-end> <footer>

    Additionally:
     --  The <year> at the start of each <line> must increase sequentially from the first <year> in <header> to the
         last <year> in <header> inclusive.
     --  <*> indicates the wildcard character that matches any singular ASCII character
     --  All characters in the file *must* be valid ASCII characters

    :param fname:
    :param time_axis:
    :param time_label:
    :param replace_given_nan_value:
    :return:
    """
    with open(fname, 'rt') as f:
        try:
            miny, maxy = f.readline().strip().split()
            miny = int(miny)
            maxy = int(maxy)
        except ValueError:
            raise ValueError("File does not contain start/end year on first line")
        data = []
        for index, year in enumerate(range(miny, maxy+1)):
            try:
                line = f.readline()
                line = line.strip().split()
                assert line[0] == str(year)
                line = [np.float64(i) for i in line[1:13]]
                data.extend(line)
            except ValueError:
                raise ValueError(f"Line {index+2} contains invalid number(s)")
            except AssertionError:
                raise ValueError(f"Unexpected value {line[0]} at start of line {index+2}")
        nan_value = np.float64(f.readline().strip())
        data = np.array(data)
        if replace_given_nan_value:
            data[(data <= nan_value+0.000001)] = np.nan

        # Whether the time axis is start or end of month (it is usually end of month)
        freq = "MS"
        start = f"{miny}-01-01"
        end = f"{maxy}-12-01"
        da = xr.DataArray(data, [xr.date_range(start, end, freq=freq)], [time_label])

    if time_axis is None:
        return da
    else:
        return da.interp({time_label: time_axis}, kwargs={'fill_value': 'extrapolate'})


def get_weighting_roc_improvement(da: xr.DataArray, prediction_date: str, start_dates: List[str], period: int, threshold_value: float = 0.2,
                 wf: Callable[[int, xr.DataArray], Callable[[int, float], int]] = lambda _1, _2: weighting_functions.no_weights,
                 lat_label: Hashable = 'lat', lon_label: Hashable = 'lon', time_label: Hashable = 'time'):
    unweighted  = get_roc_auc(da, prediction_date, start_dates, period, threshold_value, lat_label=lat_label, lon_label=lon_label, time_label=time_label, integration_steps=5000)
    weighted = get_roc_auc(da, prediction_date, start_dates, period, threshold_value, wf, integration_steps=5000, lat_label=lat_label, lon_label=lon_label, time_label=time_label)
    unweighted.plot.imshow(x=lon_label, y=lat_label, col="start dates")
    weighted.plot.imshow(x=lon_label, y=lat_label, col="start dates")
    anomaly = (weighted - unweighted)
    anomaly.plot.imshow(x=lon_label, y=lat_label, col="start dates", cmap="BrBG", vmin=-0.5, vmax=0.5)
    rel_anomaly = anomaly / (1 - unweighted)
    rel_anomaly.plot.imshow(x=lon_label, y=lat_label, col="start dates", cmap="BrBG", robust=True)

def process_data(filename: str,
                 ensemble_length: int,
                 field: Hashable,
                 weighting_data_file: str = None,
                 ensemble_start: int = None,
                 lon_label: str = "lon", lat_label: str = "lat", time_label: str = "time"
                 ) -> Tuple[xr.DataArray, xr.DataArray]:
    ds = xr.load_dataset(filename)
    print(ds)
    period = get_periodicity(ds, field, lon_label=lon_label, lat_label=lat_label, point=(150, 150))
    if weighting_data_file is not None:
        weighting_data = read_noaa_data_file(weighting_data_file, ds.coords[time_label], time_label)
    else:
        weighting_data = np.ones(ds.coords[time_label].shape)

    print(weighting_data)

    print(f"Period of {period} timesteps")
    ''''
    get_weighting_roc_improvement(
        ds[field],
        "2012-07-31",
        ["2012-04-30", "2012-05-31", "2012-06-30", "2012-07-16", "2012-07-31"],
        period,
        wf=lambda true_time, _: weighting_functions.weight_value(true_time, weighting_data),
    )
    get_weighting_roc_improvement(
        ds[field],
        "2012-07-31",
        ["2012-04-30", "2012-05-31", "2012-06-30", "2012-07-16", "2012-07-31"],
        period,
        wf=lambda true_time, _: weighting_functions.nearby_time(true_time),
    )
    '''
    # plot_predictions(ds[field], '2001-03-01', ['2000-12-01','2001-01-01', '2001-02-01', '2001-02-15', '2001-03-01'],
    #                 period, wf=lambda true_time, _: weighting_functions.nearby_time(true_time), mean_kwargs={'vmin':0})
    # plot_ppmcc(ds[field],'2012-07-31', ['2012-04-30', '2012-05-31', '2012-06-30', '2012-07-16', '2012-07-31'],period)
    # plot_roc_auc(ds[field][:,:,:], '2012-07-31', ['2012-04-30', '2012-05-31', '2012-06-30', '2012-07-16', '2012-07-31'],period)
    plot_roc_auc(ds[field][:,:,:], '2012-07-31', ['2011-07-31'], period, wf=lambda true_time, _: weighting_functions.nearby_time(true_time))
    plt.show()

    if ensemble_start is None:
        ensemble_start = ds.dims[time_label] - ensemble_length

    '''
    wf = weighting_functions.neighbour_error(ensemble_start, ds[field])
    
    ensembles, weights = get_ensembles(ds[field], period, ensemble_length, ensemble_start, look_back=6,
                                       weighting_function=wf, lon_label=lon_label, lat_label=lat_label, time_label=time_label)
    plot_changing_lookback(ds[field], period, ensemble_length, ensemble_start, wf, lon_label=lon_label, lat_label=lat_label, time_label=time_label)
    plt.show()
    
    return ensembles, weights
    '''


if __name__ == '__main__':
    process_data("drought-model-driving-data_pakistan_19820101-present_0.05.nc", 6, "ndvi", "oni.data")
