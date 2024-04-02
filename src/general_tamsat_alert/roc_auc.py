import xarray as xr
import fastroc
from typing import List
from scipy.stats import norm
import numpy as np

from .ensembles import get_ensemble_indices, get_hindcasts_observed
from . import weighting_functions as wfs

def get_roc_auc(
    da: xr.DataArray,
    prediction_date: str,
    start_dates: List[str],
    period: int,
    threshold_value: float = 0.2,
    wf: wfs.WeightingFunctionIntermediateType = wfs.no_weights_intermediate,
    integration_steps=50,
    time_label: str = "time",
):
    start_indices, ensemble_lengths, end_index = get_ensemble_indices(
        da, prediction_date, start_dates, time_label=time_label
    )
    means, stddev, observed = get_hindcasts_observed(
        da,
        ensemble_lengths,
        start_indices,
        period,
        wf,
        time_label,
    )
    coords = [da[i] for i in da.dims if i != time_label]
    dims = [i for i in da.dims if i != time_label]
    roc_auc = xr.DataArray(
        np.empty([len(i) for i in coords] + [len(start_dates)]),
        coords=coords + [da[time_label][start_indices]],
        dims=dims + ["start dates"],
    )
    climate_mean = da.isel({time_label: slice(end_index % period, None, period)}).mean(dim=time_label)
    climate_std = da.isel({time_label: slice(end_index % period, None, period)}).std(dim=time_label)
    threshold = norm.ppf(threshold_value) * climate_std + climate_mean
    for i, start_index in enumerate(start_indices):
        events = xr.DataArray(np.array(observed[i] < threshold, dtype=bool), means[i].coords)

        standardised = (threshold - means[i]) / stddev[i]
        percentiles = xr.DataArray(norm.cdf(standardised), means[i].coords)

        roc_auc[..., i] = fastroc.calc_roc_auc(
            events.values,
            percentiles.values,
            thread_count=16,
            integral_precision=integration_steps,
        )
    return roc_auc
