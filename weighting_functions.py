import numpy as np
import xarray as xr

def nearby_time(true_time, weighting_strength=1.0):
    def weighting_function(sample_time, weighting_strength=weighting_strength):
        if sample_time == true_time:
            return 0.0
        else:
            return np.exp(-0.0001 * (weighting_strength * (sample_time - true_time))**2)
    return weighting_function


def no_weights(sample_time, weighting_strength=1.0):
    return 1.0


def neighbour_error(true_time, data, neighbour_shifts=((1,0),(0,1),(-1,0),(0,-1)),neighbour_weights=None):
    if neighbour_weights is None:
        neighbour_weights = [1/np.sqrt(x*x + y*y) for y,x in neighbour_shifts]

    shifts = np.arange(len(neighbour_shifts))
    weights = xr.DataArray(neighbour_weights, coords=[shifts],dims=["shift"])
    lat = data.coords["lat"].values
    lon = data.coords["lon"].values
    error_values = np.empty((len(lat), len(lon), len(shifts)))
    errors = xr.DataArray(error_values, coords=[lat, lon, shifts], dims=["lat", "lon", "shift"])
    shifted_data = []
    for (y, x) in neighbour_shifts:
        shifted_data.append(data[true_time, :, :].shift(lat=y, lon=x).values)

    def weighting_function(sample_time, weighting_strength=1.0):
        if sample_time == true_time:
            return np.zeros((len(data.coords["lat"]), len(data.coords["lon"])))

        values = data[sample_time, :, :]
        for index, (y, x) in enumerate(neighbour_shifts):
            errors[:, :, index] = ((values.shift(lat=y, lon=x) - shifted_data[index])**2).values
        mean = errors.weighted(weights).mean(dim="shift", skipna=True).values
        mean[(mean == 0)] += 0.000001
        mean = 1/mean
        mean[(np.isnan(mean))] = 0
        return mean
    return weighting_function


def weight_value(true_time, values):
    def weighting_function(sample_time, weighting_strength=1):
        delta = values[sample_time] - values[true_time]
        delta = np.abs(delta)
        # return 1/(delta**(weighting_strength+1) + 1)
        return np.exp(-(weighting_strength*delta)**2)
    return weighting_function
