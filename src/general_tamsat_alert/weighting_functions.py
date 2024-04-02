import numpy as np
import xarray as xr
from typing import Callable, Union, Sequence, Tuple, Optional

WeightingFunctionType = Callable[[int], Union[float, np.ndarray, xr.DataArray]]
WeightingFunctionIntermediateType = Callable[[int], WeightingFunctionType]
WeightingFunctionBuilderType = Callable[..., WeightingFunctionIntermediateType]


def no_weights(member_index: int) -> float:
    return 1.0


def no_weights_intermediate(initiation_index: int) -> WeightingFunctionType:
    return no_weights


def no_weights_builder(weighting_strength: float = 1.0) -> WeightingFunctionIntermediateType:
    return no_weights_intermediate


def weight_time_builder(period: int = 24, weighting_strength: float = 1.0) -> WeightingFunctionIntermediateType:
    def weight_time_intermediate(initiation_index: int) -> WeightingFunctionType:
        def weight_time(member_index: int) -> float:
            if member_index == initiation_index:
                return 0.0
            else:
                return np.exp(-0.0001 * (weighting_strength * (member_index - initiation_index) * period/24) ** 2)
        return weight_time
    return weight_time_intermediate


def weight_neighbour_error_builder(data: xr.DataArray,
                                   neighbour_shifts: Sequence[Tuple[int, int]] = ((1, 0), (0, 1), (-1, 0), (0, -1)),
                                   neighbour_weights: Optional[Sequence[Tuple[int, int]]] = None,
                                   lat_label: str = "lat", lon_label: str = "lon",
                                   weighting_strength: float = 1.0) -> WeightingFunctionIntermediateType:
    def weight_neighbor_error_intermediate(initiation_index: int) -> WeightingFunctionType:
        if neighbour_weights is None:
            internal_neighbour_weights = [1 / np.sqrt(x * x + y * y) for y,x in neighbour_shifts]
        else:
            internal_neighbour_weights = neighbour_weights

        shifts = np.arange(len(neighbour_shifts))
        weights = xr.DataArray(internal_neighbour_weights, coords=[shifts], dims=["shift"])

        lat = data.coords[lat_label].values
        lon = data.coords[lon_label].values

        error_values = np.empty((len(lat), len(lon), len(shifts)))

        errors = xr.DataArray(error_values, coords=[lat, lon, shifts], dims=["lat", "lon", "shift"])
        shifted_data = []
        for (y, x) in neighbour_shifts:
            shifted_data.append(data[initiation_index, :, :].shift(lat=y, lon=x).values)

        def weight_neighbour_error(member_index: int) -> np.ndarray:
            if member_index == initiation_index:
                return np.zeros((len(lat), len(lon)))

            values = data[member_index, :, :]
            for index, (y, x) in enumerate(neighbour_shifts):
                errors[:, :, index] = ((values.shift({lat_label: y, lon_label: x}) - shifted_data[index])**2).values
            mean = errors.weighted(weights).mean(dim="shift", skipna=True).values
            mean[(mean == 0)] += 0.000001
            mean = 1/mean
            mean[(np.isnan(mean))] = 0
            return mean
        return weight_neighbour_error
    return weight_neighbor_error_intermediate


def weight_value_builder(values: Sequence[float], weighting_strength: float = 1.0) -> WeightingFunctionIntermediateType:
    def weight_value_intermediate(initiation_index: int) -> WeightingFunctionType:
        def weight_value(member_index: int) -> float:
            delta = values[member_index] - values[initiation_index]
            delta = np.abs(delta)
            return np.exp(-(weighting_strength*delta)**2)
        return weight_value
    return weight_value_intermediate
