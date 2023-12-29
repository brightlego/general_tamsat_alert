from general_tamsat_alert.periodicity import *
import xarray as xr
import numpy as np
import pytest

ds = xr.load_dataset("tests/example.nc")


def test_standardise():
    generator = np.random.default_rng(1)
    for i in range(100):
        data = generator.random(1000)
        standard_data = standardise(data)
        assert np.abs(standard_data.mean()) < 1e-15
        assert np.abs(standard_data.std() - 1) < 1e-15


def test_smooth():
    generator = np.random.default_rng(1)
    data = generator.random(1000)
    smoothed_data = np.zeros(int(len(data)) - 1)
    for index in range(1, int(len(data)) - 1):
        smoothed_data[index - 1] = np.mean((data[:-index] - data[index:]) ** 2)
    assert np.all(smoothed_data == smooth(data, 1))

    for proportion in np.linspace(0.1, 1, 7):
        assert smooth(data, proportion).shape[0] == int(data.shape[0] * proportion) - 1

    with pytest.raises(ValueError):
        smooth(data, 0)


def test_get_axis_periodicity():
    for i in get_non_nan_indices(ds, "ndvi", step=1):
        assert round(get_axis_periodicity(ds["ndvi"][i].values)) % 12 == 0


def test_get_non_nan_indices():
    assert len(get_non_nan_indices(ds, "ndvi", step=1)) == 320
    assert len(get_non_nan_indices(ds, "ndvi", step=10)) == 4
    assert len(get_non_nan_indices(ds, "vhi", step=1)) == 312
    assert len(get_non_nan_indices(ds, "vhi", step=10)) == 4
    assert len(get_non_nan_indices(ds, "precip", step=1)) == 324
    assert len(get_non_nan_indices(ds, "precip", step=10)) == 4

def test_get_periodicity():
    assert get_periodicity(ds, 'ndvi') == 24
    assert get_periodicity(ds, 'ndvi', point={'lon': 0, 'lat': 0}) == 24
    with pytest.raises(OverflowError):
        get_periodicity(ds, 'ndvi', point={'lon':4, 'lat':12})