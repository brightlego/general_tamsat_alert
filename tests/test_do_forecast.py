import numpy as np
import xarray as xr
import pytest
import datetime

import src.general_tamsat_alert as gta

ds = xr.load_dataset("example.nc")

def test_time_out_of_bounds():
    poi_start = datetime.datetime(year=2022, month=2, day=15)
    poi_end = datetime.datetime(year=2022, month=4, day=15)
    init_date = datetime.datetime(year=2021, month=12, day=15)

    tmpout = gta.do_forecast("example.nc", "ndvi", init_date, poi_start, poi_end, period=24, do_increments=1)
    print(tmpout)

    assert tmpout["data"].values.shape == (9, 20, 20, 39)

