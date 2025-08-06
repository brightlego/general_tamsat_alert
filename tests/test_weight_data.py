import numpy as np
import xarray as xr
import pytest
import datetime

import src.general_tamsat_alert as gta


datafile = "drought-model-driving-data_senegal_19830101-present_0.05.nc"
weightfile = "precip_weights.nc"

def test_weights():
    yearin = 2011
    init_month = 7
    init_date = datetime.datetime(yearin, init_month, 1)
    poi_start_date = datetime.datetime(yearin, 7, 1)  # For now keep the forecast init in the same year as the poi
    poi_end_date = datetime.datetime(yearin, 8, 1)
    field_name = 'TAMSAT_rfe'
    period = 24


    result = gta.do_forecast(datafile, field_name, init_date, poi_start_date, poi_end_date, period=period, weights_flag=3, weighting_data_file=weightfile)
    result2 = gta.do_forecast(datafile, field_name, init_date, poi_start_date, poi_end_date, period=period,
                             weights_flag=0, weighting_data_file=weightfile)
    assert np.mean(np.abs(result["ens_mean"] - result2["ens_mean"])) > 0.01