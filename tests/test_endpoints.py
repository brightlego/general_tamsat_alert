import numpy as np
import xarray as xr
import pytest
import datetime
import matplotlib.pyplot as plt
import base64
import io

import src.general_tamsat_alert as gta

test_data = b'Q0RGAgAAAAAAAAAKAAAAAQAAAAR0aW1lAAAAkAAAAAAAAAAAAAAACwAAAAQAAAAEdGltZQAAAAEAAAAAAAAADAAAAAIAAAAFdW5pdHMAA\
AAAAAACAAAAFWRheXMgc2luY2UgMTk4My0wMS0xNQAAAAAAAAhjYWxlbmRhcgAAAAIAAAATcHJvbGVwdGljX2dyZWdvcmlhbgAAAAAEAAACQAAAAAAAAAGU\
AAAABG5kdmkAAAABAAAAAAAAAAwAAAACAAAAC2Nvb3JkaW5hdGVzAAAAAAIAAAAHbGF0IGxvbgAAAAAKX0ZpbGxWYWx1ZQAAAAAABQAAAAF/wAAAAAAABQA\
AAkAAAAAAAAAD1AAAAANsYXQAAAAAAAAAAAwAAAABAAAACl9GaWxsVmFsdWUAAAAAAAYAAAABf/gAAAAAAAAAAAAGAAAACAAAAAAAAAYUAAAAA2xvbgAAAA\
AAAAAADAAAAAEAAAAKX0ZpbGxWYWx1ZQAAAAAABgAAAAF/+AAAAAAAAAAAAAYAAAAIAAAAAAAABhwAACaGAAAmlgAAJqUAACayAAAmwQAAJtEAACbgAAAm7\
wAAJv4AACcOAAAnHQAAJywAACc7AAAnSwAAJ1oAACdqAAAneQAAJ4gAACeXAAAnpwAAJ7YAACfFAAAn1AAAJ+QAACfzAAAoAwAAKBIAACgfAAAoLgAAKD4A\
AChNAAAoXAAAKGsAACh7AAAoigAAKJkAACioAAAouAAAKMcAACjXAAAo5gAAKPUAACkEAAApFAAAKSMAACkyAAApQQAAKVEAAClgAAApcAAAKX8AACmNAAA\
pnAAAKawAACm7AAApygAAKdkAACnpAAAp+AAAKgcAACoWAAAqJgAAKjUAACpFAAAqVAAAKmMAACpyAAAqggAAKpEAACqgAAAqrwAAKr8AACrOAAAq3gAAKu\
0AACr6AAArCQAAKxkAACsoAAArNwAAK0YAACtWAAArZQAAK3QAACuDAAArkwAAK6IAACuyAAArwQAAK9AAACvfAAAr7wAAK/4AACwNAAAsHAAALCwAACw7A\
AAsSwAALFoAACxnAAAsdgAALIYAACyVAAAspAAALLMAACzDAAAs0gAALOEAACzwAAAtAAAALQ8AAC0fAAAtLgAALT0AAC1MAAAtXAAALWsAAC16AAAtiQAA\
LZkAAC2oAAAtuAAALccAAC3UAAAt4wAALfMAAC4CAAAuEQAALiAAAC4wAAAuPwAALk4AAC5dAAAubQAALnwAAC6MAAAumwAALqoAAC65AAAuyQAALtgAAC7\
nAAAu9gAALwY+dsi0Pk/fOz41P30+MCDFPjMzMz49cKQ+TMzNPmBBiT5vnbI+haHLPpYEGT6dLxs+r52yPsgxJz7EGJM+zMzNPtiTdT7MSbo+zlYEPsrAgz\
7Jul4+u+dtPqdsiz6O2Rc+ZWBCPjU/fT4dsi0+HKwIPh64Uj4k3S8+Mi0OPkKPXD5Lxqg+aHKwPom6Xj6euFI+rpeNPrjU/j68an8+xBiTPs3S8j7KwIM+y\
sCDPsxJuj7FHrg+tLxqPpgQYj587ZE+RqfwPhul4z4HKwI+AAAAPf3ztj4MSbo+JN0vPjMzMz40OVg+RaHLPlwo9j6FHrg+n753Pq0OVj7JN0w+2hysPt87\
ZD7fvnc+3KwIPtiTdT7LQ5Y+uNT+PrAgxT6cKPY+hR64PmBBiT5BiTc+NkWiPjhR7D5ItDk+YEGJPnKwIT57520+hR64Ppqfvj687ZE+y0OWPs5WBD7NT98\
+zlYEPtWBBj7aHKw+3Cj2PtaHKz7KPXE+uuFIPql41T6OVgQ+cKPXPkSbpj4l41Q+GZmaPheNUD4i0OU+NT99Pkan8D5dLxs+ggxKPp0vGz62yLQ+v3zuPr\
xqfz63ztk+rItEPrCj1z65WBA+wxJvPsUeuD664Ug+qHKwPpHrhT5ul40+PGp/PhmZmj4LQ5Y+BysCPggxJz4MSbo+Cj1xPggxJz4QYk4+J++ePlLxqj6Sb\
pg+szMzPrnbIz6/fO4+w5WBPscrAj7CDEo+xiTdPs1P3z7Gp/A+udsjPq0OVj6crAhAKXMzMzMzC8ArWZmZmZlx'


def test_ensemble_time_axis(expected, **kwargs):
    expected = np.array(expected, dtype=np.datetime64)
    datafile = io.BytesIO(base64.b64decode(test_data))
    forecast_xr = gta.do_forecast(datafile=datafile, **kwargs)
    assert np.all(expected == forecast_xr['time'].data)


def test_endpoints():
    yearin = 2011
    init_month = 7
    init_date = datetime.datetime(yearin, 7, 1)
    poi_start_date = datetime.datetime(yearin, 7, 1)  # For now keep the forecast init in the same year as the poi
    poi_end_date = datetime.datetime(yearin, 8, 1)
    field_name = 'ndvi'
    period = 24
    test_ensemble_time_axis(expected=['2011-06-30', '2011-07-15', '2011-07-31'],
                            field_name=field_name, init_date=init_date,
                            poi_start=poi_start_date, poi_end=poi_end_date, period=period,
                            do_increments=1, weights_flag=0,
                            weighting_strength=1,
                            poi_endpoint_inclusive=True, time_nearest_neighbour=True)
    test_ensemble_time_axis(expected=['2011-07-15', '2011-07-31'],
                            field_name=field_name, init_date=init_date,
                            poi_start=poi_start_date, poi_end=poi_end_date, period=period,
                            do_increments=1, weights_flag=0,
                            weighting_strength=1,
                            poi_endpoint_inclusive=True, time_nearest_neighbour=False)
    test_ensemble_time_axis(expected=['2011-06-30', '2011-07-15'],
                            field_name=field_name, init_date=init_date,
                            poi_start=poi_start_date, poi_end=poi_end_date, period=period,
                            do_increments=1, weights_flag=0,
                            weighting_strength=1,
                            poi_endpoint_inclusive=False, time_nearest_neighbour=True)
    test_ensemble_time_axis(expected=['2011-07-15'],
                            field_name=field_name, init_date=init_date,
                            poi_start=poi_start_date, poi_end=poi_end_date, period=period,
                            do_increments=1, weights_flag=0,
                            weighting_strength=1,
                            poi_endpoint_inclusive=False, time_nearest_neighbour=False)

    init_date = datetime.datetime(yearin, 7, 1)
    poi_start_date = datetime.datetime(yearin, 7, 1)  # For now keep the forecast init in the same year as the poi
    poi_end_date = datetime.datetime(yearin, 8, 14)
    test_ensemble_time_axis(expected=['2011-06-30', '2011-07-15', '2011-07-31', '2011-08-15'],
                            field_name=field_name, init_date=init_date,
                            poi_start=poi_start_date, poi_end=poi_end_date, period=period,
                            do_increments=1, weights_flag=0,
                            weighting_strength=1,
                            poi_endpoint_inclusive=True, time_nearest_neighbour=True)
    test_ensemble_time_axis(expected=['2011-07-15', '2011-07-31'],
                            field_name=field_name, init_date=init_date,
                            poi_start=poi_start_date, poi_end=poi_end_date, period=period,
                            do_increments=1, weights_flag=0,
                            weighting_strength=1,
                            poi_endpoint_inclusive=True, time_nearest_neighbour=False)
    test_ensemble_time_axis(expected=['2011-07-15', '2011-07-31'],
                            field_name=field_name, init_date=init_date,
                            poi_start=poi_start_date, poi_end=poi_end_date, period=period,
                            do_increments=1, weights_flag=0,
                            weighting_strength=1,
                            poi_endpoint_inclusive=False, time_nearest_neighbour=False)