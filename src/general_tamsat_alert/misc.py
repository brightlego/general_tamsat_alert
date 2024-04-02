import xarray as xr
import numpy as np


def read_noaa_data_file(
    fname: str,
    time_axis: xr.DataArray = None,
    time_label: str = "time",
    replace_given_nan_value=True,
):
    """

    Data format (BNF for anyone that can read it):
    <ws-char> ::= " " | "\t"
    <ws> ::= <ws-char> | <ws_char> <ws>
    <ws-opt> ::= "" | <ws>
    <digit> ::= "0"|"1"|"2"|"3"|"4"|"5"|"6"|"7"|"8"|"9"
    <year> ::= <digit> <digit> <digit> <digit>
    <natural> ::= <digit> | <digit> <natural>
    <integer> ::= "-" <natural> | "+" <natural>
    <real> ::= <integer> "." <natural>
             | <real> "E" <integer>
             | <real> "e" <integer>
    <line-end> ::= <ws-opt> "\\n" | <ws-opt> "\\r\\n"

    <real-3> ::= <real> <ws> <real> <ws> <real>
    <real-12> ::= <real-3> <ws> <real-3> <ws> <real-3> <ws> <real-3>

    <any-str> ::= "" | <any-str> <*>

    <header> ::= <ws-opt> <year> <ws> <year>
    <line> ::= <ws-opt> <year> <ws> <real-12>
             | <ws-opt> <year> <ws> <real-12> <ws> <any-str>
    <data-matrix> ::= <line> | <line> <line-end> <data>

    <nan-value> ::= <real>

    <footer> ::= <any-str>

    <file-format> ::= <header> <line-end> <data-matrix> <line-end>
                      <nan-value> <line-end> <footer>

    Additionally:
     --  The <year> at the start of each <line> must increase
         sequentially from the first <year> in <header> to the last
         <year> in <header> inclusive.
     --  <*> indicates the wildcard character that matches any singular
         ASCII character
     --  All characters in the file *must* be valid ASCII characters

    :param fname:
    :param time_axis:
    :param time_label:
    :param replace_given_nan_value:
    :return:
    """
    with open(fname, "rt") as f:
        try:
            miny, maxy = f.readline().strip().split()
            miny = int(miny)
            maxy = int(maxy)
        except ValueError:
            raise ValueError("File does not contain start/end year on first "
                             "line")
        data = []
        for index, year in enumerate(range(miny, maxy + 1)):
            try:
                line = f.readline()
                line = line.strip().split()
                assert line[0] == str(year)
                line = [np.float64(i) for i in line[1:13]]
                data.extend(line)
            except ValueError:
                raise ValueError(f"Line {index+2} contains invalid number(s)")
            except AssertionError:
                raise ValueError(
                    f"Unexpected value {line[0]} at start of line {index+2}"
                )
        nan_value = np.float64(f.readline().strip())
        data = np.array(data)
        if replace_given_nan_value:
            data[(data <= nan_value + 0.000001)] = np.nan

        # Whether the time axis is start or end of month
        # (it is usually end of month)
        freq = "MS"
        start = f"{miny}-01-01"
        end = f"{maxy}-12-01"
        da = xr.DataArray(data,
                          [xr.date_range(start, end, freq=freq)],
                          [time_label])

    if time_axis is None:
        return da
    else:
        return da.interp({time_label: time_axis},
                         kwargs={"fill_value": "extrapolate"})