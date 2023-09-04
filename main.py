import pandas as pd
import numpy as np
import scipy.fft as fft
from scipy.stats import linregress
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from weighting_functions import *
import scipy.stats as stats

def get_data(filename):
    data = pd.read_csv(filename, header=None, parse_dates=[[1,2,3]])
    data = data.rename(columns={"1_2_3":"Date", 0:"Data"})
    return data


def get_data2(filename):
    data = pd.read_csv(filename, parse_dates=[0])
    return data


def rms(data):
    data -= np.mean(data)
    return np.sqrt(np.mean(data**2))


def standardise(data):
    data -= np.mean(data)
    data /= rms(data)
    return data


def extract_peaks(data):
    changes = np.convolve(data, [0.5, -1, 0.5], mode='valid')
    changes /= data[1:-1]
    indices = np.arange(1, len(data)-1)
    # print(indices[(changes < -0.5)])
    return indices[(changes < -0.5) & (indices > 4)]


def get_periodicity(data):
    data = np.array(data)
    raw_data = data - np.mean(data)
    out = np.zeros(raw_data.shape)
    for index in range(1,len(raw_data)-1):
        out[index] = np.mean((raw_data[:-index] - raw_data[index:])**2)
    out = out[1:-len(out)//4]
    result = linregress(np.arange(len(out)), out)
    # plt.figure()
    # plt.plot(out)
    out -= np.arange(len(out))*result.slope + result.intercept
    fourier = fft.fft(out)
    fourier = np.abs(fourier)
    freq = np.arange(len(fourier))/len(fourier)
    fourier = fourier[(freq < 0.5)]
    freq = freq[(freq < 0.5)]
    # plt.figure()
    # plt.plot(data)
    # plt.figure()
    # plt.plot(fourier)
    peaks = extract_peaks(fourier)
    # peak_T = 1/freq[peaks]
    # print(peak_T)
    # plt.figure()
    # plt.plot(data.copy())
    # plt.plot(standardise(out.copy()))
    if len(peaks) == 0:
        return 1
    else:
        return 1/freq[peaks[np.argmax(fourier[peaks])]]


def validate_weighting(data, period, start_time, predict_time, weighting_strength, weighting_function_type=None):
    X = []
    r_values = []
    rms_values = []
    bias_values = []
    true_data = data[start_time : start_time + predict_time]
    for strength in weighting_strength:
        weighting_function = no_weights
        if weighting_function_type == 'nearby_time':
            weighting_function = nearby_time(start_time, strength)
        ensembles, weights = get_ensembles(data, period, start_time%period, predict_time, weighting_function)
        errors = []
        correls = []
        bias = []
        for e in ensembles:
            errors.append(np.sqrt((e[-1]+true_data[0]-true_data[-1])**2))
            correls.append(stats.pearsonr(e+true_data[0], true_data)[0])
            bias.append(e[-1]+true_data[0]-true_data[-1])
        X.append(strength)
        r_values.append(np.average(correls, weights=weights))
        rms_values.append(np.average(errors, weights=weights))
        bias_values.append(np.average(bias, weights=weights))
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(X, r_values)
    plt.xlabel("Weight Strength")
    plt.ylabel("r")
    plt.subplot(3,1,2)
    plt.plot(X, rms_values)
    plt.xlabel("Weight Strength")
    plt.ylabel("RMSE")
    plt.subplot(3, 1, 3)
    plt.plot(X, bias_values)
    plt.xlabel("Weight Strength")
    plt.ylabel("Bias")


def validate(data, period, start_offsets, offset_period, predict_time, weighting_function_type=None):
    for index, offset in zip(range(len(start_offsets)), start_offsets):
        plt.subplot(np.ceil(np.sqrt(len(start_offsets))), np.ceil(np.sqrt(len(start_offsets))), index+1)
        X = []
        Y = []
        for year in range(offset, len(data)-predict_time, offset_period):
            weighting_function = no_weights
            if weighting_function_type == 'nearby_time':
                weighting_function = nearby_time(year, 10)
            true_data = data[year: year + predict_time]
            ensembles, weights = get_ensembles(data, period, offset, predict_time, weighting_function)
            # ensemble_mean = np.average([np.mean(e+true_data[0]) for e in ensembles], weights=weights)
            # true_mean = np.mean(true_data)

            # X.append(true_mean)
            # Y.append(ensemble_mean)

            X.append(true_data[-1])
            Y.append(np.average([e[-1]+true_data[0] for e in ensembles], weights=weights))

        plt.scatter(X, Y, alpha=0.5)
        ax = plt.gca()
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]

        # now plot both limits against eachother
        ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
    plt.tight_layout()


def get_ensembles(past_data, period, start_time, predict_time, weighting_function=no_weights):
    ensembles = []
    weights = []
    for i in range(start_time, len(past_data)-predict_time, period):
        ensembles.append(past_data[i:i+predict_time].copy())
        ensembles[-1] -= ensembles[-1][0]
        weights.append(weighting_function(i))

    weights = np.array(weights)
    weights /= np.sum(weights)
    return ensembles, weights


def process_temp_data():
    data = get_data('formated_stats.csv')
    plt.plot(data['Date'], data['Data'])
    period = get_periodicity(data['Data'])
    print(f'Data has a period of {period}')
    period = round(period)
    pred_date = len(data) // period * period - period
    ensembles, weights = get_ensembles(np.array(data['Data'])[:pred_date], period, 0, 180)

    times = data['Date'][pred_date:pred_date + 180]
    combined_x = np.array([], dtype=times.dtype)
    combined_y = np.array([])
    combined_weights = np.array([])

    for ensemble, weight in zip(ensembles, weights):
        combined_x = np.append(combined_x, times)
        combined_y = np.append(combined_y, ensemble)
        combined_weights = np.append(combined_weights, [weight]*len(ensemble))

    combined_y += data['Data'][pred_date]

    plt.figure()
    plt.hist2d(combined_x.astype(int), combined_y, weights=combined_weights)

    plt.figure()
    for ensemble in ensembles:
        plt.plot(times, ensemble)
    plt.show()


def process_other_data(predict_time=12, field='Precip'):
    data = get_data2('metrics_601.csv')
    plt.plot(data['Time'], data[field])
    period = get_periodicity(data[field])
    period = round(period)
    print(period)
    pred_date = len(data) // period * period - int(predict_time/period+1)*period
    ensembles, weights = get_ensembles(np.array(data[field])[:pred_date], period, pred_date % period, predict_time)

    times = data['Time'][pred_date:pred_date + predict_time]
    combined_x = np.array([], dtype=times.dtype)
    combined_y = np.array([])

    for ensemble in ensembles:
        combined_x = np.append(combined_x, times)
        combined_y = np.append(combined_y, ensemble)

    combined_y += data[field][pred_date]

    plt.figure()
    plt.plot(times.astype(np.int64), data[field][pred_date:pred_date+predict_time])
    plt.hist2d(combined_x.astype(np.int64), combined_y, bins=[predict_time, 20])

    plt.figure()
    for ensemble in ensembles:
        plt.plot(times, ensemble + data[field][pred_date])
    plt.plot(data['Time'][max(pred_date - predict_time//2,0):pred_date+1],
             data[field][max(pred_date - predict_time//2,0):pred_date+1])

    validate(np.array(data[field]), period, range(0,24,2), 24, 3)
    validate(np.array(data[field]), period, range(0,24,2), 24, 3, weighting_function_type='nearby_time')
    validate_weighting(np.array(data[field]), period, pred_date, 6, np.arange(0,10,0.01),
                       weighting_function_type='nearby_time')

    plt.show()


if __name__ == '__main__':
    process_other_data(8)
