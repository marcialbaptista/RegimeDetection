import numpy as np
import pandas as pd
from collections import defaultdict
import math


class Utils:

    @staticmethod
    def calculate_online_derivative_signal(signal):
        derivative_ypts = []
        len_derivative = len(signal)
        if len_derivative <= 5:
            return signal
        first_value = np.nanmedian(signal[:5])
        for index in range(len_derivative):
            if np.isnan(signal[index]):
                val = np.nan
            else:
                val = signal[index] - first_value
            derivative_ypt = val
            derivative_ypts.append(derivative_ypt)
        return np.array(derivative_ypts)

    @staticmethod
    def calculate_derivative_signal(signal, window_size=10):
        derivative_ypts = []
        len_derivative = len(signal) - window_size
        if len_derivative < 0:
            return derivative_ypts
        for index in range(len_derivative):
            derivative_ypt = signal[index + window_size] - signal[index]
            derivative_ypts.append(derivative_ypt)
        return np.array(derivative_ypts)

    @staticmethod
    def normalize_old(signal, range):
        return (signal / np.max(signal))  * range

    @staticmethod
    def normalize_old(signal):
        return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

    @staticmethod
    def normalize(signal):
        if len(signal) == 0 or np.isnan(np.sum(signal)):
            return signal
        if (np.max(signal) - np.min(signal)) == 0:
            return (signal - np.min(signal))/ np.max(signal)
        return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

    @staticmethod
    def standardize(signal):
        return (signal - np.mean(signal)) / (np.std(signal))

    @staticmethod
    def moving_average(signal, window_size):
        new_signal = np.copy(signal)
        if len(signal) == 0:
            return signal
        for i in range(len(signal)):
            new_signal[i] = np.mean(signal[max(0, i-window_size):i])
        new_signal[0] = signal[0]
        return new_signal

    @staticmethod
    def exponential_smoothing(series, alpha):
        """given a series and alpha, return series of expoentially smoothed points"""
        results = np.zeros_like(series)

        if len(series) < 2:
            return series
        # first value remains the same as series,
        # as there is no history to learn from
        previous_result = np.nan
        for t in range(0, series.shape[0]):
            if math.isnan(series[t]):
                results[t] = np.nan
            elif math.isnan(previous_result):
                results[t] = series[t]
                previous_result = results[t]
            else:
                # not math.isnan(previous_result) and not math.isnan(series[t])
                results[t] = alpha * series[t] + (1 - alpha) * previous_result
                previous_result = results[t]

        return results

    @staticmethod
    def double_exponential_smoothing(series, alpha, beta, n_preds=1):
        """
        Given a series, alpha, beta and n_preds (number of
        forecast/prediction steps), perform the prediction.
        """
        n_record = series.shape[0]
        results = np.zeros(n_record + n_preds)

        # first value remains the same as series,
        # as there is no history to learn from;
        # and the initial trend is the slope/difference
        # between the first two value of the series
        level = series[0]
        results[0] = series[0]
        trend = series[1] - series[0]
        for t in range(1, n_record + 1):
            if t >= n_record:
                # forecasting new points
                value = results[t - 1]
            else:
                value = series[t]

            previous_level = level
            level = alpha * value + (1 - alpha) * (level + trend)
            trend = beta * (level - previous_level) + (1 - beta) * trend
            results[t] = level + trend

        # for forecasting beyond the first new point,
        # the level and trend is all fixed
        if n_preds > 1:
            results[n_record + 1:] = level + np.arange(2, n_preds + 1) * trend

        return results

    @staticmethod
    def moving_quantile(signal, window_size):
        new_signal = np.copy(signal)
        for i in range(1,len(signal)):
            new_signal[i] = np.quantile(signal[max(0, i - window_size):i], 0.75)
        new_signal[0] = signal[0]
        return new_signal


    @staticmethod
    def frequency_amplitude(s):
        n = len(s)
        fft_result = np.fft.fft(s, n)
        num_freq_bins = len(fft_result)
        sampling_rate = 1
        fft_freqs = np.fft.fftfreq(num_freq_bins, d=1 / sampling_rate)
        half_freq_bins = num_freq_bins // 2

        fft_freqs = fft_freqs[:half_freq_bins]
        fft_result = fft_result[:half_freq_bins]
        fft_amplitudes = np.abs(fft_result)

        #fft_amplitudes = 2 * fft_amplitudes / (len(s))
        return (fft_freqs, fft_amplitudes)

    @staticmethod
    def process_removals(plane_tail):
        removals_df = pd.read_csv('data//removals_wouter2021_2.csv')
        removals_df['Rem/Inst Date'] = pd.to_datetime(removals_df.loc[:, 'Rem/Inst Date'], format='%m/%d/%Y')
        type_removals = np.unique(removals_df['Name'].values.astype(str))
        removals_aircraft_dic = defaultdict()

        average_tsis_removals_dic = defaultdict(int)
        for type_removal in type_removals:
            tsis_removal_type = removals_df.loc[removals_df.Name == type_removal, 'TSI Hours'].values
            average_tsis_removals_dic[type_removal] = np.median(tsis_removal_type)

        removals_df_airplane = removals_df.loc[removals_df['AC Reg'] == plane_tail, :]

        indexes = np.argsort(removals_df_airplane['Rem/Inst Date'].values)
        removals_aircraft_dic['Date'] = [removals_df_airplane['Rem/Inst Date'].values[i] for i in indexes]
        removals_aircraft_dic['Previous Date'] = [None] + removals_aircraft_dic['Date'][:-1]
        removals_aircraft_dic['TSI Hours'] = [removals_df_airplane['TSI Hours'].values[i] for i in indexes]
        removals_aircraft_dic['Type Removal'] = [removals_df_airplane['Name'].values[i] for i in indexes]
        removals_aircraft_dic['Position'] = [removals_df_airplane['Position'].values[i] for i in indexes]
        removals_aircraft_dic['Reason'] = [removals_df_airplane['Reason'].values[i] for i in indexes]
        removals_aircraft_dic['HealthCheck'] = [removals_df_airplane['health check or specifically requested PCCV replacement'].values[i] for i in indexes]
        removals_aircraft_dic['Root Cause'] = [removals_df_airplane['Shop Findings'].values[i] for i in indexes]

        return type_removals, average_tsis_removals_dic, removals_aircraft_dic
