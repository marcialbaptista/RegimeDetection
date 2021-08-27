import pickle as pkl
from pathlib import Path
import os
import pyarrow.parquet as pq
import numpy as np
from collections import defaultdict
from os import listdir
from os.path import isfile, join
import pandas as pd
from matplotlib import pyplot as plt
from numpy import sqrt, abs, round
from scipy.stats import norm
from hampel import hampel
import time
import math
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN


class Fligh_Stages():

    def moving_average(self, a, n=3):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    def twoSampZ(self, X1, X2, mudiff, sd1, sd2, n1, n2):
        pooledSE = sqrt((sd1 ** 2) / n1 + (sd2 ** 2) / n2)
        z = ((X1 - X2) - mudiff) / pooledSE
        pval = 2 * (1 - norm.cdf(abs(z)))
        return z, pval

    def estimate_elbow_ztest(self, signal, windowSize, sensibility, threshold=0.1):
        for i in range(windowSize, len(signal) - windowSize):
            past_signal = signal[(i - windowSize):(i + 1)]
            next_signal = signal[(i):(i + windowSize)]
            past_mean = np.mean(past_signal)
            next_mean = np.mean(next_signal)
            past_std = np.std(past_signal)
            next_std = np.std(next_signal)

            if next_std == past_std == 0:
                continue

            z, pval = self.twoSampZ(next_mean, past_mean, sensibility, next_std, past_std, windowSize, windowSize)
            if pval > 0.01 and next_mean > threshold:
                return i
        print("Could not detect with z-score... generating average value")
        return -1

    def normalize_signal(self, signal):
        return np.array((signal - np.min(signal)) / (np.max(signal) - np.min(signal)))

    def transform_into_binary_signal(self, signal, key='OFF'):
        transformed_signal = (signal == key)
        return transformed_signal * 0.5

    def plot_altitude_overtime(self, flight_df):
        plt.plot(flight_df['row_number'], self.normalize_signal(flight_df['pressure_altitude']))
        plt.plot(flight_df['row_number'], self.normalize_signal(flight_df['flight_phase']))
        plt.show()

    def custom_hampel(self, vals_orig, window_size=7, n=3):
        '''
        vals: pandas series of values from which to remove outliers
        k: size of window (including the sample; 7 is equal to 3 on either side of value)
        '''

        # Make copy so original not edited
        vals = pd.DataFrame(vals_orig.copy())

        # Hampel Filter
        L = 1.4826
        rolling_median = vals.rolling(window=window_size, center=True).median()
        MAD = lambda x: np.median(np.abs(x - np.median(x)))
        rolling_MAD = vals.rolling(window=window_size, center=True).apply(MAD)
        threshold = n * L * rolling_MAD
        difference = np.abs(vals - rolling_median)

        '''
        Perhaps a condition should be added here in the case that the threshold value
        is 0.0; maybe do not mark as outlier. MAD may be 0.0 without the original values
        being equal. See differences between MAD vs SDV.
        '''

        outlier_idx = difference > threshold
        vals[outlier_idx] = rolling_median[outlier_idx]
        return (vals)

    def calculate_derivative_signal(self, signal, derivative_window_size=20):
        derivative_ypts = [0] * derivative_window_size
        for index in range(len(signal) - derivative_window_size):
            derivative_ypt = signal[index + derivative_window_size] - signal[index]
            derivative_ypts.append(derivative_ypt)
        return np.array(derivative_ypts)

    def do_kmeans_flight_stages(self, data, norm_ypts, number_flight_stages=5):
        flight_stages = None
        try:
            kmeans = KMeans(n_clusters=number_flight_stages, random_state=0).fit(data)
            flight_stages = kmeans.labels_
        except ValueError as e:
            print("ERROR: error on producing flight stages ")
            flight_stages = []
        return flight_stages

    def detect_cruise(self, derivative_signal, norm_xpts, norm_ypts, min_cruise_altitude_threshold,
                      max_cruise_altitude_threshold):
        columns = ["Derivative"]
        index_condition = np.where(
            (norm_ypts <= max_cruise_altitude_threshold) & (norm_ypts > min_cruise_altitude_threshold))
        signal = derivative_signal[index_condition]
        if len(signal) <= 4:
            return False, [], 0, 0

        data = np.array([self.normalize_signal(signal)])
        df_kmeans = pd.DataFrame(data=data.T, columns=columns)
        flight_stages = self.do_kmeans_flight_stages(df_kmeans, norm_ypts, number_flight_stages=4)

        if len(flight_stages) == 0:
            return False, flight_stages, 0, 0

        cruise_flight_stage, numberpts_cruise = 0, 0
        for flight_stage in np.unique(flight_stages):
            numberpts_flight_stage = len(flight_stages[flight_stages == flight_stage])
            if numberpts_flight_stage > numberpts_cruise:
                cruise_flight_stage = flight_stage
                numberpts_cruise = numberpts_flight_stage
        min_altitude_this_cruise = np.min(norm_ypts[index_condition][flight_stages == cruise_flight_stage])
        median_derivative_this_cruise = np.median(
            np.abs(derivative_signal[index_condition][flight_stages == cruise_flight_stage]))

        return median_derivative_this_cruise < 0.001, flight_stages, cruise_flight_stage, min_altitude_this_cruise

    def detect_takeoff(self, derivative_signal, xpts, norm_ypts, highest_cruise_starting_xpt):
        columns = ["Derivative"]
        index_condition = np.where(
            (xpts <= highest_cruise_starting_xpt) & (derivative_signal < 0.01) & (norm_ypts < 0.1))
        signal = derivative_signal[index_condition]
        if len(signal) <= 4:
            return False, [], 0
        data = np.array([self.normalize_signal(signal)])
        df_kmeans = pd.DataFrame(data=data.T, columns=columns)
        flight_stages = self.do_kmeans_flight_stages(df_kmeans, norm_ypts, number_flight_stages=2)

        takeoff_flight_stage, altitude_takeoff = 0, 0
        for flight_stage in np.unique(flight_stages):
            altitude_flight_stage = np.median(norm_ypts[index_condition][flight_stages == flight_stage])
            if altitude_flight_stage < altitude_takeoff:
                takeoff_flight_stage = flight_stage
                altitude_takeoff = altitude_flight_stage
        median_derivative_this_takeoff = np.median(
            np.abs(derivative_signal[index_condition][flight_stages == takeoff_flight_stage]))

        return median_derivative_this_takeoff < 0.01, flight_stages, takeoff_flight_stage

    def detect_taxi(self, derivative_signal, xpts, norm_ypts, highest_cruise_ending_xpt):
        columns = ["Derivative", "Time"]
        index_condition = np.where((xpts >= highest_cruise_ending_xpt) & (derivative_signal < 0.01) & (norm_ypts < 0.1))
        signal = derivative_signal[index_condition]
        if len(signal) <= 4:
            return False, [], 0

        data = np.array([self.normalize_signal(signal), self.normalize_signal(norm_ypts[index_condition])])
        df_kmeans = pd.DataFrame(data=data.T, columns=columns)
        flight_stages = self.do_kmeans_flight_stages(df_kmeans, norm_ypts, number_flight_stages=3)

        taxi_flight_stage, altitude_taxi = 0, 2
        for flight_stage in np.unique(flight_stages):
            altitude_flight_stage = np.max(norm_ypts[index_condition][flight_stages == flight_stage])
            if altitude_flight_stage < altitude_taxi:
                taxi_flight_stage = flight_stage
                altitude_taxi = altitude_flight_stage
        median_derivative_this_taxi = np.median(
            np.abs(derivative_signal[index_condition][flight_stages == taxi_flight_stage]))

        return median_derivative_this_taxi < 0.01, flight_stages, taxi_flight_stage

    def process_flight_aircraft(self, flight_df, plane_tail):
        flight_df = flight_df.copy()
        xpts = flight_df['row_number'].values
        norm_xpts = self.normalize_signal(xpts)

        flight_df['Flight_Stage'] = 'UNKNOWN'

        orig_ypts = flight_df['pressure_altitude'].values
        # hampel_ypts = custom_hampel(orig_ypts, window_size=50, n=1).values.ravel()
        norm_ypts = self.normalize_signal(orig_ypts)

        derivative_signal = self.calculate_derivative_signal(norm_ypts)

        if False:
            index_flight_stage = 0
            for flight_stage in np.unique(flight_df.Flight_Stage):
                signal_total = flight_df['pressure_altitude'].values
                signal_flight_stage = flight_df['pressure_altitude'].values[
                    flight_df['Flight_Stage'].values == flight_stage]
                xx = flight_df['row_number'].values[flight_df['Flight_Stage'].values == flight_stage]
                plt.scatter(xx / 3600.0, signal_flight_stage, alpha=0.75, color='black')
                plt.title('Randomly selected flight of aircraft ' + plane_tail)
                plt.ylabel('Altitude')
                plt.grid(True, linestyle='--', linewidth=0.5)
                plt.xlabel('Time in hours')
                index_flight_stage += 1
            plt.legend()
            plt.tight_layout()
            plt.savefig('figs/begin.png')
            plt.close()

        found_at_least_one_cruise = False
        cruise_starting_xpts, cruise_ending_xpts, cruise_lengths = [], [], []
        min_cruise_altitude_threshold, max_cruise_altitude_threshold = 0.98, 1.0
        while min_cruise_altitude_threshold > 0.15:
            found_cruise, cruise_stages, cruise_flight_stage, min_altitude_previous_cruise = self.detect_cruise(
                derivative_signal, norm_xpts, norm_ypts, min_cruise_altitude_threshold=min_cruise_altitude_threshold,
                max_cruise_altitude_threshold=max_cruise_altitude_threshold)
            index_condition = np.where(
                (norm_ypts <= max_cruise_altitude_threshold) & (norm_ypts > min_cruise_altitude_threshold))
            if ~found_cruise:
                max_cruise_altitude_threshold = min_cruise_altitude_threshold
                min_cruise_altitude_threshold = max_cruise_altitude_threshold - 0.01
                continue

            cruise_starting_xpts.append(xpts[index_condition][cruise_stages == cruise_flight_stage][0])
            cruise_ending_xpts.append(xpts[index_condition][cruise_stages == cruise_flight_stage][-1])
            cruise_lengths.append(xpts[index_condition][cruise_stages == cruise_flight_stage][-1] -
                                  xpts[index_condition][cruise_stages == cruise_flight_stage][0])

            if found_cruise and not found_at_least_one_cruise:  # first cruise found
                found_at_least_one_cruise = True
                xpts_cruise = xpts[index_condition][cruise_stages == cruise_flight_stage]
                start_xpt_cruise = xpts_cruise[0]
                end_xpt_cruise = xpts_cruise[-1]
                row_numbers_cruise = flight_df['row_number'].values[
                    np.where((xpts >= start_xpt_cruise) & (xpts <= end_xpt_cruise))]
                flight_df.loc[flight_df['row_number'].isin(row_numbers_cruise), 'Flight_Stage'] = 'CRUISE_ZERO'
                flight_df.loc[flight_df['row_number'] < start_xpt_cruise, 'Flight_Stage'] = 'CLIMB'
                flight_df.loc[flight_df['row_number'] > end_xpt_cruise, 'Flight_Stage'] = 'DESCENT'
            elif found_cruise and found_at_least_one_cruise:  # after first cruise found
                xpts_cruise = xpts[index_condition][cruise_stages == cruise_flight_stage]
                start_xpt_cruise = xpts_cruise[0]
                end_xpt_cruise = xpts_cruise[-1]
                if start_xpt_cruise < cruise_starting_xpts[0]:
                    row_numbers_cruise = flight_df['row_number'].values[
                        np.where((xpts >= start_xpt_cruise) & (xpts <= end_xpt_cruise))]
                    flight_df.loc[flight_df['row_number'].isin(row_numbers_cruise), 'Flight_Stage'] = 'CRUISE_CLIMB'
                else:
                    row_numbers_cruise = flight_df['row_number'].values[
                        np.where((xpts >= start_xpt_cruise) & (xpts <= end_xpt_cruise))]
                    flight_df.loc[flight_df['row_number'].isin(row_numbers_cruise), 'Flight_Stage'] = 'CRUISE_DESCENT'

            dictionary_flight_stages = defaultdict(str)
            dictionary_flight_stages['TAKEOFF'] = 'Takeoff'
            dictionary_flight_stages['DESCENT'] = 'Descent'
            dictionary_flight_stages['TAXI'] = 'Landing'
            dictionary_flight_stages['VALID_CRUISE_SUB'] = 'Cruise'
            dictionary_flight_stages['CRUISE_ZERO'] = 'Cruise'
            dictionary_flight_stages['CRUISE_DESCENT'] = 'Cruise'
            dictionary_flight_stages['CRUISE_CLIMB'] = 'Cruise'
            dictionary_flight_stages['CRUISE_ZERO'] = 'Cruise'
            dictionary_flight_stages['CLIMB'] = 'Climb'

            dictionary_colors = defaultdict(str)
            dictionary_colors['Takeoff'] = 'g'
            dictionary_colors['Climb'] = 'blue'
            dictionary_colors['Cruise'] = 'cyan'
            dictionary_colors['Descent'] = 'orange'
            dictionary_colors['Landing'] = 'red'

            if False:
                index_flight_stage = 0
                for flight_stage in np.unique(flight_df.Flight_Stage):
                    flight_stage_pretty = dictionary_flight_stages[flight_stage]
                    signal_total = flight_df['pressure_altitude'].values
                    signal_flight_stage = flight_df['pressure_altitude'].values[
                        flight_df['Flight_Stage'].values == flight_stage]
                    xx = flight_df['row_number'].values[flight_df['Flight_Stage'].values == flight_stage]
                    plt.scatter(xx / 3600.0, signal_flight_stage, alpha=0.75,
                                color=dictionary_colors[flight_stage_pretty], label=flight_stage_pretty)
                    plt.title('Randomly selected flight of aircraft ' + plane_tail)
                    plt.ylabel('Altitude')
                    plt.grid(True, linestyle='--', linewidth=0.5)
                    plt.xlabel('Time in hours')
                    index_flight_stage += 1
                plt.axhline(max_cruise_altitude_threshold * np.max(signal_total), color='red', linewidth=1,
                            linestyle='--')
                plt.axhline(min_cruise_altitude_threshold * np.max(signal_total), color='red', linewidth=1,
                            linestyle='--')
                plt.legend()
                plt.tight_layout()
                plt.savefig('figs/' + str(min_cruise_altitude_threshold) + '.png')
                plt.close()
                # plt.show()

            max_cruise_altitude_threshold = min_cruise_altitude_threshold
            min_cruise_altitude_threshold = max_cruise_altitude_threshold - 0.01

        if not found_at_least_one_cruise:
            print('Did not find any cruise!')
            return [False] * len(flight_df['row_number'])

        max_cruise_length = np.max(cruise_lengths)
        for cruise_start_pts, cruise_end_pts, cruise_length in zip(cruise_starting_xpts, cruise_ending_xpts,
                                                                   cruise_lengths):
            cruise_length = cruise_end_pts - cruise_start_pts
            if cruise_length == max_cruise_length:
                row_numbers_cruise = flight_df['row_number'].values[
                    np.where((xpts >= cruise_start_pts) & (xpts <= cruise_end_pts))]
                flight_df.loc[flight_df['row_number'].isin(row_numbers_cruise), 'Flight_Stage'] = 'VALID CRUISE'
                median_longest_cruise = np.median(
                    flight_df.loc[flight_df['row_number'].isin(row_numbers_cruise), 'par6_sys_1'].values)
                std_longest_cruise = np.std(
                    flight_df.loc[flight_df['row_number'].isin(row_numbers_cruise), 'par6_sys_1'].values)
                break

        max_altitude_cruise = np.max(flight_df['pressure_altitude'].values)
        for cruise_start_pts, cruise_end_pts, cruise_length in zip(cruise_starting_xpts, cruise_ending_xpts,
                                                                   cruise_lengths):
            row_numbers_cruise = flight_df['row_number'].values[
                np.where((xpts >= cruise_start_pts) & (xpts <= cruise_end_pts))]
            median_cruise = np.median(
                flight_df.loc[flight_df['row_number'].isin(row_numbers_cruise), 'par6_sys_1'].values)
            std_cruise = np.std(flight_df.loc[flight_df['row_number'].isin(row_numbers_cruise), 'par6_sys_1'].values)
            altitude_cruise = np.min(
                flight_df.loc[flight_df['row_number'].isin(row_numbers_cruise), 'pressure_altitude'].values)

            if abs(median_cruise - std_cruise) < abs(
                    median_longest_cruise + std_longest_cruise) and altitude_cruise >= 0.5 * max_altitude_cruise:
                flight_df.loc[flight_df['row_number'].isin(row_numbers_cruise), 'Flight_Stage'] = 'VALID CRUISE SUB'

        highest_cruise_starting_xpt = cruise_starting_xpts[-1]
        found_takeoff, takeoff_stages, takeoff_flight_stage = self.detect_takeoff(derivative_signal, xpts, norm_ypts,
                                                                                  highest_cruise_starting_xpt=highest_cruise_starting_xpt)

        if not found_takeoff:
            print('Did not find any takeoff!')
            return [False] * len(flight_df['row_number'])

        index_condition = np.where(
            (xpts <= highest_cruise_starting_xpt) & (derivative_signal < 0.01) & (norm_ypts < 0.1))
        start_xpt_takeoff = xpts[index_condition][takeoff_stages == takeoff_flight_stage][0]
        end_xpt_takeoff = xpts[index_condition][takeoff_stages == takeoff_flight_stage][-1]
        row_numbers_takeoff = flight_df['row_number'].values[np.where((xpts <= end_xpt_takeoff))]
        flight_df.loc[flight_df['row_number'].isin(row_numbers_takeoff), 'Flight_Stage'] = 'TAKEOFF'

        if False:
            plt.scatter(xpts, norm_ypts)
            plt.plot(xpts, normalize_signal(flight_df['par6_sys_1']))
            flight_stage = takeoff_flight_stage
            feature = 'par6_sys_1'

            plt.scatter(xpts[index_condition][takeoff_stages == flight_stage],
                        norm_ypts[index_condition][takeoff_stages == flight_stage],
                        label="Altitude on stage" + str(flight_stage))
            plt.plot(xpts[index_condition][takeoff_stages == flight_stage][20:-20],
                     normalize_signal(flight_df[feature].values[index_condition][takeoff_stages == flight_stage])[
                     20:-20],
                     label=feature + ' on stage ' + str(flight_stage))
            feature2 = 'par5_sys_gen'

            plt.scatter(xpts[index_condition][takeoff_stages == flight_stage][20:-20], transform_into_binary_signal(
                flight_df[feature2].values[index_condition][takeoff_stages == flight_stage], 'CLOSE')[20:-20],
                        label=feature2 + ' on stage ' + str(flight_stage))

            plt.title('Flight phases')
            plt.xlabel('Row number')
            plt.ylabel(feature)
            plt.legend()
            plt.show()

        highest_cruise_ending_xpt = cruise_ending_xpts[0]
        found_taxi, taxi_stages, taxi_flight_stage = self.detect_taxi(derivative_signal, xpts, norm_ypts,
                                                                 highest_cruise_ending_xpt=highest_cruise_ending_xpt)

        if not found_taxi:
            print('Did not find any taxi!')
            return [False] * len(flight_df['row_number'])

        index_condition = np.where((xpts >= highest_cruise_ending_xpt) & (derivative_signal < 0.01) & (norm_ypts < 0.1))
        start_xpt_taxi = xpts[index_condition][taxi_stages == taxi_flight_stage][0]
        end_xpt_taxi = xpts[index_condition][taxi_stages == taxi_flight_stage][-1]
        row_numbers_taxi = flight_df['row_number'].values[np.where((xpts >= start_xpt_taxi))]
        flight_df.loc[flight_df['row_number'].isin(row_numbers_taxi), 'Flight_Stage'] = 'TAXI'

        if False:

            dictionary_flight_stages = defaultdict(str)
            dictionary_flight_stages['TAKEOFF'] = 'Takeoff'
            dictionary_flight_stages['DESCENT'] = 'Descent'
            dictionary_flight_stages['TAXI'] = 'Landing'
            dictionary_flight_stages['VALID CRUISE SUB'] = 'Cruise'
            dictionary_flight_stages['CRUISE_ZERO'] = 'Cruise'
            dictionary_flight_stages['CRUISE_DESCENT'] = 'Cruise'
            dictionary_flight_stages['CRUISE_CLIMB'] = 'Cruise'
            dictionary_flight_stages['CRUISE_ZERO'] = 'Cruise'
            dictionary_flight_stages['CLIMB'] = 'Climb'

            dictionary_colors = defaultdict(str)
            dictionary_colors['Takeoff'] = 'green'
            dictionary_colors['Climb'] = 'blue'
            dictionary_colors['Cruise'] = 'cyan'
            dictionary_colors['Descent'] = 'orange'
            dictionary_colors['Landing'] = 'red'

            index_flight_stage = 0
            for flight_stage in np.unique(flight_df.Flight_Stage):
                flight_stage_pretty = dictionary_flight_stages[flight_stage]
                signal_flight_stage = flight_df['pressure_altitude'].values[
                    flight_df['Flight_Stage'].values == flight_stage]
                xx = flight_df['row_number'].values[flight_df['Flight_Stage'].values == flight_stage]
                plt.scatter(xx / 3600.0, signal_flight_stage, alpha=0.75, c=dictionary_colors[flight_stage_pretty],
                            label=flight_stage_pretty)
                plt.title('Randomly selected flight of aircraft ' + plane_tail)
                plt.ylabel('Altitude')
                plt.grid(True, linestyle='--', linewidth=0.5)
                plt.xlabel('Time in hours')
                index_flight_stage += 1
            plt.axhline(min_cruise_altitude_threshold * np.max(signal_flight_stage), color='red', linewidth=1,
                        linestyle='--')
            plt.axhline(0, color='red', linewidth=1, linestyle='--')
            plt.legend()
            plt.tight_layout()
            plt.savefig('figs/final.png')
            plt.close()

        if False:
            plt.scatter(xpts, norm_ypts)
            plt.plot(xpts, normalize_signal(flight_df['par6_sys_1']))
            flight_stage = taxi_flight_stage
            feature = 'par6_sys_1'
            plt.scatter(xpts[index_condition][taxi_stages == flight_stage],
                        norm_ypts[index_condition][taxi_stages == flight_stage],
                        label="Altitude on stage" + str(flight_stage))
            plt.plot(xpts[index_condition][taxi_stages == flight_stage][20:-20],
                     normalize_signal(flight_df[feature].values[index_condition][taxi_stages == flight_stage])[20:-20],
                     label=feature + ' on stage ' + str(flight_stage))
            feature2 = 'par5_sys_gen'


            plt.scatter(xpts[index_condition][taxi_stages == flight_stage][20:-20], transform_into_binary_signal(
                flight_df[feature2].values[index_condition][taxi_stages == flight_stage], 'CLOSE')[20:-20],
                        label=feature2 + ' on stage ' + str(flight_stage))

            plt.title('Flight phases')
            plt.xlabel('Row number')
            plt.ylabel(feature)
            plt.legend()
            plt.show()

        COLORS = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen', 'pink', 'gray', 'silver',
                  'chartreuse', 'darkturquoise']

        if False:
            norm_par_6_sys1 = [elm * max_altitude_cruise for elm in normalize_signal(flight_df['par6_sys_1'])]
            j = 0
            for flight_stage in np.unique(flight_df['Flight_Stage']):
                xpts_flight_stage = flight_df.loc[flight_df['Flight_Stage'] == flight_stage, 'row_number']
                altitude_pressure_stage = flight_df.loc[flight_df['Flight_Stage'] == flight_stage, 'pressure_altitude']
                norm_par_6_sys1_stage = np.array(norm_par_6_sys1)[flight_df['Flight_Stage'].values == flight_stage]
                plt.scatter(xpts_flight_stage, altitude_pressure_stage, label="Altitude on stage " + str(flight_stage),
                            color=COLORS[j])
                plt.plot(xpts_flight_stage, norm_par_6_sys1_stage, label="Parameter 6 on stage " + str(flight_stage),
                         color=COLORS[j])
                j += 1
            plt.title('Flight phases')
            plt.legend()
            plt.xlabel('Row number')
            plt.ylabel('Altitude_pressure')
            plt.legend()
            plt.show()

        new_directory_name_path = Path('data/' + str(plane_tail) + '/')
        new_directory_name_path.mkdir(parents=True, exist_ok=True)

        import random
        # flight_df.to_parquet('data/' + str(plane_tail) + '/' + str(flight_df['departure_datetime'][0]).replace(' ', '_').replace(':', '') + '_' + str(int(random.random()*100)) + '.snappy.parquet')

        return flight_df['Flight_Stage'].values
