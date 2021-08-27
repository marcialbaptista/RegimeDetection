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
import ruptures as rpt

class FlightStages:
    """
            A class that is used to represent the sensor data of an
            aircraft

            Attributes
            ----------
            aircraft_id         : str
                the aircraft identifier (PH-01, PH-33, etc.)
            df        : list
                dataframe with sensor data of aircraft flight
            debug               : str
                the warning print flag
            error               : int
                the error print flag
            colors               : list
                list of strings indicating colors
            min_flight_size      : int
                flight size threshold limit
        """
    def __init__(self, aircraft_id, df, debug, error):
        self.aircraft_id = aircraft_id
        self.df = df
        self.debug = debug
        self.error = error
        self.colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k',
                       'Brown', 'ForestGreen', 'pink', 'gray', 'silver',
                       'chartreuse', 'darkturquoise']
        self.min_flight_size = 500
        self.time = self.df['row_number'].values
        self.altitude = self.df['pressure_altitude'].values

    @staticmethod
    def moving_average(signal, window_size):
        new_signal = np.copy(signal)
        if len(signal) == 0:
            return signal
        for i in range(len(signal)):
            new_signal[i] = np.mean(signal[max(0, i - window_size):i])
        new_signal[0] = signal[0]
        return new_signal

    @staticmethod
    def min_max_normalize(signal):
        if len(signal) == 0 or np.isnan(np.sum(signal)):
            return signal
        if (np.max(signal) - np.min(signal)) == 0:
            return (signal - np.min(signal)) / np.max(signal)
        return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

    @staticmethod
    def calc_derivative(signal, window_size=5):
        derivative_ypts = []
        len_derivative = len(signal)
        if len_derivative <= 1:
            return signal
        start_window = max(len_derivative, window_size)
        for index in range(start_window, len_derivative):
            val = np.nan
            if not np.isnan(signal[index]):
                val = signal[index] - signal[index - max(index, window_size)]
            derivative_ypts.append(val)
        return np.array(derivative_ypts)

    def remove_short_phases(self, l):
        flight_phases = self.df['flight_phase'].values
        indexes = np.where(np.diff(flight_phases))[0] + 1
        res = []
        for a, b in zip(indexes[:-1], indexes[1:]):
            phase = l[a:b]
            if np.size(phase) > 10:
                res.extend(phase)
        return np.array(res)

        results = []
        for flight_phase in pd.unique(flight_phases):
            B = np.split(flight_phases, np.where(flight_phases == flight_phase)[0])
            print('B', B, 'B')
            C = np.split(l, np.where(flight_phases == flight_phase)[0])
            f1 = np.vectorize(lambda a: np.size(a) > 50)
            res = np.extract(f1(C), C)
            if len(res) > 0:
                results.append(res[0][0])
        print('results=', results, len(results))
        return results

    def detect(self):
        orig_flight_phases = self.df['flight_phase'].values
        par3_sys_1 = self.remove_short_phases(self.df['par3_sys_1'].values)
        par3_sys_2 = self.remove_short_phases(self.df['par3_sys_2'].values)
        par8_sys_2 = self.remove_short_phases(self.df['par8_sys_2'].values)
        altitude = self.remove_short_phases(self.altitude)
        norm_altitude = self.min_max_normalize(altitude)
        self.time = np.array(range(len(altitude)))

        orig_flight_phases = self.remove_short_phases(orig_flight_phases)
        new_flight_phases = np.array([0] * len(orig_flight_phases))

        # Preparation for flight phase detection
        indexes = np.array(range(len(new_flight_phases)))

        start_index_pink_descent = len(orig_flight_phases) * 0.75
        end_index_climb_cruise_descent = len(orig_flight_phases) * 0.5
        start_index_climb_cruise_descent = len(orig_flight_phases) * 0.5
        indexes_pink_descent = np.where((orig_flight_phases == 10))[0]

        if len(indexes_pink_descent) > 0:
            start_index_pink_descent = indexes_pink_descent[0]

        indexes_climb_cruise_descent = np.where((orig_flight_phases == 8))[0]
        if len(indexes_climb_cruise_descent) > 0:
            end_index_climb_cruise_descent = indexes_climb_cruise_descent[-1]
            start_index_climb_cruise_descent = indexes_climb_cruise_descent[0]

        mask_taxi_takeoff = indexes < start_index_climb_cruise_descent
        mask_taxi_descent = indexes > end_index_climb_cruise_descent
        mask_high_altitude = (indexes >= start_index_climb_cruise_descent) & (indexes <= end_index_climb_cruise_descent)

        # Controlled turbulence taxi stage
        mask_taxi_controlled_turbulence = mask_taxi_takeoff & (orig_flight_phases == 2) & ((par3_sys_1 == 'ON') & (par3_sys_2 == 'ON'))
        new_flight_phases[mask_taxi_controlled_turbulence] = 2
        indexes_controlled_turbulence = np.where(new_flight_phases == 2)[0]
        if len(indexes_controlled_turbulence) > 0:
            index_controlled_turbulence = indexes_controlled_turbulence[0]
        else:
            index_controlled_turbulence = len(orig_flight_phases)

        dict_phases = {1: 'takeoff_taxi_warmup',
                       3: 'takeoff_taxi_high_turbulence',
                       4: 'takeoff_low_turbulence',
                       12: 'landing_low_turbulence',
                       10: 'landing_high_turbulence',
                       9: 'cruise',
                       8: 'descent_high_turbulence',
                       6: 'descent_low_turbulence',
                       13: 'descent_final_turbulence',
                       7: 'climb',
                       2: 'takeoff_taxi_controlled_turbulence',
                       0: 'Unknown'
        }

        # Warmup taxi stage
        mask_taxi_warmup = mask_taxi_takeoff & (orig_flight_phases == 2) & (par3_sys_1 == 'OFF') &\
                           (par3_sys_2 == 'OFF') & (indexes < index_controlled_turbulence)
        new_flight_phases[mask_taxi_warmup] = 1

        # High turbulence taxi stage
        if len(indexes_controlled_turbulence) <= 0:
            index_controlled_turbulence = 0
            mask_taxi_high_turbulence = mask_taxi_takeoff & (indexes >= index_controlled_turbulence) & ((orig_flight_phases == 3))
        else:
            mask_taxi_high_turbulence = mask_taxi_takeoff & (indexes >= index_controlled_turbulence) \
                                        & ((orig_flight_phases == 3) & ((par3_sys_1 == 'OFF') | (par3_sys_2 == 'OFF')) \
                                           | ((orig_flight_phases == 2) & (par3_sys_1 == 'OFF') & (par3_sys_2 == 'OFF')))
        new_flight_phases[mask_taxi_high_turbulence] = 3

        # Low turbulence taxi stage
        mask_taxi_low_turbulence = mask_taxi_takeoff & (orig_flight_phases == 4)
        new_flight_phases[mask_taxi_low_turbulence] = 4

        # Return taxi stage
        mask_taxi_return = mask_taxi_descent & (orig_flight_phases == 12)
        new_flight_phases[mask_taxi_return] = 12

        # Descent turbulence stage
        mask_taxi_return = mask_taxi_descent & (orig_flight_phases == 10)
        new_flight_phases[mask_taxi_return] = 10

        mask_potential_cruise = mask_high_altitude
        signal = (norm_altitude - par8_sys_2)[mask_potential_cruise]

        algo_c = rpt.KernelCPD(kernel="rbf", min_size=int(len(signal) * 0.1), jump=1).fit(signal)  # written in C

        breakpoints_climb_cruise_descent = algo_c.predict(n_bkps=3)
        start_cruise = breakpoints_climb_cruise_descent[0] + start_index_climb_cruise_descent

        end_turbulent_descent = start_index_climb_cruise_descent + breakpoints_climb_cruise_descent[-2]
        end_cruise = start_index_climb_cruise_descent + breakpoints_climb_cruise_descent[1]

        mask_cruise = (indexes > start_cruise) & (indexes < end_cruise)
        new_flight_phases[mask_cruise] = 9

        mask_turbulent_descent = mask_high_altitude & (indexes > end_cruise) & (indexes < end_turbulent_descent)
        new_flight_phases[mask_turbulent_descent] = 8

        mask_climb = mask_high_altitude & (indexes < start_cruise)
        new_flight_phases[mask_climb] = 7

        mask_descent = mask_high_altitude & (indexes > end_turbulent_descent + 100)
        new_flight_phases[mask_descent] = 6

        mask_green_descent = (indexes > end_turbulent_descent) & (indexes < start_index_pink_descent)
        new_flight_phases[mask_green_descent] = 13

        self.orig_flight_phases = orig_flight_phases
        self.new_flight_phases = new_flight_phases
        self.norm_altitude = norm_altitude
        self.start_cruise = start_cruise
        self.end_cruise = end_cruise
        self.end_turbulent_descent = end_turbulent_descent
        self.dict_phases = dict_phases
        self.plot()
        return self.new_flight_phases, dict_phases

    def plot(self):
        fig, axes = plt.subplots(2,1)
        plt.axvline(self.start_cruise, color='red', label='start cruise')
        plt.axvline(self.end_cruise, color='yellow', label='end cruise')
        plt.axvline(self.end_turbulent_descent, color='black', label='turbulent descent')
        for flight_phases, j in zip([self.orig_flight_phases, self.new_flight_phases], range(2)):
            index_flight_stage = 0
            unique_flight_phases = pd.unique(flight_phases)
            norm_altitude = self.norm_altitude
            for i, flight_phase in zip(range(len(unique_flight_phases)), unique_flight_phases):
                norm_altitude_flight_stage = norm_altitude[flight_phases == flight_phase]
                xx = self.time[flight_phases == flight_phase]
                if j == 1:
                    phase_label = self.dict_phases[flight_phase]
                    axes[j].scatter(xx, norm_altitude_flight_stage, alpha=0.75, color=self.colors[flight_phase], label=phase_label)
                else:
                    axes[j].scatter(xx, norm_altitude_flight_stage, alpha=0.75, color=self.colors[flight_phase])
                plt.title('Flight of aircraft ' + self.aircraft_id)
                axes[j].set_ylabel('Altitude')
                axes[j].grid(True, linestyle='--', linewidth=0.5)
                axes[j].set_xlabel('Time in hours')
                index_flight_stage += 1
            for col_name, m in zip(self.df.columns[:7], range(1, 8)):
                signal = self.remove_short_phases(self.df[col_name].values)
                labels = pd.unique(signal)
                signal = (signal == labels[0])
                axes[j].plot(self.time, signal*(-0.1*m), color=self.colors[m], label=col_name + labels[0])
            for m in range(6, 10):
                signal = self.remove_short_phases(self.min_max_normalize(self.df['par' + str(m) + '_sys_1'].values))
                axes[j].plot(self.time, signal, color=self.colors[m], label=m)
                axes[j].legend()
        plt.show()
