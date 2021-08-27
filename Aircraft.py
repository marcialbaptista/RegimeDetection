import random as rnd
from Flight import Flight
from collections import defaultdict
import numpy as np
import pandas as pd
from Removals import Removals
import matplotlib.pyplot as plt
import pickle
import os
from statsmodels.tsa.stattools import adfuller
from Utils import Utils

class Aircraft:
    """
        A class that is used to represent the sensor data of an
        aircraft

        Attributes
        ----------
        aircraft_id         : str
            the aircraft identifier (PH-01, PH-33, etc.)
        flights_dirs        : list
            list with the locations of the different flights on disk
        debug               : str
            the warning print flag
        error               : int
            the error print flag
        colors               : list
            list of strings indicating colors
        no_flights           : int
            number of flights
        min_flight_size      : int
            flight size threshold limit
        dates                 : dates (list of datetime)
            list of flight dates [sorted]
        taxi_warmup_dict      : dictionary with mean diffs of the different flights
                                for different pairs of sensor parameters
        removals              : Removals
            Removals object
    """
    def __init__(self, aircraft_id, flights_dirs, debug, error):
        """Initializes the aircraft objects

            :param aicraft_id: id of aircraft
            :type: str
            :param flights_dirs: list with the locations of
                                 the different flights on disk
            :type: list of str
            :param debug: the warning print flag
            :type: bool
            :param error: the error print flag
            :type: bool
        """
        self.debug = debug
        self.error = error
        self.aircraft_id = aircraft_id
        self.flights_dirs = flights_dirs
        self.colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k',
                       'Brown', 'ForestGreen', 'pink', 'gray', 'silver',
                       'chartreuse', 'darkturquoise']
        self.min_flight_size = 500
        self.dates = []
        self.sort_flights()
        self.no_flights = len(self.flights_dirs)
        self.taxi_warmup_dict = defaultdict(list)
        self.taxi_warmup_turbulence_dict = defaultdict(list)
        self.taxi_high_turbulence_dict = defaultdict(list)
        self.removals = Removals()


    def plot_random_flight(self):
        """Plots a random flight of the aircraft

            :param: col_index  : index of parameter to plot
            :type: int
        """
        no_flights = len(self.flights_dirs)
        index_flight = int(rnd.random() * no_flights)
        flight_dir = self.flights_dirs[index_flight]
        flight = Flight(self.aircraft_id, flight_dir, self.debug, self.error)
        flight.plot()

    def sort_flights(self):
        """Sorts the flight dirs, dates and filters the short flights
        """
        filename1 = 'data\\aircraft\\flights_' + self.aircraft_id + '.pkl'
        filename2 = 'data\\aircraft\\dates_' + self.aircraft_id + '.pkl'
        if os.path.exists(filename1) and os.path.exists(filename2):
            with open(filename1, 'rb') as f:
                self.flights_dirs = pickle.load(f)
            with open(filename2, 'rb') as f:
                self.dates = pickle.load(f)
                self.dates = pd.to_datetime(self.dates, format='%Y-%m-%d %H:%M:%S')
            return 0

        dates = []
        sizes = []
        for flight_dir in self.flights_dirs:
            flight = Flight(self.aircraft_id, flight_dir, self.debug, self.error)
            date = flight.get_date()
            dates.append(date)
            size = flight.get_size()
            sizes.append(size)
        self.flights_dirs = np.array([x for _, x in sorted(zip(dates, self.flights_dirs))])
        sizes = np.array([x for _, x in sorted(zip(dates, sizes))])
        self.flights_dirs = self.flights_dirs[sizes > self.min_flight_size]
        dates = pd.to_datetime(dates, format='%Y-%m-%d %H:%M:%S')
        self.dates = np.array(sorted(dates))
        self.dates = self.dates[sizes > self.min_flight_size]
        with open(filename1, 'wb') as f:
            pickle.dump(self.flights_dirs, f)
        with open(filename2, 'wb') as f:
            pickle.dump(self.dates, f)
        return 0

    def get_stages(self):
        """Gets a list of the features at different stages of the different aircraft flight
        """
        for stage in ['warmup', 'warmup_turbulence', 'high_turbulence', 'climb', 'cruise', 'descent']:
            result_dict = defaultdict(list)
            for i in range(self.no_flights):
                flight = Flight(self.aircraft_id, self.flights_dirs[i], self.debug, self.error)
                if stage == 'climb':
                    dict_mean = flight.get_climb()
                if stage == 'cruise':
                    dict_mean = flight.get_cruise()
                if stage == 'descent':
                    dict_mean = flight.get_descent()
                if stage == 'warmup':
                    dict_mean = flight.get_taxi_warmup()
                elif stage == 'warmup_turbulence':
                    dict_mean = flight.get_taxi_warmup_turbulence()
                elif stage == 'high_turbulence':
                    dict_mean = flight.get_taxi_high_turbulence()
                for key in dict_mean.keys():
                    result_dict[key].append(dict_mean[key])
                    if False and key == 'par6_sys_1' and stage == 'high_turbulence' and dict_mean[key] > 50:
                        print('Key = %s has mean diff equal to %.2f' % (key, dict_mean[key]))
                        flight.plot()
            if stage == 'warmup':
                self.taxi_warmup_dict = result_dict
            elif stage == 'warmup_turbulence':
                self.taxi_warmup_turbulence_dict = result_dict
            elif stage == 'high_turbulence':
                self.taxi_high_turbulence_dict = result_dict
            elif stage == 'climb':
                self.climb_dict = result_dict
            elif stage == 'cruise':
                self.cruise_dict = result_dict
            elif stage == 'descent':
                self.descent_dict = result_dict
        #self.plot()
        #self.plot_stationarity()


    def plot(self):
        """Plots the aircraft aggregated signals overtime
           (please run self.get_taxi_warmup first)
        """
        param_name = 'Meandiff_' + 'par' + str(6) + '_sys_' + str(1)
        param_name2 = 'Meandiff_' + 'par' + str(7) + '_sys_' + str(1)
        param_name3 = 'Meandiff_' + 'par' + str(8) + '_sys_' + str(1)
        fig, ax = plt.subplots(6, 3, sharex=True, figsize=(20,18))
        ax[0, 0].scatter(self.dates, self.taxi_warmup_dict[param_name], label='Warmup stable')
        ax[0, 1].scatter(self.dates, self.taxi_warmup_dict[param_name2], label='Warmup stable')
        ax[0, 2].scatter(self.dates, self.taxi_warmup_dict[param_name3], label='Warmup stable')

        ax[1, 0].scatter(self.dates, self.taxi_warmup_turbulence_dict[param_name], color='red', marker='.', label='Warmup turbulence')
        ax[1, 1].scatter(self.dates, self.taxi_warmup_turbulence_dict[param_name2], color='red', marker='.',
                         label='Warmup turbulence')
        ax[1, 2].scatter(self.dates, self.taxi_warmup_turbulence_dict[param_name3], color='red', marker='.',
                         label='Warmup turbulence')

        ax[2, 0].scatter(self.dates, self.taxi_high_turbulence_dict[param_name], color='green', marker='.', label='High turbulence')
        ax[2, 1].scatter(self.dates, self.taxi_high_turbulence_dict[param_name2], color='green', marker='.',
                         label='High turbulence')
        ax[2, 2].scatter(self.dates, self.taxi_high_turbulence_dict[param_name3], color='green', marker='.',
                         label='High turbulence')

        ax[3, 0].scatter(self.dates, self.climb_dict[param_name], color='green', marker='.',
                      label='Climb')
        ax[3, 1].scatter(self.dates, self.climb_dict[param_name2], color='green', marker='.',
                      label='Climb')
        ax[3, 2].scatter(self.dates, self.climb_dict[param_name3], color='green', marker='.',
                      label='Climb')

        ax[4, 0].scatter(self.dates, self.cruise_dict[param_name], color='green', marker='.',
                         label='Cruise')
        ax[4, 1].scatter(self.dates, self.cruise_dict[param_name2], color='green', marker='.',
                         label='Cruise')
        ax[4, 2].scatter(self.dates, self.cruise_dict[param_name3], color='green', marker='.',
                         label='Cruise')

        ax[5, 0].scatter(self.dates, self.descent_dict[param_name], color='green', marker='.',
                         label='Descent')
        ax[5, 1].scatter(self.dates, self.descent_dict[param_name2], color='green', marker='.',
                         label='Descent')
        ax[5, 2].scatter(self.dates, self.descent_dict[param_name3], color='green', marker='.',
                         label='Descent')

        index = 0
        for removal in self.removals.get_removals(self.aircraft_id):
            label = removal.to_str()
            for i in range(6):
                for j in range(3):
                    ax[i, j].axvline(removal.date, label=label, color=self.colors[index])
            index += 1
        #ruls, removal_ids = self.removals.get_remaining_useful_life(self.aircraft_id, self.dates)
        #plt.scatter(self.dates[:len(ruls)], removal_ids, label='RUL')
        plt.title('Aircraft ' + self.aircraft_id)
        #plt.legend()
        plt.grid(linestyle='dotted')
        #plt.show()
        #plt.savefig('figs\\flights\\' + self.aircraft_id + 'maxmedian.png')
        plt.show()

    def plot_stationarity(self):
        """Plots stationarity plots on top of sensor parameter
        """
        param_name = 'Meandiff_' + 'par' + str(6) + '_sys_' + str(1)

        fig, ax = plt.subplots(nrows=3, ncols=1)
        dict_features = [self.taxi_warmup_dict, self.taxi_warmup_turbulence_dict, self.taxi_high_turbulence_dict]
        for i in range(len(dict_features)):
            index = 0
            for removal in self.removals.get_removals(self.aircraft_id):
                label = removal.to_str()
                ax[i].axvline(removal.date, label=label, color=self.colors[index])
                if removal.previous_date is None:
                    mask = (self.dates < removal.date)
                else:
                    mask = np.logical_and((self.dates < removal.date), (self.dates >= removal.previous_date))
                time = np.array(self.dates)[mask]
                data = np.array(dict_features[i][param_name])[mask]
                df = pd.DataFrame({'data': data, 'time': time}).set_index('time')
                lag = 10
                previous_date = removal.previous_date
                if previous_date is None:
                    previous_date = pd.to_datetime('2019-01-01', format='%Y-%m-%d')

                if len(df.data.dropna()) > 5:
                    green = False
                    dftest = adfuller(df.data.dropna(), autolag='AIC')
                    for k, v in dftest[4].items():
                        if v < dftest[0]:
                            ax[i].axvspan(previous_date, removal.date, alpha=0.25, color='green', label=str(100 - int(k[:-1])))
                            green = True
                            break
                    if not green:
                        ax[i].axvspan(previous_date, removal.date, alpha=0.25, color='red')
                ax[i].scatter(df.index, df.data, marker='.', color='black')
                moving_average_signal = df.data.rolling(window=lag, min_periods=1).mean()
                exponential_smoothed_signal = Utils.exponential_smoothing(df.data, 0.1)
                ax[i].scatter(df.index, moving_average_signal, color='red', lw=3, marker='.');
                ax[i].scatter(df.index, exponential_smoothed_signal, color='green', marker='.');
                index += 1
            ax[i].set_title('Aircraft ' + self.aircraft_id + ' param ' + param_name + str(i))
            ax[i].legend()
            ax[i].grid(linestyle='dotted')
        plt.show()

    def create_df(self):
        """Gets a list of the features at different stages of the different aircraft flight
        """
        from Flight_Stages_WorkProgress import FlightStages
        for i in range(self.no_flights):
            flight = Flight(self.aircraft_id, self.flights_dirs[i], self.debug, self.error)
            stage_processor = FlightStages(aircraft_id=self.aircraft_id, df=flight.df, debug=True, error=True)
            stage_processor.detect()


    def create_df2(self):
        """Create dataframe with pre-processed features
            of the aircraft

            :return: df: dataframe
            :rtype: pd.DataFrame
        """
        self.get_stages()

        df_dict = defaultdict(list)
        ruls, removal_ids = self.removals.get_remaining_useful_life(self.aircraft_id, self.dates)

        df_dict['RUL'] = ruls
        df_dict['RemovalID'] = removal_ids
        len_ruls = len(ruls)
        alpha = 0.1
        lag = 10
        dict_features = {'taxi_warmup_': self.taxi_warmup_dict,
                         'taxi_warmup_turbulence_': self.taxi_warmup_turbulence_dict,
                         'taxi_high_turbulence_': self.taxi_high_turbulence_dict,
                         'taxi_climb_': self.climb_dict,
                         'taxi_cruise_': self.cruise_dict,
                         'taxi_descent_': self.descent_dict
                         }
        for name, d in dict_features.items():
            for key in self.taxi_warmup_dict.keys():
                signal = np.array(d[key][:len_ruls])
                print(len(d[key]), key, name)
                df_dict[name + str(key)] = signal
                df_dict['MvgAvg_' + name + str(key)] = Utils.moving_average(signal, window_size=lag)
                df_dict['ExpSmooth_' + name + str(key)] = Utils.exponential_smoothing(signal, alpha=alpha)

        df_dict['Date'] = self.dates[:len_ruls]
        df_dict['AircraftID'] = [self.aircraft_id] * len_ruls

        return pd.DataFrame(df_dict)
