import pickle as pkl
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import random as rnd
from collections import defaultdict
from Utils import Utils
from ACD import ACD
import pickle


class DataLoader:
    """
        A class that is used to load the data from a pickle file
        and create aircraft sensor data

        Attributes
        ----------
        read_dir        : str
            the name of the directory where the pickle file is,
            the file indicates where each flight is located on
            the disk
        filename        : str
            the name of the pickle file
        debug           : str
            the warning print flag
        error_print     : int
            the error print flag


        Methods
        -------
        says(sound=None)
            Prints the animals name and what sound it makes
    """
    def __init__(self, debug, error):
        self.read_dir = 'data\\'
        self.filename = 'flights_dir.pickle'
        self.debug = debug
        self.error_print = error
        self.aircraft = self.load_aircraft()
        self.aircraft_ids = list(self.aircraft.keys())

    def load(self):
        with open(self.read_dir + self.filename, 'rb') as handle:
            self.flights_dirs = pkl.load(handle)

    def load_aircraft(self):
        self.load()
        print_index = 0
        aircraft = defaultdict(AircraftData)
        # iterate over all flights directories
        for aircraft_id in self.flights_dirs:
            if self.debug:
                print('[%d] - Processing aircraft %s' % (print_index, aircraft_id))
                print_index += 1
            flights_dirs_aircraft = self.flights_dirs[aircraft_id]
            aircraft[aircraft_id] = AircraftData(self.debug, self.error_print, flights_dirs_aircraft, aircraft_id)
        return aircraft

    def plot_flights(self, percentage_flights_per_aircraft_plot, col_index):
        total_no_aircraft = len(self.aircraft_ids)
        for _ in range(total_no_aircraft):
            index_aicraft = int(rnd.random() * total_no_aircraft)
            aircraft_id = self.aircraft_ids[index_aicraft]
            self.aircraft[aircraft_id].plot_flights(percentage_flights_per_aircraft_plot, col_index=col_index)

    def plot_histogram_len_flights(self):
        total_no_aircraft = len(self.aircraft_ids)
        len_flights = []
        print_index = 0
        for i in range(total_no_aircraft):
            aircraft_id = self.aircraft_ids[i]
            lens_aircraft = self.aircraft[aircraft_id].calc_len()
            len_flights.extend(lens_aircraft)
            print('[{0} {1} {2}%] - Processing flight of aircraft {3}'.format(aircraft_id, print_index, round(
                print_index / total_no_aircraft * 100, 0), aircraft_id))
            print_index+= 1
        plt.hist(len_flights, color='red', bins=100)
        plt.show()

    def plot_aircraft(self, save_plot, col_index):
        total_no_aircraft = len(self.aircraft_ids)
        print_index = 0
        for index_aicraft in range(total_no_aircraft):
            aircraft_id = self.aircraft_ids[index_aicraft]
            print('[{0} {1} {2}%] - Processing aircraft {3}'.format(aircraft_id, print_index, round(
                print_index / total_no_aircraft * 100, 0), aircraft_id))
            self.aircraft[aircraft_id].plot_aircraft(save_plot=save_plot, col_index=col_index)
            print_index += 1

    def plot_taxi_aircraft(self, col_index):
        total_no_aircraft = len(self.aircraft_ids)
        print_index = 0
        for index_aicraft in range(total_no_aircraft):
            aircraft_id = self.aircraft_ids[index_aicraft]
            print('[{0} {1} {2}%] - Processing aircraft {3}'.format(aircraft_id, print_index, round(
                print_index / total_no_aircraft * 100, 0), aircraft_id))
            self.aircraft[aircraft_id].plot_taxi_aircraft(col_index=col_index)
            print_index += 1

    def find_outliers(self, col_index):
        total_no_aircraft = len(self.aircraft_ids)
        print_index = 0
        for index_aicraft in range(total_no_aircraft):
            aircraft_id = self.aircraft_ids[index_aicraft]
            print('[{0} {1} {2}%] - Finding outliers in aircraft {3}'.format(aircraft_id, print_index, round(
                print_index / total_no_aircraft * 100, 0), aircraft_id))
            self.aircraft[aircraft_id].find_outliers(col_index)
            print_index += 1

    def create_taxi_df(self):
        total_no_aircraft = len(self.aircraft_ids)
        print_index = 0
        dfs = []
        for index_aicraft in range(total_no_aircraft):
            aircraft_id = self.aircraft_ids[index_aicraft]
            print('[{0} {1} {2}%] - Processing aircraft {3}'.format(aircraft_id, print_index, round(
                print_index / total_no_aircraft * 100, 0), aircraft_id))
            df = self.aircraft[aircraft_id].create_taxi_df(col_index=8)
            dfs.append(df)
            print_index += 1
        self.df = pd.concat(dfs)

    def save_taxi_df(self):
        f = open("data//data.pickle", "wb")
        pickle.dump(self.df, f)
        f.close()


class AircraftData:

    def __init__(self, debug, error, flights_dirs_aircraft, aircraft_id):
        self.debug = debug
        self.error_print = error
        self.aircraft_id = aircraft_id
        self.flights = self.read_flights(flights_dirs_aircraft, aircraft_id)
        self.colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen', 'pink', 'gray', 'silver',
                       'chartreuse', 'darkturquoise']



    def read_flights(self, flights_dirs_aircraft, aircraft_id):
        # iterate over all flights of aircraft
        print_index = 0
        flights = []
        total_no_flights = len(flights_dirs_aircraft)
        for flight_dir in flights_dirs_aircraft:
            if self.debug:
                print('[{0} {1} {2}%] - Processing flight of aircraft {3}'.format(aircraft_id, print_index, round(print_index/total_no_flights * 100,0), aircraft_id))
                print_index += 1
            flights.append(FlightData(self.debug, self.error_print, flight_dir, aircraft_id))
        return flights

    def plot_flights(self, percentage_flights_plot, col_index):
        total_no_flights = len(self.flights)
        no_flights_plot = int(len(self.flights)*percentage_flights_plot)
        for _ in range(no_flights_plot):
            index_flight = int(rnd.random() * total_no_flights)
            self.flights[index_flight].plot_flight(col_index=col_index)

    def find_outliers(self, col_index):
        total_no_flights = len(self.flights)
        for i in range(total_no_flights):
            self.flights[i].find_outliers(col_index)

    def create_taxi_df(self, col_index):
        total_no_flights = len(self.flights)
        vals, vals2, dates = [], [], []

        for i in range(total_no_flights):
            val, val2, date, _ = self.flights[i].get_mean_taxi(col_index=col_index)
            vals.append(val)
            vals2.append(val2)
            dates.append(date)

        vals = np.array([x for _, x in sorted(zip(dates, vals))])
        vals2 = np.array([x for _, x in sorted(zip(dates, vals2))])
        dates = np.array(sorted(dates))

        type_removals, average_tsis_removals_dic, removals_aircraft_dic = Utils.process_removals(self.aircraft_id)
        index_removal = 0
        prev_removal_date = None
        RULs, removal_ids = [], []
        for removal_date in np.unique(removals_aircraft_dic['Date']):
            removal_date = pd.to_datetime(removal_date)
            if prev_removal_date is None:
                RULs_removal = removal_date - dates[dates <= removal_date]
                RULs.extend(RULs_removal)
                removal_ids.extend([index_removal] * len(RULs_removal))
            else:
                RULs_removal = removal_date - dates[(dates > prev_removal_date) & (dates <= removal_date)]
                RULs.extend(RULs_removal)
                removal_ids.extend([index_removal] * len(RULs_removal))
            prev_removal_date = removal_date
            index_removal += 1
        return pd.DataFrame({'RUL':RULs, 'Taxi_Mean': vals[:len(RULs)], 'Taxi_Mean_Varies': vals2[:len(RULs)], 'RemovalID': removal_ids[:len(RULs)],
                             'AircraftID': [self.aircraft_id] * len(RULs)})





    def plot_taxi_aircraft(self, col_index):
        total_no_flights = len(self.flights)
        vals, len_flights, dates = [], [], []

        for i in range(total_no_flights):
            val, date, len_flight = self.flights[i].get_mean_taxi(col_index=col_index)
            vals.append(val)
            len_flights.append(len_flight)
            dates.append(date)

        mean_vals = np.nanmean(vals)
        std_vals = np.nanstd(vals)
        vals = [x if (x < mean_vals + 1.5 * std_vals) & (x > mean_vals - 1.5 * std_vals) else np.nan for x in vals]

        vals = [x for _, x in sorted(zip(dates, vals))]
        len_flights = [x for _, x in sorted(zip(dates, len_flights))]
        dates = np.array(sorted(dates))
        vals, dates, len_flights = np.array(vals), np.array(dates), np.array(len_flights)

        type_removals, average_tsis_removals_dic, removals_aircraft_dic = Utils.process_removals(self.aircraft_id)
        acd_signal = []
        prev_removal_date = None
        removals_unique_dates = list(np.unique(removals_aircraft_dic['Date']))
        if len(removals_unique_dates) <= 0:
            return 0
        removals_unique_dates.append(None)
        for removal_date in removals_unique_dates:
            removal_date = pd.to_datetime(removal_date)
            if prev_removal_date is None:
                vals_removal = np.array(vals)[dates <= removal_date]
                dates_removal = dates[dates <= removal_date]
            elif removal_date is None:
                vals_removal = np.array(vals)[dates > prev_removal_date]
                dates_removal = dates[dates > prev_removal_date]
            else:
                vals_removal = np.array(vals)[(dates > prev_removal_date) & (dates <= removal_date)]
                dates_removal = dates[(dates > prev_removal_date) & (dates <= removal_date)]
            prev_removal_date = removal_date
            vals_removal2 = vals_removal[np.isfinite(vals_removal)]
            vals_removal2 = list(vals_removal2)
            print(len(vals_removal2))
            if len(vals_removal2) > 1:
                acd_signal_removal = list(ACD.increase_monotonicity(vals_removal2))
                acd_signal.extend(acd_signal_removal)
            else:
                acd_signal_removal = vals_removal2
                acd_signal.extend(acd_signal_removal)
            if False:
                plt.scatter(dates_removal[np.isfinite(vals_removal)], vals_removal2)
                plt.plot(dates_removal[np.isfinite(vals_removal)], acd_signal_removal)
                plt.show()

        print(len(np.array(vals)[np.isfinite(vals)]), len(acd_signal), len(vals))

        monotone_signal = []
        j = 0
        for i in range(len(dates)):
            if np.isnan(vals[i]):
                monotone_signal.append(np.nan)
            else:
                monotone_signal.append(acd_signal[j])
                j += 1

        colors_flights = ['black' if i >= 1000 else self.colors[int(i / 100)] for i in len_flights]
        plt.subplots(figsize=(15, 10))
        plt.scatter(dates, vals, c=colors_flights)
        plt.plot(dates, monotone_signal, marker='.' , c='red', lw=6)
        plt.title('Flights of aircraft %s col_index=%d' % (self.aircraft_id, col_index))

        type_removals, average_tsis_removals_dic, removals_aircraft_dic = Utils.process_removals(self.aircraft_id)
        index_removal = 0
        plt.grid(True, linestyle='--', linewidth=0.5)
        for removal_date in removals_aircraft_dic['Date']:
            reason = str(removals_aircraft_dic['Reason'][index_removal])
            name = str(removals_aircraft_dic['Type Removal'][index_removal])
            position = removals_aircraft_dic['Position'][index_removal]
            import datetime as dt
            removal_date = pd.to_datetime(removal_date)
            plt.axvline(removal_date + dt.timedelta(days=index_removal),
                        label=name + ' ' + reason + ' ' + str(position), color=self.colors[index_removal])
            index_removal += 1

        plt.legend()
        plt.savefig(
            'figs//exploratory//taxi_aircraft//' + str(col_index) + '//flights_aircraft_' + self.aircraft_id + '_norm.png')
        plt.close()

    def plot_aircraft(self, save_plot, col_index):
        total_no_flights = len(self.flights)
        vals, len_flights, dates = [], [], []
        for i in range(total_no_flights):
            val, len_flight, date = self.flights[i].get_mean(col_index=col_index)
            vals.append(val)
            len_flights.append(len_flight)
            dates.append(date)
        colors_flights = ['black' if i >= 1000 else self.colors[int(i / 100)] for i in len_flights]
        plt.subplots(figsize=(15, 10))
        plt.scatter(dates, vals, c=colors_flights)
        plt.title('Flights of aircraft %s col_index=%d' % (self.aircraft_id, col_index))
        if save_plot:
            plt.savefig('figs//exploratory//aircraft//'+str(col_index)+'//flights_aircraft_' + self.aircraft_id + '.png')
            plt.close()
        else:
            plt.show()

    def calc_len(self):
        total_no_flights = len(self.flights)
        lens_aircraft = []
        for i in range(total_no_flights):
            len_flight = self.flights[i].calc_len()
            lens_aircraft.append(len_flight)
        return lens_aircraft


class FlightData:

    def __init__(self, debug, error, flights_dirs_aircraft, aircraft_id):
        self.colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen', 'pink', 'gray', 'silver',
                       'chartreuse', 'darkturquoise']
        self.debug = debug
        self.error_print = error
        self.aircraft_id = aircraft_id
        self.filename = flights_dirs_aircraft
        self.dataframe = None
        #self.dataframe = self.load_flight(flights_dirs_aircraft)
        self.len = -1
        self.short_limit = 1000
        self.outliers = self.get_outliers_thresholds()

    def get_outliers_thresholds(self):
        outliers = defaultdict(list)
        outliers[8] = [-10, 50]
        outliers[10] = [50, 200]
        outliers[12] = [10, 100]
        outliers[14] = [0, 500]
        return outliers

    def get_taxi_stage(self, dataframe):
        mask = np.logical_or((dataframe['par3_sys_1'].values == 'OFF'), (dataframe['par3_sys_2'].values == 'OFF'))
        mask2 = np.logical_and((dataframe['par5_sys_gen'].values == 'OPEN'), (dataframe['par4_sys_2'].values == 'ON'))
        mask3 = np.logical_and((dataframe['par4_sys_1'].values == 'ON'), (dataframe['par2_sys_1'].values == 'OFF'))
        mask4 = np.logical_and((dataframe['par2_sys_2'].values == 'OFF'), (dataframe['par1_sys_gen'].values == 'OFF'))
        mask5 = mask & mask2 & mask3 & mask4
        par6_sys_1 = dataframe['par6_sys_1'].values
        par6_sys_2 = dataframe['par6_sys_2'].values
        par6_sys_1_taxi = dataframe.loc[mask5 == True, 'par6_sys_1'].values
        par6_sys_2_taxi = dataframe.loc[mask5 == True, 'par6_sys_2'].values
        len_taxi = len(par6_sys_1_taxi)
        center_taxi_index = int(len(par6_sys_1_taxi)*0.5)
        found_yellow, found_index = True, -1
        stage_colors = [0] * len(mask5)
        if len_taxi <= 5:
            return np.array(stage_colors)

        alpha = 0.15
        low_index = max(0, int(center_taxi_index * (1 - alpha)))
        high_index = min(len_taxi, int(center_taxi_index * (1 + alpha)))

        #print(low_index, high_index, par6_sys_1_taxi)

        center_max_value = np.max(par6_sys_1_taxi[low_index:high_index])*1.2
        center_max_value2 = np.max(par6_sys_2_taxi[low_index:high_index])*1.2
        if center_max_value > 5 or center_max_value2 > 5:
            return np.array(stage_colors)

        for val, i in zip(mask5, range(len(mask5))):
            if val and not (found_index != -1 and i > found_index+1):
                stage_colors[i] = 1
                if par6_sys_1[i] > center_max_value and par6_sys_2[i] > center_max_value2:
                    if i < center_taxi_index:
                        stage_colors[i] = 2
                    else: stage_colors[i] = 3
                found_index = i
            elif not val and found_index != -1 and i > found_index+1:
                break
        return np.array(stage_colors)

    def load_flight(self, flight_dir):
        orig_df = pd.read_parquet(flight_dir)
        orig_df.sort_values(by="row_number", inplace=True)
        orig_df.reset_index(drop=True, inplace=True)
        return orig_df

    def normalize(self, param, range_param):
        param = (param - np.min(param)) / (np.max(param) - np.min(param))
        param = param * range_param
        return param

    def plot_flight(self, col_index):
        dataframe = self.load_flight(self.filename)
        dataframe = dataframe
        time = dataframe['row_number'].values
        col = dataframe.columns[col_index]
        col2 = dataframe.columns[col_index+1]
        param = dataframe[col]
        param2 = dataframe[col2]
        for i in range(7):
            col_binary = dataframe.columns[i]
            range_param = np.nanmax(param) - np.nanmin(param)
            binary_values = dataframe[col_binary].values
            #if binary_values[0] == np.nan and  not (binary_values is None) and not ('ON' in binary_values):
            #    return 0
            param_binary = (dataframe[col_binary].values == 'ON')*np.max(param) * (-0.1*i)
            plt.plot(time, param_binary, color=self.colors[i], label=col_binary, lw=3)
        i = i+1
        col_binary = dataframe.columns[i]
        param_binary = (dataframe[col_binary].values == 'OPEN') * np.max(param) * (-0.1 * (i))
        plt.plot(time, param_binary, color=self.colors[i], label=col_binary, lw=3)
        plt.plot(time, param, color='purple', label=col)
        plt.plot(time, param2*0.5, color='black', label=col2)
        stages_colors = self.get_taxi_stage(dataframe)
        param_altitude = dataframe['pressure_altitude'].values
        param_altitude = self.normalize(param_altitude, range_param)
        for stage_color in np.unique(stages_colors):
            altitude_stage = param_altitude[stages_colors == stage_color]
            time_stage = time[stages_colors == stage_color]
            plt.plot(time_stage, altitude_stage, c=self.colors[stage_color])
        plt.title("Aircraft=%s Param=%s %s %s " % (self.aircraft_id, col, col2,  col_binary))
        plt.tight_layout()
        plt.legend()
        plt.show()

        stage_color = 1
        param = dataframe['par6_sys_1'].values[stages_colors == stage_color]
        param2 = dataframe['par6_sys_2'].values[stages_colors == stage_color]
        time_stage = time[stages_colors == stage_color]
        plt.plot(time_stage, param, c='purple')
        plt.plot(time_stage, param2, c='black')
        plt.show()

    def save_flight(self, dataframe, col_index, color, folder):
        matplotlib.use('Agg')
        time = dataframe['row_number']
        date = dataframe['departure_datetime'][0]
        col = dataframe.columns[col_index]
        param = dataframe[col]
        param2 = (dataframe['pressure_altitude'] - np.min(param)) / (np.max(param) - np.min(param))
        plt.plot(time, param, color=color, marker='.', label=col)
        plt.plot(time, param2, color='red', marker='.', label='pressure_altitude')
        plt.title("Aircraft=%s Param=%s " % (self.aircraft_id, col))
        plt.tight_layout()
        plt.legend()
        filename = 'aircraft_' + self.aircraft_id + str(date).replace(':','_').replace(' ', '_') + '_param_' + col + '.png'
        plt.savefig(folder + filename)
        plt.close()

    def calc_len(self, dataframe):
        return len(dataframe)

    def process_flight_signal(self, dataframe, col_index):
        outlier_min = self.outliers[col_index][0]
        outlier_max = self.outliers[col_index][1]
        col = dataframe.columns[col_index]
        mask = (dataframe[col] >= outlier_min) & (dataframe[col] <= outlier_max)
        dataframe = dataframe.loc[mask, :]
        dataframe.sort_values(by="row_number", inplace=True)
        dataframe['row_number'] = 1 + np.array(range(len(dataframe)))  # reassign the row numbers
        dataframe.reset_index(drop=True, inplace=True)
        return dataframe

    def is_negative_outlier(self, dataframe, col_index):
        col = dataframe.columns[col_index]
        param = dataframe[col]
        outlier_min = self.outliers[col_index][0]
        if np.min(param) < outlier_min and np.max(param) < outlier_min:
            return True
        return False

    def is_bad_trajectory(self, dataframe, col_index):
        col = dataframe.columns[col_index]
        param = dataframe[col]
        mean_param = np.mean(param)
        self.mean = mean_param
        outlier_min = self.outliers[col_index][0]
        outlier_max = self.outliers[col_index][1]
        if mean_param > outlier_max or mean_param < outlier_min:
            return True
        return False

    def is_short(self, dataframe):
        col_index = 8
        if self.len == -1:
            col = dataframe.columns[col_index]
            param = dataframe[col]
            self.len = len(param)
        if self.len < self.short_limit:
            return True
        return False

    def color_len(self, dataframe):
        if self.is_short(dataframe):
            index_len = int(self.len / 100)
            return self.colors[index_len]
        return 'green'

    def find_outliers(self, col_index):
        dataframe = self.load_flight(self.filename)
        if self.is_short(dataframe):
            folder = 'figs\\exploratory\\flights\\' + str(col_index) + '\\' + 'short\\'
            color_len = self.color_len(dataframe)
            self.save_flight(dataframe, col_index, color=color_len, folder=folder)
        elif self.is_negative_outlier(dataframe, col_index):
            folder = 'figs\\exploratory\\flights\\' + str(col_index) + '\\' + 'outliers\\'
            self.save_flight(dataframe, col_index, color='blue', folder=folder)
        elif self.is_bad_trajectory(dataframe, col_index):
            folder = 'figs\\exploratory\\flights\\' + str(col_index) + '\\' + 'bad_trajectories\\'
            self.save_flight(dataframe, col_index, color='orange', folder=folder)
            # new_dataframe = self.process_flight_signal(dataframe, col_index)
            #self.plot_flight(new_dataframe, col_index, color='blue')

    def get_taxi(self, col_index):
        dataframe = pd.read_parquet(self.filename)
        col = dataframe.columns[col_index]
        param = dataframe[col]
        date = dataframe['departure_datetime']
        date = pd.to_datetime(date[0], format='%Y-%m-%d %H:%M:%S')
        len_flight = len(param)
        mean_param = np.mean(param)
        return mean_param, len_flight, date

    def get_mean_taxi(self, col_index):
        dataframe = self.load_flight(self.filename)
        taxi_stages = self.get_taxi_stage(dataframe)
        col = dataframe.columns[col_index]
        col2 = dataframe.columns[col_index+1]
        all_param = dataframe[col].values
        all_param2 = dataframe[col2].values
        param = dataframe.loc[taxi_stages == 1, col].values
        param2 = dataframe.loc[taxi_stages == 1, col2].values
        dates = dataframe['departure_datetime']
        dates = pd.to_datetime(dates, format='%Y-%m-%d %H:%M:%S')
        date = dates[0]
        len_flight = len(dates)
        if len_flight < self.short_limit or len(param) <= 0:
            return np.nan, np.nan, date, len_flight
        #param = Utils.normalize(param)
        #param = Utils.normalize(param2)

        #val = feature_calculators.binned_entropy(param, len(param)) / len(param)
        if len(np.greater_equal(all_param, all_param2)) > len(all_param) * 0.6:
            diff = param - param2
        else:
            diff = param2 - param

        val = np.mean(param - param2)
        val2 = np.mean(diff)

        if val < -1 or val > 1:
            return np.nan, np.nan, date, len_flight

        return val, val2,  date, len_flight

