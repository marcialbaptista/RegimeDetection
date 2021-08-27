import pickle as pkl
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import random as rnd
from collections import defaultdict
from Utils import Utils
from ACD import ACD
from Aircraft import Aircraft
import pickle
from LSTMMultiClassifier import LSTMMultiClassifier
from RFMultiClassifier import RFMultiClassifer
import os
import pickle as pkl
import os
import pyarrow.parquet as pq
import numpy as np
from collections import defaultdict
from os import listdir
from os.path import isfile, join
import pandas as pd
from pathlib import Path

class Program:
    """
        A class that is used to load the data from a pickle file
        and create aircraft sensor data

        Attributes
        ----------
        filename       : str
            the name of the directory and location where the pickle file is,
            the file indicates where each flight is located on
            the disk
        debug           : str
            the warning print flag
        error           : int
            the error print flag
        model           : MultiClassifier
            the model to run
    """
    def __init__(self, debug, error):
        """Inits the auxiliary units
            :param debug: the warning print flag
            :param error: the error print flag
        """
        self.debug = debug
        self.error = error
        self.aircraft = defaultdict(Aircraft)
        self.model = None
        self.read_dir = 'D:\\Projects\\B737\\01 BLEED\\01 SENSOR\\Data\\'

    def is_created(self):
        return os.path.isfile(self.write_dir + self.filename)

    def create(self, filename):
        flight_dirs = [x[0] for x in os.walk(self.read_dir)]
        flight_filenames = defaultdict(list)
        if self.debug:
            print("[0] - Reading parquet directory:", self.read_dir)
        parquet_index = 1

        for flight_dir in flight_dirs[1:]:

            flight_filename = [f for f in listdir(flight_dir) if isfile(join(flight_dir, f))]

            if self.error and len(flight_filename) > 1:
                print('Error: more than one file per flight in directory', flight_dir)

            flight_filename = flight_dir + '//' + flight_filename[0]

            if self.debug:
                print('[%d] - Reading parquet file: %s' % (parquet_index, flight_filename))

            df = pq.read_table(flight_filename).to_pandas()
            plane_tail = np.unique(df['plane_tail'].values)
            departure_datetime = np.unique(df['departure_datetime'].values)

            from Flight import Flight
            f = Flight(plane_tail, flight_filename, False, False)
            f.plot()

            if len(plane_tail) > 1 and self.error:
                print('Error: more than one plane_tail in parquet', len(plane_tail), plane_tail)

            if len(departure_datetime) > 1 and self.error:
                print('Error: more than one datetime in parquet', len(departure_datetime), departure_datetime)

            flight_filenames[plane_tail[0]].append(flight_filename)
            parquet_index += 1

        with open(filename, 'wb') as handle:
            pkl.dump(flight_filenames, handle, protocol=pkl.HIGHEST_PROTOCOL)

        return 1

    def load(self, filename):
        """Loads the aircraft data into a dictionary
            :param filename: the filename from which
                             to load the aircraft data
        """
        with open(filename, 'rb') as handle:
            flights_dirs = pkl.load(handle)

        print_index = 0
        no_aircraft = len(flights_dirs)
        # iterate over all flights directories
        for aircraft_id in flights_dirs:
            if self.debug:
                progress = int(100 * print_index / no_aircraft)
                print('[%d] - Loading aircraft %s' % (progress, aircraft_id))
                print_index += 1
            flights_aircraft_dirs = flights_dirs[aircraft_id]
            self.aircraft[aircraft_id] = Aircraft(aircraft_id, flights_aircraft_dirs, self.debug, self.error)

    def get_aircraft_ids(self):
        """Gets the list of aircraft IDs

            :returns: a list of strings representing the aircraft ids (PH-01, PH-33, etc.)
            :rtype: list
        """
        return np.unique(list(self.aircraft.keys()))

    def plot_random_flights2(self, no_flights):
        """Plots at each time a random flight from a random aircraft

            :param: col_index  : index of parameter to plot
            :rtype: int
            :param: no_flights : number of flights to randomly plot
            :rtype: int
        """
        aircraft_ids = self.get_aircraft_ids()
        no_aircraft_ids = len(aircraft_ids)
        print_index = 1
        for i in range(no_flights):
            index_aircraft = int(rnd.random() * no_aircraft_ids)
            aircraft_id = aircraft_ids[index_aircraft]
            for _ in range(5):
                print('[%d] - Plotting flight %d from aircraft %s, remaining %d' % (
                print_index, print_index, aircraft_id, no_flights * 5 - print_index))
                self.aircraft[aircraft_id].plot_random_flight()
                print_index += 1

    def plot_random_flights(self, col_index, no_flights):
        """Plots at each time a random flight from a random aircraft

            :param: col_index  : index of parameter to plot
            :rtype: int
            :param: no_flights : number of flights to randomly plot
            :rtype: int
        """
        aircraft_ids = ['PH-1'] #self.get_aircraft_ids()
        no_aircraft_ids = len(aircraft_ids)
        print_index = 1
        for i in range(no_flights):
            index_aircraft = int(rnd.random() * no_aircraft_ids)
            aircraft_id = aircraft_ids[index_aircraft]
            for _ in range(5):
                print('[%d] - Plotting flight %d from aircraft %s, remaining %d' % (
                print_index, print_index, aircraft_id, no_flights * 5 - print_index))
                self.aircraft[aircraft_id].plot_random_flight(col_index=col_index+2)
                print_index += 1

    def create_df(self, filename):
        """Create and save dataframe with program sensor features and rul
        """
        print_index = 1
        dfs = []
        aircraft_ids = self.get_aircraft_ids()
        no_aircraft = len(aircraft_ids)
        for aircraft_id in aircraft_ids:
            progress = int(100 * print_index/no_aircraft)
            print('[{0}%] - Processing aircraft {1}'.format(progress, aircraft_id))
            df = self.aircraft[aircraft_id].create_df()
            dfs.append(df)
            print_index += 1
        df = pd.concat(dfs)
        f = open(filename, "wb")
        pickle.dump(df, f)
        f.close()

    def run_lstm_multi_classification(self, filename):
        self.model = LSTMMultiClassifier(filename=filename)
        self.model.cross_validate()

    def run_rf_multi_classification(self, filename):
        self.model = RFMultiClassifer(filename=filename)
        self.model.cross_validate()
