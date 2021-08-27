import pickle as pkl
import os
import pyarrow.parquet as pq
import numpy as np
from collections import defaultdict
from os import listdir
from os.path import isfile, join
import pandas as pd
from pathlib import Path


class AircraftDirs:

    def __init__(self, debug, error):
        self.read_dir = 'D:\\Projects\\B737\\01 BLEED\\01 SENSOR\\Data\\'
        self.write_dir = 'data\\'
        self.filename = 'flights_dir2.pickle'
        self.debug = debug
        self.error_print = error

    def is_created(self):
        return os.path.isfile(self.write_dir + self.filename)

    def create(self):
        flight_dirs = [x[0] for x in os.walk(self.read_dir)]
        flight_filenames = defaultdict(list)
        if self.debug:
            print("[0] - Reading parquet directory:", self.read_dir)
        parquet_index = 1

        for flight_dir in flight_dirs[1:]:

            flight_filename = [f for f in listdir(flight_dir) if isfile(join(flight_dir, f))]

            if self.error_print and len(flight_filename) > 1:
                print('Error: more than one file per flight in directory', flight_dir)

            flight_filename = flight_dir + '//' + flight_filename[0]

            if self.debug:
                print('[%d] - Reading parquet file: %s' % (parquet_index, flight_filename))

            df = pq.read_table(flight_filename).to_pandas()

            plane_tail = np.unique(df['plane_tail'].values)
            departure_datetime = np.unique(df['departure_datetime'].values)

            if len(plane_tail) > 1 and self.error:
                print('Error: more than one plane_tail in parquet', len(plane_tail), plane_tail)

            if len(departure_datetime) > 1 and self.error:
                print('Error: more than one datetime in parquet', len(departure_datetime), departure_datetime)

            flight_filenames[plane_tail[0]].append(flight_filename)

            new_dir_path = Path(self.write_dir)
            new_dir_path.mkdir(parents=True, exist_ok=True)
            parquet_index += 1

        if self.debug:
            print('[%d] - Writting flights %s' % (parquet_index, self.write_dir + self.filename))

        with open(self.write_dir + self.filename, 'wb') as handle:
            pkl.dump(flight_filenames, handle, protocol=pkl.HIGHEST_PROTOCOL)

        return 1

    def load(self):
        with open(self.write_dir + self.filename, 'rb') as handle:
            self.flights_dirs = pkl.load(handle)

    def find_duplicates(self):
        found_duplicates = False
        for aircraft_id in self.flights_dirs:
            flights_dirs_aircraft = self.flights_dirs[aircraft_id]
            print("Aircraft=%s" % aircraft_id)
            departures = []
            for flight_filename in flights_dirs_aircraft:
                df = pq.read_table(flight_filename).to_pandas()
                departure = df['departure_datetime'].values[0]
                departures.append(departure)

            no_unique_flights = len(np.unique(departures))
            no_actual_flights = len(departures)
            if no_actual_flights != no_unique_flights:
                print('Error in aircraft %s: %d instead of %d files' % (aircraft_id, no_unique_flights, no_actual_flights))
                found_duplicates = True
            print('Finished finding duplicates')
        return found_duplicates