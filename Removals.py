import pandas as pd
from collections import defaultdict
import numpy as np


class Removals:
    """
            A class that is used to represent the removals of the fleet

            Attributes
            ----------
            filename            : str
                the location of the removals file on disk
            df                  : dataframe
                raw dataframe with the removals
            debug               : str
                the warning print flag
            error               : int
                the error print flag
            removals            : defaultdict(list of Removal)
                dictionary with list of removals per aircraft
    """
    def __init__(self):
        """Inits the removals objects
        """
        self.filename = 'data//removals_wouter2021_2.csv'
        self.df = pd.read_csv(self.filename)
        self.removals = self.load_removals()


    def load_removals(self):
        """Inits the removals dictionary (called by init)

            :return removals: a dictionary with removals info per aircraft
            :type: defaultdict(list of Removal)
        """
        aircraft_ids = np.unique(self.df['AC Reg'].values)
        removals = defaultdict(list)
        removal_index = 1
        for aircraft_id in aircraft_ids:
            previous_removal = None
            aircraft_removals_df = self.df.loc[self.df['AC Reg'] == aircraft_id, :]
            orig_dates = pd.to_datetime(aircraft_removals_df['Rem/Inst Date'], format='%m/%d/%Y')
            sorted_dates = sorted(pd.to_datetime(aircraft_removals_df['Rem/Inst Date'], format='%m/%d/%Y'))
            for date in np.unique(sorted_dates):
                removals_at_date = aircraft_removals_df.loc[orig_dates == date, :]
                removals_at_date = removals_at_date.reset_index(drop=True)
                duplicate = False
                for row in range(len(removals_at_date)):
                    removal = Removal(removals_at_date.iloc[row, :], duplicate, removal_index, previous_removal)
                    if not(removal.root_cause == 'NFF') and not duplicate:
                        removals[aircraft_id].append(removal)
                        removal_index += 1
                        previous_removal = removal
                        duplicate = True
        return removals

    def get_removals(self, aircraft_id):
        """Generator of all removals of an aircraft
            :param aircraft_id: identifier of the aircraft
            :type: str
            :return removal: a removal of the aircraft
            :type: Removal
        """
        no_removals = len(self.removals[aircraft_id])
        for i in range(no_removals):
            yield self.removals[aircraft_id][i]

    def get_previous_unique_removals(self, aircraft_id):
        """Generator of the previous unique removals of an aircraft
        [see below]
        """
        no_removals = len(self.removals[aircraft_id])
        if no_removals == 0:
            return
        yield None
        for i in range(no_removals-1):
            removal = self.removals[aircraft_id][i]
            if not removal.duplicate:
                yield self.removals[aircraft_id][i]

    def get_unique_removals(self, aircraft_id):
        """Generator of all unique removals of an aircraft
            :param aircraft_id: identifier of the aircraft
            :type: str
            :return removal: the first unique removal of the aircraft
            :type: Removal
        """
        no_removals = len(self.removals[aircraft_id])
        for i in range(no_removals):
            removal = self.removals[aircraft_id][i]
            if not removal.duplicate:
                yield self.removals[aircraft_id][i]

    def get_remaining_useful_life(self, aircraft_id, dates):
        """Generator of all unique removals of an aircraft
            :param aircraft_id: identifier of the aircraft
            :type: str
            :param dates: dates to classify
            :type: list of datetimes
            :return ruls: list of remaining useful lives for each date
            :type: list
            :return removal_ids: list of RemovalIDs for each date
            :type: list
        """
        index_removal = 0
        ruls = []
        removal_ids = []
        for removal, previous_removal in zip(self.get_unique_removals(aircraft_id), self.get_previous_unique_removals(aircraft_id)):
            if previous_removal is None:
                mask = (dates <= removal.date)
            else:
                mask = (dates > previous_removal.date) & (dates <= removal.date)
            ruls_removal = (removal.date - dates).total_seconds()[mask] * 1.15741E-5
            ruls.extend(ruls_removal)
            removal_ids.extend([removal.id] * len(ruls_removal))
            index_removal += 1
        return ruls, removal_ids


class Removal:
    """
            A class that is used to represent the removal of an aircraft

            Attributes
            ----------
            duplicate            : bool
                if it happens after another removal in the same date
            date                 : datetime
                date of removal
            tsi_hours            : int
                number of hours since installation
            type_removal         : str
                name of removal type
            position             : int
                position of valve that was removed
            reason               : str
                reason for valve removal
            health_check          : str
                healthcheck
            root_cause          : str
                cause of problem found at shop
            removal_id          : str
                unique id of removal
    """
    def __init__(self, df, duplicate, removal_id, previous_removal):
        """Inits the removal
        """
        self.duplicate = duplicate
        self.date = pd.to_datetime(df['Rem/Inst Date'], format='%m/%d/%Y')
        self.tsi_hours = df['TSI Hours']
        self.type_removal = df['Name']
        self.position = df['Position']
        self.reason = df['Reason']
        self.health_check = df['health check or specifically requested PCCV replacement']
        self.root_cause = df['Shop Findings']
        self.id = removal_id
        if previous_removal is None:
            self.previous_date = None
        else:
            self.previous_date = previous_removal.date

    def to_str(self):
        label = str(self.type_removal) + ' ' + str(self.reason) + ' ' + str(self.position) + ' ' + \
                str(self.root_cause) + ' ' + str(self.health_check)
        return label
