import pandas as pd
import numpy as np
import random
import time
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from itertools import chain
from MultiClassifier import MultiClassifier
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from ACD import ACD
import matplotlib.pyplot as plt
import math


class RFMultiClassifer(MultiClassifier):

    def cross_validate(self):
        units = np.unique(self.df['AircraftID'].values)
        result_dfs = []
        random.Random(5).shuffle(units)
        number_units = len(units)
        index_fold = 1

        for test_units_index in range(0, number_units, 7):
            print('Starting to cross validating fold %d' % (index_fold))
            start_time = time.time()
            self.test_units = units[test_units_index:test_units_index + 7]
            result_df = self.validate_fold()
            result_dfs.append(result_df)
            self.result_df = pd.concat(result_dfs)
            index_fold += 1
            self.evaluate()
            print('Finished cross validating fold %d %.2f seconds' % (index_fold, (time.time() - start_time)))
        self.result_df = pd.concat(result_dfs)

    def validate_fold(self):
        data = self.df
        test_units = self.test_units
        train_units = np.unique(self.df['AircraftID'].values[~self.df['AircraftID'].isin(test_units)])
        no_train_units = int(len(train_units))

        train_data = data.loc[data['AircraftID'].isin(train_units[:no_train_units]), :]
        train_data.reset_index(drop=True, inplace=True)
        print(train_data.index)
        test_data = data.loc[data['AircraftID'].isin(test_units), :]
        test_data.reset_index(drop=True, inplace=True)

        col_to_drop = ['AircraftID', 'Date', 'RemovalID']
        train_data_filter = train_data.drop(col_to_drop, axis=1)
        test_data_filter = test_data.drop(col_to_drop, axis=1)

        y_train = self.construct_labels(train_data_filter['RUL'])
        y_test = self.construct_labels(test_data_filter['RUL'])

        x_train = train_data_filter.drop(['RUL'], axis=1)
        x_test = test_data_filter.drop(['RUL'], axis=1)

        # Fitting Random Forest Classification to the Training set
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(n_estimators=150, criterion='entropy', random_state=42)
        classifier.fit(x_train, y_train)

        predictions_rf = classifier.predict_proba(np.float32(x_test))

        print(np.array(predictions_rf)[0,:])
        for i in range(self.n_classes):
            test_data.loc[:, 'Predictions' + str(i)] = np.array(predictions_rf)[i,:][:,1]
        return test_data

    def construct_labels(self, r):
        res = []
        for RUL in r:
            if RUL >= 300:
                labels = [1, 0, 0, 0, 0, 0, 0]
            elif RUL >= 250:
                labels = [0, 1, 0, 0, 0, 0, 0]
            elif RUL >= 200:
                labels = [0, 0, 1, 0, 0, 0, 0]
            elif RUL >= 150:
                labels = [0, 0, 0, 1, 0, 0, 0]
            elif RUL >= 100:
                labels = [0, 0, 0, 0, 1, 0, 0]
            elif RUL >= 50:
                labels = [0, 0, 0, 0, 0, 1, 0]
            elif RUL >= 0:
                labels = [0, 0, 0, 0, 0, 0, 1]
            res.append(labels)
        self.n_classes = len(labels)
        return res


