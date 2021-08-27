import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestRegressor
import time
from Model import  Model

class RFModel(Model):

    def cross_validate(self):
        units = np.unique(self.df['AircraftID'])
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
            self.evaluate()
            index_fold += 1
            print('Finished cross validating fold %d %.2f seconds' % (index_fold, (time.time() - start_time)))
        self.result_df = pd.concat(result_dfs)

    def validate_fold(self):
        data = self.df
        test_units = self.test_units

        train_data = data.loc[~data['AircraftID'].isin(test_units), :]
        test_data = data.loc[data['AircraftID'].isin(test_units), :]
        X_train = train_data
        y_train = train_data['RUL']
        X_train = X_train.drop(['RUL', 'AircraftID'], axis=1)

        X_test = test_data
        X_test = X_test.drop(['RUL', 'AircraftID'], axis=1)

        # Instantiate model with 150 decision trees
        model = RandomForestRegressor(n_estimators=150, random_state=42)

        model.fit(X_train, y_train)
        all_predictions_rf = model.predict(X_test)

        test_data.loc[:, 'Predictions'] = all_predictions_rf

        return test_data


