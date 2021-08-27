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


class LSTMMultiClassifier(MultiClassifier):

    def get_row_lags(self, unit_data, i):
        periods = self.LAGS
        ix = [i] + [i - period for period in periods]

        ix = np.array(ix)
        ix[ix < 0] = [0] * np.sum((ix < 0)*1)

        selected_data = unit_data.loc[ix, unit_data.columns != 'RUL'].values
        RUL = unit_data.loc[i, 'RUL']

        if RUL >= 300: labels = [1, 0, 0, 0, 0, 0, 0]
        elif RUL >= 250: labels = [0, 1, 0, 0, 0, 0, 0]
        elif RUL >= 200: labels = [0, 0, 1, 0, 0, 0, 0]
        elif RUL >= 150: labels = [0, 0, 0, 1, 0, 0, 0]
        elif RUL >= 100:  labels = [0, 0, 0, 0, 1, 0, 0]
        elif RUL >= 50:  labels = [0, 0, 0, 0, 0, 1, 0]
        elif RUL >= 0:  labels = [0, 0, 0, 0, 0, 0, 1]

        self.n_classes = len(labels)
        return selected_data, labels

    def get_lagged_unit_data(self, data, unit, removal_ids):
        unit_data = data.loc[removal_ids == unit, :].reset_index(drop=True)

        lagged_unit_data = [self.get_row_lags(unit_data, i)
                            for i in range(0, len(unit_data))]
        return lagged_unit_data

    def get_lagged_data(self, data, removal_ids, aircraft_ids):
        lagged_data = chain(*[self.get_lagged_unit_data(data, unit, removal_ids) for unit in removal_ids.unique()])
        measurement, rul = zip(*list(lagged_data))
        measurement, rul = np.array(measurement), np.array(rul)
        return measurement, rul

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

    def create_model2(self, shape, batch_size, optimizer="rmsprop", kernel_initializer="glorot_uniform", loss_function='kl_divergence', learning_rate=0.001):
        optimizer_dic = {"adam": tf.keras.optimizers.Adam(learning_rate),
                         "nadam": tf.keras.optimizers.Nadam(learning_rate),
                         "adadelta": tf.keras.optimizers.Adadelta(learning_rate),
                         "rmsprop": tf.keras.optimizers.RMSprop(learning_rate),
                         "sgd": tf.keras.optimizers.SGD(learning_rate=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)}
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(512, batch_input_shape=(batch_size, shape[1], shape[2]),
                                kernel_initializer=kernel_initializer, stateful=True, return_sequences=True),
            tf.keras.layers.SimpleRNN(512, kernel_initializer=kernel_initializer, stateful=True, return_sequences=False),
            tf.keras.layers.Dense(self.n_classes, kernel_initializer=kernel_initializer, activation='softmax')
        ])
        model.compile(loss=loss_function, optimizer=optimizer_dic[optimizer], metrics='categorical_accuracy')
        return model

    def create_model(self, neurons, shape, batch_size, optimizer="rmsprop", kernel_initializer="glorot_uniform", loss_function='kl_divergence', learning_rate=0.001):
        optimizer_dic = {"adam": tf.keras.optimizers.Adam(learning_rate),
                         "nadam": tf.keras.optimizers.Nadam(learning_rate),
                         "adadelta": tf.keras.optimizers.Adadelta(learning_rate),
                         "rmsprop": tf.keras.optimizers.RMSprop(learning_rate),
                         "sgd": tf.keras.optimizers.SGD(learning_rate=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)}
        model = tf.keras.Sequential([
            tf.keras.layers.GRU(neurons, batch_input_shape=(batch_size, shape[1], shape[2]),
                                 kernel_initializer=kernel_initializer, stateful=True, return_sequences=True),
            tf.keras.layers.GRU(neurons, kernel_initializer=kernel_initializer, stateful=True, return_sequences=False),
            tf.keras.layers.Dense(neurons, kernel_initializer=kernel_initializer, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(self.n_classes, kernel_initializer=kernel_initializer, activation='softmax')
        ])
        model.compile(loss=loss_function, optimizer=optimizer_dic[optimizer], metrics='categorical_accuracy')
        return model

    def validate_fold(self):
        self.LAGS = 1 + np.array(range(50)) # 30; 50
        data = self.df
        test_units = self.test_units
        train_units = np.unique(self.df['AircraftID'].values[~self.df['AircraftID'].isin(test_units)])
        no_train_units = int(len(train_units) * 0.8)

        train_data = data.loc[data['AircraftID'].isin(train_units[:no_train_units]), :]
        train_data.reset_index(drop=True, inplace=True)
        val_data = data.loc[data['AircraftID'].isin(train_units[no_train_units:]), :]
        val_data.reset_index(drop=True, inplace=True)
        test_data = data.loc[data['AircraftID'].isin(test_units), :]
        test_data.reset_index(drop=True, inplace=True)

        col_to_drop = ['AircraftID', 'Date', 'RemovalID']
        train_data_filter = train_data.drop(col_to_drop, axis=1)
        val_data_filter = val_data.drop(col_to_drop, axis=1)
        test_data_filter = test_data.drop(col_to_drop, axis=1)

        scaler = StandardScaler()
        names_columns = train_data_filter.columns
        train_data_scaled = pd.DataFrame(scaler.fit_transform(train_data_filter), columns=names_columns)
        val_data_scaled = pd.DataFrame(scaler.fit_transform(val_data_filter), columns=names_columns)
        test_data_scaled = pd.DataFrame(scaler.fit_transform(test_data_filter), columns=names_columns)

        train_data_scaled['RUL'] = train_data['RUL'].values
        val_data_scaled['RUL'] = val_data['RUL'].values
        test_data_scaled['RUL'] = test_data['RUL'].values

        train_data_scaled = train_data_scaled.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
        val_data_scaled = val_data_scaled.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
        test_data_scaled = test_data_scaled.replace([np.inf, -np.inf], np.nan).dropna(axis=1)

        train_data_scaled.reset_index(drop=True, inplace=True)
        val_data_scaled.reset_index(drop=True, inplace=True)
        test_data_scaled.reset_index(drop=True, inplace=True)

        x_train, y_train = self.get_lagged_data(train_data_scaled, train_data['RemovalID'], train_data['AircraftID'])
        x_val, y_val = self.get_lagged_data(val_data_scaled, val_data['RemovalID'], val_data['AircraftID'])
        x_test, y_test = self.get_lagged_data(test_data_scaled, test_data['RemovalID'], test_data['AircraftID'])
        x_train = np.nan_to_num(x_train, -1)
        x_test = np.nan_to_num(x_test, -1)
        x_val = np.nan_to_num(x_val, -1)


        #model = self.create_model(x_train.shape[1:], 'adam', 'glorot_uniform', 'kl_divergence', learning_rate=0.0001)
        #model = self.create_model(x_train.shape[1:], 'adam', 'glorot_uniform', 'kl_divergence', learning_rate=1e-6)
        callbacks = [
            EarlyStopping(patience=20, verbose=1),
            #LearningRateScheduler(self.annealing, verbose=1)
        ]

        len(x_train) % 100
        size_x_train = int(len(x_train) - (len(x_train) % 100))
        size_x_val = int(len(x_val) - (len(x_val) % 100))
        batch_size_training = int(math.gcd(size_x_val, size_x_train))
        batch_size_testing = 1

        if False:
            model = self.create_model(x_train.shape[1:], 'adam', 'glorot_uniform', 'kl_divergence', learning_rate=0.001)
            self.history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=1024, epochs=500, verbose=1)
            # self.plot_history()
        if True:
            print('Sizes: ', len(x_train) % 100, len(x_val) % 100, batch_size_testing, len(x_train), len(x_val))
            #model = self.create_model(x_train.shape, batch_size_training, 'adam', 'glorot_normal', 'categorical_crossentropy', learning_rate=0.00001)
            model = self.create_model(512, x_train.shape, batch_size_training, 'adam', 'glorot_normal',
                                      'categorical_crossentropy', learning_rate=0.001) # 10 epochs = very good
            neurons = 512
            model = self.create_model(neurons, x_train.shape, batch_size_training, 'adam', 'glorot_normal',
                                      'categorical_crossentropy', learning_rate=0.001)
            for i in range(10):
                print('Epoch %d' % i)
                self.history = model.fit(x_train[:size_x_train], y_train[:size_x_train], validation_data=(x_val[:size_x_val], y_val[:size_x_val]),
                                         shuffle=False, batch_size=batch_size_training, epochs=1, verbose=1)
                model.reset_states()


        # copy weights
        new_model = self.create_model(neurons, x_train.shape, batch_size_testing,  'adam', 'glorot_uniform', 'categorical_crossentropy', learning_rate=0.001)
        old_weights = model.get_weights()
        new_model.set_weights(old_weights)
        predictions_lstm = new_model.predict(x_test, batch_size=batch_size_testing)
        for i in range(self.n_classes):
            test_data.loc[:, 'Predictions' + str(i)] = predictions_lstm[:, i]

        return test_data

    def annealing(self, epoch):
        lr = 0.01 # 0.01
        annealing_start = 5
        return lr if epoch < annealing_start else lr * np.exp(0.1 * (annealing_start - epoch))

