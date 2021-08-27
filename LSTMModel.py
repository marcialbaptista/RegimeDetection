import pandas as pd
import numpy as np
import random
import time
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from itertools import chain
from Model import Model
import math
from collections import defaultdict
from tsaug import TimeWarp, Crop, Quantize, Drift, Reverse


class LSTMModelClassification(Model):

    def get_row_lags(self, unit_data, i):
        periods = self.LAGS

        ix = [i] + [i - period for period in periods]

        if np.sum(np.array(ix) < 0) > 0:
            ix = [i] + ([i] * len(periods))

        selected_data = unit_data.loc[ix, unit_data.columns != 'RUL']
        selected_data = selected_data.loc[:, selected_data.columns != 'Unnamed: 0'].values

        RUL = unit_data.loc[i, 'RUL']
        RUL = np.array([RUL >= 300 ,RUL in range(250, 300), RUL in range(200, 250), RUL in range(150, 200), RUL in range(100, 150), RUL in range(50, 100), RUL in range(20, 50), RUL in range(20)]) * 1
        return selected_data, RUL

    def get_lagged_unit_data(self, data, unit):
        unit_data = data[data['RemovalID'] == unit].reset_index(drop=True)

        lagged_unit_data = [self.get_row_lags(unit_data, i)
                            for i in range(0, len(unit_data))]

        return lagged_unit_data

    def get_lagged_data(self, data):
        lagged_data = chain(*[self.get_lagged_unit_data(data, unit)
                              for unit in data['RemovalID'].unique()])

        measurement, rul = zip(*list(lagged_data))
        measurement, rul = np.array(measurement), np.array(rul)
        return measurement, rul

    def cross_validate(self):
        units = np.unique(self.df['AircraftID'])
        result_dfs = []
        random.Random(5).shuffle(units)
        number_units = len(units)
        index_fold = 1
        self.df['Time'] = 0
        good_removals = []
        removal_ids = np.unique(self.df['RemovalID'].values)
        for removal_id in removal_ids:
            RUL_removal = self.df.loc[self.df['RemovalID'] == removal_id, 'RUL'].values
            self.df.loc[self.df['RemovalID'] == removal_id, 'Time'] = np.max(RUL_removal) - RUL_removal

        for test_units_index in range(0, number_units, 7):
            print('Starting to cross validating fold %d' % (index_fold))
            start_time = time.time()
            self.test_units = units[test_units_index:test_units_index + 7]
            result_df = self.validate_fold()
            result_dfs.append(result_df)
            self.result_df = pd.concat(result_dfs)
            index_fold += 1
            self.evaluate_multiclassification()
            print('Finished cross validating fold %d %.2f seconds' % (index_fold, (time.time() - start_time)))
        self.result_df = pd.concat(result_dfs)

    def create_model(self, shape, optimizer="rmsprop", kernel_initializer="glorot_uniform", loss_function='kl_divergence', learning_rate=0.001, l1l2=0.9, dropout_rate=0.5):
        optimizer_dic = {"adam": tf.keras.optimizers.Adam(learning_rate),
                         "adadelta": tf.keras.optimizers.Adadelta(learning_rate),
                         "rmsprop": tf.keras.optimizers.RMSprop(learning_rate),
                         "sgd": tf.keras.optimizers.SGD(learning_rate=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)}
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(16, input_shape=shape,
                                 #kernel_regularizer = tf.keras.regularizers.L1L2(l1l2, l1l2),
                                 return_sequences=False),
            # 128 neurons give best performance
            #tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(
                8, kernel_initializer=kernel_initializer, activation='softmax')
        ])

        model.compile(
            loss=loss_function, optimizer=optimizer_dic[optimizer], metrics='categorical_accuracy')
        return model

    def create_model_advanced(self, shape, optimizer="rmsprop", kernel_initializer="glorot_uniform", loss_function='kl_divergence', learning_rate=0.001, l1l2=0.9, dropout_rate=0.5):
        optimizer_dic = {"adam": tf.keras.optimizers.Adam(learning_rate),
                         "adadelta": tf.keras.optimizers.Adadelta(learning_rate),
                         "rmsprop": tf.keras.optimizers.RMSprop(learning_rate),
                         "sgd": tf.keras.optimizers.SGD(learning_rate=learning_rate, decay=1e-6, momentum=0.9,
                                                        nesterov=True)}
        model = tf.keras.Sequential([
            (tf.keras.layers.LSTM(12, input_shape=shape, kernel_regularizer = tf.keras.regularizers.L1L2(l1l2, l1l2), return_sequences=True)),
            tf.keras.layers.Dropout(0.9),
            (tf.keras.layers.GRU(12, kernel_regularizer = tf.keras.regularizers.L1L2(l1l2, l1l2), return_sequences=False)),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(6, kernel_regularizer = tf.keras.regularizers.L1L2(l1l2, l1l2)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(
                8,
                # The predicted deltas should start small
                # So initialize the output layer with zeros
                kernel_initializer=tf.initializers.zeros(), activation='softmax')
        ])
        model.compile(loss=loss_function, optimizer=optimizer_dic[optimizer], metrics='categorical_accuracy')
        return model

    def validate_fold(self):
        self.LAGS = 1 + np.array(range(10)) # 30
        data = self.df
        test_units = self.test_units
        train_units = np.unique(self.df['AircraftID'].values[~self.df['AircraftID'].isin(test_units)])
        no_train_units = int(len(train_units) * 0.8)

        train_data = data.loc[data['AircraftID'].isin(train_units[:no_train_units]), :]
        val_data = data.loc[data['AircraftID'].isin(train_units[no_train_units:]), :]
        test_data = data.loc[data['AircraftID'].isin(test_units), :]

        train_data_orig = train_data.drop(['AircraftID'], axis=1)
        val_data = val_data.drop(['AircraftID'], axis=1)
        test_data_orig = test_data.drop(['AircraftID'], axis=1)

        test_data_orig = self.remove_noise_ACD_xtest(test_data_orig)
        val_data_orig = self.remove_noise_ACD_xtest(val_data)

        scaler = StandardScaler()
        names_columns = train_data_orig.columns
        print(names_columns)
        train_data = pd.DataFrame(scaler.fit_transform(train_data_orig), columns=names_columns)
        val_data = pd.DataFrame(scaler.fit_transform(val_data_orig), columns=names_columns)
        test_data = pd.DataFrame(scaler.fit_transform(test_data_orig), columns=names_columns)

        train_data['RUL'] = train_data_orig['RUL'].values
        val_data['RUL'] = val_data_orig['RUL'].values
        test_data['RUL'] = test_data_orig['RUL'].values

        if False:
            my_augmenter = (TimeWarp() * 5  # random time warping 5 times in parallel
                            + Quantize(n_levels=[10, 20, 30])  # random quantize to 10-, 20-, or 30- level sets
                            + Drift(max_drift=(0.1, 0.5)) @ 0.8
                            # with 80% probability, random drift the signal up to 10% - 50%
                            )
            dic = defaultdict(list)
            for col in train_data.columns:
                if col == 'RUL' or col == 'Unnamed: 0':
                    continue
                removal_ids = np.unique(train_data['RemovalID'])
                for removal_id in removal_ids:
                    mask = train_data['RemovalID'] == removal_id
                    col_train, y_train = my_augmenter.augment(train_data.loc[mask,col].values, train_data.loc[mask, 'RUL'].values)
                    dic[col].extend(col_train.reshape(-1))
                    dic[col].extend(train_data.loc[mask,col].values)
                    size_removal = len(col_train.reshape(-1))
                    if col == train_data.columns[-1]:
                        dic['RUL'].extend(size_removal - np.array(range(size_removal)))
                        dic['RUL'].extend(train_data.loc[mask, 'RUL'].values)

            train_data = pd.DataFrame(dic)
            import matplotlib.pyplot as plt
            plt.plot(train_data['RUL'].values)
            plt.show()

        x_train, y_train = self.get_lagged_data(train_data)
        x_val, y_val = self.get_lagged_data(val_data)
        x_test, y_test = self.get_lagged_data(test_data)




        callbacks = [
                 EarlyStopping(patience=300, verbose=1, min_delta=0.0001,restore_best_weights=True),
                 #LearningRateScheduler(self.annealing, verbose=1)
             ]
        # opt = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        # opt = tf.keras.optimizers.RMSprop(learning_rate=0.0001, rho=0.9, epsilon=1e-08, decay=0.0)
        #
        # model.compile(loss=['binary_crossentropy'], optimizer=opt, metrics=['accuracy'])


        optimizers = ["rmsprop", "adam"]
        kernel_initializers = ["glorot_uniform", "normal", "zeros"]
        epochs = [50, 100]
        learning_rates = [0.001, 0.0001, 0.0002, 0.0004]
        param_grid = dict(
            optimizer=optimizers,
            nb_epoch=epochs,
            kernel_initializer=kernel_initializers,
            learning_rate= learning_rates)
        # 'categorical_crossentropy'
        # x_train.shape[1:], 'adam', 'zeros', 'kl_divergence', learning_rate=0.001, l1l2=0.1, dropout_rate=0 epochs = 100 good
        # x_train.shape[1:], 'adam', 'zeros', 'sparse_categorical_crossentropy', learning_rate=0.001, l1l2=0.01, dropout_rate=0 epochs = 100 better
        # x_train.shape[1:], 'adadelta', 'zeros', 'kl_divergence', learning_rate=0.001, l1l2=0.01, dropout_rate=0 good training 500
        model1 = self.create_model(x_train.shape[1:], 'adam', 'zeros', 'categorical_crossentropy', learning_rate=0.01,
                                  l1l2=400000, dropout_rate=0.99) #zeros gives best performance
        model2 = self.create_model_advanced(x_train.shape[1:], 'adam', 'zeros', 'categorical_crossentropy', learning_rate=1e-6,
                                  l1l2=40, dropout_rate=0.7)  # zeros gives best performance
        model3 = self.create_model(x_train.shape[1:], 'adam', 'zeros', 'categorical_crossentropy', learning_rate=0.001,
                                   l1l2=400000, dropout_rate=0.99)

        model = model3

        class_weight = {0: 0.5, 1: 0.5, 2: 1, 3: 1}
        history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=1024, epochs=500, verbose=1)
        self.plot_multiclass_history(history)
        predictions_lstm = model.predict(x_test)
        for i in range(8):
            test_data_orig.loc[:, 'Predictions' + str(i)] = predictions_lstm[:, i]

        return test_data_orig

    def annealing(self, epoch):
        lr = 0.001 #0.001 # 0.01
        annealing_start = 5
        return lr if epoch < annealing_start else lr * np.exp(0.1 * (annealing_start - epoch))

    def exp_decay(self, epoch):
        initial_lrate = 0.001
        k = 0.1
        lrate = initial_lrate * math.exp(-k * epoch)
        return lrate


class LSTMModelRegression(Model):

    def get_row_lags(self, unit_data, i):
        periods = self.LAGS

        ix = [i] + [i - period for period in periods]

        if np.sum(np.array(ix) < 0) > 0:
            ix = [i] + ([i] * len(periods))

        selected_data = unit_data.loc[ix, unit_data.columns != 'RUL'].values
        RULs = unit_data.loc[i, 'RUL']
        return selected_data, RULs

    def get_lagged_unit_data(self, data, unit):
        unit_data = data[data['RemovalID'] == unit].reset_index(drop=True)

        lagged_unit_data = [self.get_row_lags(unit_data, i)
                            for i in range(0, len(unit_data))]

        return lagged_unit_data

    def get_lagged_data(self, data):
        lagged_data = chain(*[self.get_lagged_unit_data(data, unit)
                              for unit in data['RemovalID'].unique()])

        measurement, rul = zip(*list(lagged_data))
        measurement, rul = np.array(measurement), np.array(rul)
        return measurement, rul

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
            index_fold += 1
            self.evaluate_classification()
            print('Finished cross validating fold %d %.2f seconds' % (index_fold, (time.time() - start_time)))
        self.result_df = pd.concat(result_dfs)


    def validate_fold(self):
        self.LAGS = 1 + np.array(range(15))
        data = self.df
        test_units = self.test_units
        train_units = np.unique(self.df['AircraftID'].values[~self.df['AircraftID'].isin(test_units)])
        no_train_units = int(len(train_units) * 0.8)

        train_data = data.loc[data['AircraftID'].isin(train_units[:no_train_units]), :]
        val_data = data.loc[data['AircraftID'].isin(train_units[no_train_units:]), :]
        test_data = data.loc[data['AircraftID'].isin(test_units), :]

        train_data = train_data.drop(['AircraftID'], axis=1)
        val_data = val_data.drop(['AircraftID'], axis=1)
        test_data_orig = test_data.drop(['AircraftID'], axis=1)

        std_test = np.nanstd(test_data_orig['RUL'])
        mean_test = np.nanmean(test_data_orig['RUL'])

        test_data_orig = self.remove_noise_ACD_xtest(test_data_orig)
        val_data = self.remove_noise_ACD_xtest(val_data)

        x_train, y_train = self.get_lagged_data(train_data)
        x_val, y_val = self.get_lagged_data(val_data)
        x_test, y_test = self.get_lagged_data(test_data)

        scaler = StandardScaler()
        names_columns = train_data.columns[train_data.columns != 'RUL']
        x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=names_columns)
        x_val = pd.DataFrame(scaler.fit_transform(x_val), columns=names_columns)
        x_test = pd.DataFrame(scaler.fit_transform(x_test), columns=names_columns)

        # Instantiate LSTM Model
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(128, input_shape=x_train.shape[1:], return_sequences=False),
            tf.keras.layers.Dense(
                1, kernel_initializer=tf.initializers.zeros())
        ])

        callbacks = [
            EarlyStopping(patience=15, verbose=1),
            LearningRateScheduler(self.annealing, verbose=1)
        ]
        learning_rate = 0.01
        opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        model.compile(loss=['binary_crossentropy'], optimizer=opt, metrics=['binary_crossentropy'])
        model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=128, epochs=150, callbacks=callbacks,
                  verbose=1)
        predictions_lstm = model.predict(x_test)
        predictions_nn = predictions_lstm.reshape(-1)
        y_pred = predictions_nn * std_test + mean_test
        test_data_orig.loc[:, 'Predictions'] = y_pred

        return test_data_orig

    def annealing(self, epoch):
        lr = 0.001 # 0.01
        annealing_start = 5
        return lr if epoch < annealing_start else lr * np.exp(0.1 * (annealing_start - epoch))