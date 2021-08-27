import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from collections import defaultdict
from random import random
from ACD import ACD


class MultiClassifier:

    def __init__(self, filename):
        self.colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen', 'pink', 'gray', 'silver',
                       'chartreuse', 'darkturquoise']
        self.result_df = None
        self.test_units = []
        self.n_classes = -1
        self.df = self.load(filename)
        self.get_time()

        if False:
            for col in self.df.columns:
                if '7' in col or '8' in col or 'turbulence' in col or '9' in col or 'MvgAvg' in col or 'DiffMedian' in col or 'ExpSmooth' in col or 'MaxMedian' in col or 'Paramedian' in col:
                    self.df.drop(col, axis=1, inplace=True)
        for i in range(20):
            self.df['taxi_warmup_Meandiff_par6_sys_1_' + str(i)] = self.df['taxi_warmup_Meandiff_par6_sys_1'].values
        cols = list(self.df.columns)
        print('\nFiltered columns:', cols)

        print(self.df.head())
        print(self.df.RUL)

        #self.remove_noise()

    def load(self, filename):
        with open(filename, 'rb') as handle:
            aircraft_dic = pickle.load(handle)
        dict1 = {x: aircraft_dic[x] for x in aircraft_dic.keys()}
        df = pd.DataFrame(dict1)

        df.replace([np.inf, -np.inf, None], np.nan)
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        cols = list(df.columns)
        print('Columns: ' + str(cols))
        df['RemovalID'] = df['AircraftID'] + '-' + df['RemovalID'].astype(str)
        return df

    def remove_noise(self):
        for col in self.df.columns:
            if 'taxi' in col and 'ExpSmooth_' not in col and 'MvgAvg_'not in col:
                self.remove_noise_acd(col)
        self.df.to_csv('data//taxi_data_denoised.csv')

    def remove_noise_acd(self, feature_name):
        removal_ids = np.unique(self.df['RemovalID'])
        self.df['ACD_' + feature_name] = 0
        bad_removals = []
        for removal_id in removal_ids:
            print('Removing noise from removal %s of param %s' % (removal_id, feature_name))
            signal_removal = self.df.loc[self.df['RemovalID'] == removal_id, feature_name].values
            try:
                acd_signal = ACD.increase_monotonicity(signal_removal)
                self.df.loc[self.df['RemovalID'] == removal_id, 'ACD' + feature_name] = acd_signal
            except:
                print('Problematic:', list(signal_removal))
                bad_removals.append(removal_id)
        self.df = self.df.loc[~self.df['RemovalID'].isin(bad_removals), :]


    def filter_short_removals(self):
        removal_ids = np.unique(self.df['RemovalID'])
        good_removals = []
        for removal_id in removal_ids:
            ruls_removal = self.df.loc[self.df['RemovalID'] == removal_id, 'RUL'].values
            if len(ruls_removal) > 5: good_removals.append(removal_id)
        return self.df.loc[self.df['RemovalID'].isin(good_removals), :]

    def get_time(self):
        removal_ids = np.unique(self.df['RemovalID'])
        self.df['Time'] = 0
        for removal_id in removal_ids:
            ruls_removal = self.df.loc[self.df['RemovalID'] == removal_id, 'RUL'].values
            self.df.loc[self.df['RemovalID'] == removal_id, 'Time'] = np.max(ruls_removal) - ruls_removal

    def evaluate_ROC_Curve(self):
        y_test = self.result_df['RUL'].values
        n_classes = self.n_classes
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        index = 0
        for i in range(350, 0, -50):
            upper = i
            lower = upper - 50
            if i == 350:
                y_test = (self.result_df['RUL'].values > 300)* 1
            else:
                y_test = ((self.result_df['RUL'].values > lower) & (self.result_df['RUL'].values <= upper)) * 1

            y_score = self.result_df.loc[:, 'Predictions' + str(index)].values
            fpr[index], tpr[index], _ = roc_curve(y_test, y_score)
            roc_auc[index] = auc(fpr[index], tpr[index])
            index += 1

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.figure()
        colors = cycle(self.colors)
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=4,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=4)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        plt.show()

    def evaluate(self):
        self.evaluate_ROC_Curve()
        removal_ids = np.unique(self.result_df['RemovalID'].values)
        for removal_id in removal_ids:
            self.plot_removal(removal_id)

    def plot_removal(self, removal_id):
        index = 0
        eol_dic = defaultdict(str)
        for i in range(350, 0, -50):
            if i == 350:
                eol_dic[index] = 'Far from removal (> 300 days)'
            else:
                eol_dic[index] = 'Between ' + str(i) + ' and ' + str(i - 50)
            index += 1

        index = 0
        y_true_removal = self.result_df.loc[self.result_df['RemovalID'] == removal_id, 'RUL'].values
        for i in range(350, 0, -50):
            feature_name = 'Predictions' + str(index)
            predictions_removal = self.result_df.loc[self.result_df['RemovalID'] == removal_id, feature_name].values
            plt.axvline(i, color=self.colors[index])
            if i != 350:
                plt.gca().axvspan(i, i - 50, alpha=0.02, color=self.colors[index])
            else:
                plt.gca().axvspan(len(y_true_removal), 300, alpha=0.05, color=self.colors[index])
            plt.scatter(y_true_removal, predictions_removal, c=self.colors[index], label=eol_dic[index])
            index += 1
        plt.xlim(0, max(y_true_removal))
        plt.gca().invert_xaxis()
        plt.ylim(0,1)
        plt.legend()
        plt.show()

    def plot_history(self):
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        history = self.history.history

        ax[0].plot(history['loss'], 'r', label='train')
        ax[0].plot(history['val_loss'], 'b', label='val')
        ax[0].set_xlabel('epoch')
        ax[0].set_ylabel('loss value')
        ax[0].set_title('Loss - MSE')
        ax[0].legend()

        ax[1].plot(history['categorical_accuracy'], 'r', label='train')
        ax[1].plot(history['val_categorical_accuracy'], 'b', label='val')
        ax[1].set_xlabel('epoch')
        ax[1].set_ylabel('metric value')
        ax[1].set_title('Metric - acc')
        ax[1].legend()

        plt.tight_layout()
        plt.savefig('figs/lstm_history.png')
        plt.show()