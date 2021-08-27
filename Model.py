import math
from Utils import Utils
from ACD import ACD
from sklearn import metrics
import pickle
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from scipy import interp


class Model:

    def __init__(self):
        self.colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen', 'pink', 'gray', 'silver',
                       'chartreuse', 'darkturquoise']
        with open('data\data.pickle', 'rb') as handle:
            aircraft_dic = pickle.load(handle)
        dict1 = {x: aircraft_dic[x] for x in aircraft_dic.keys()}
        self.df = pd.DataFrame(dict1)
        self.df['RUL'] = pd.to_timedelta(self.df['RUL']).dt.days
        aircraft_ids = [int(str.split(x, '-')[1]) * 10 for x in self.df['AircraftID'].values]
        self.df['RemovalID'] = aircraft_ids + self.df['RemovalID'].values
        self.result_df = None
        self.test_units = []
        self.result_df = None
        self.df.dropna(inplace=True)
        self.df = pd.read_csv('data//data.csv')
        self.df.dropna(inplace=True)
        if False: self.remove_noise()

    def evaluateMultiClassROCCurve(self):
        y_test = self.result_df.loc[:, 'RUL'].values
        n_classes = 8
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            if i == 0: y_test = (self.result_df['RUL'].values > 300)*1
            if i == 1: y_test = ((self.result_df['RUL'].values > 250) & (self.result_df['RUL'].values <= 300)) * 1
            if i == 2: y_test = ((self.result_df['RUL'].values > 200) & (self.result_df['RUL'].values <= 250)) * 1
            if i == 3: y_test = ((self.result_df['RUL'].values > 150) & (self.result_df['RUL'].values <= 200)) * 1
            if i == 4: y_test = ((self.result_df['RUL'].values > 100) & (self.result_df['RUL'].values <= 150)) * 1
            if i == 5: y_test = ((self.result_df['RUL'].values > 50) & (self.result_df['RUL'].values <= 100))* 1
            if i == 6: y_test = ((self.result_df['RUL'].values > 20) & (self.result_df['RUL'].values <= 50)) * 1
            if i == 7: y_test = ((self.result_df['RUL'].values <= 20)) * 1
            y_score = self.result_df.loc[:, 'Predictions' + str(i)].values

            print('y_test=', y_test, np.sum(y_test), y_score, np.sum(y_score))

            fpr[i], tpr[i], _ = roc_curve(y_test, y_score)
            roc_auc[i] = auc(fpr[i], tpr[i])

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

    def evaluate_ROCcurve(self):
        y_true = self.result_df.loc[:, 'RUL'].values
        y_pred = self.result_df.loc[:, 'Predictions'].values
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
        plt.figure()
        lw = 2
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

    def remove_noise(self):
        for col in self.df.columns:
            if col in ['Taxi_Mean', 'Taxi_Mean_Varies']:
                self.remove_noise_Mvgavg(col)
                print('Removing noise from %s' % (col))
                self.remove_noise_Accum(col)
                self.remove_noise_ACD(col)
                self.df.to_csv('data//data.csv')

    def evaluate_accuracy(self):
        dic_accuracy, dic_accuracy_horizon = defaultdict(int), defaultdict(int)
        y_true = self.result_df.loc[:, 'RUL'].values
        y_pred = self.result_df.loc[:, 'Predictions'].values
        removal_ids = np.unique(self.result_df['RemovalID'].values)
        y_true_percentage = np.copy(y_true)
        for removal_id in removal_ids:
            mask = (self.result_df['RemovalID'].values == removal_id)
            y_true_percentage[mask] = 100 * (y_true[mask] / np.max(y_true[mask]))
        for i in range(0, 100, 10):
            y_trues_life = y_true_percentage[(y_true_percentage >= i) & (y_true_percentage < i+10)]
            y_pred_life = y_pred[(y_true_percentage >= i) & (y_true_percentage < i+10)]
            acc = np.sum((y_pred_life >= y_trues_life * 0.6) & (y_pred_life <= y_trues_life * 1.4))/ len(y_pred_life)
            acc_horizon = np.sum((y_pred_life >= y_trues_life - 10) & (y_pred_life <= y_trues_life + 10)) / len(y_pred_life)
            dic_accuracy[i+10] = acc
            dic_accuracy_horizon[i + 10] = acc_horizon
        self.accuracies_dic = dic_accuracy
        self.accuracies_dic_horizon = dic_accuracy_horizon

    def plot_accuracy(self):
        for key in self.accuracies_dic.keys():
            print('Accuracy at %d equals %.2f %.2f' % (key, 100 * self.accuracies_dic[key], 100 * self.accuracies_dic_horizon[key]))

    def evaluate_classification(self):
        self.evaluate_ROCcurve()
        removal_ids = np.unique(self.result_df['RemovalID'].values)
        for removal_id in removal_ids:
            self.plot_classification_removal(removal_id)

    def evaluate_multiclassification(self):
        self.evaluateMultiClassROCCurve()
        removal_ids = np.unique(self.result_df['RemovalID'].values)
        for removal_id in removal_ids:
            self.plot_multiclassification_removal(removal_id)

    def evaluate(self):
        print('MAE = %.2f RMSE = %.2f' % (self.calculate_MAE(), self.calculate_RMSE()))
        self.evaluate_accuracy()
        self.plot_accuracy()
        removal_ids = np.unique(self.result_df['RemovalID'].values)
        for removal_id in removal_ids:
            self.plot_alpha_lambda(removal_id)

    def plot_alpha_lambda(self, removal_id):
        y_true_removal = self.result_df.loc[self.result_df['RemovalID'] == removal_id, 'RUL'].values
        predictions_removal = self.result_df.loc[self.result_df['RemovalID'] == removal_id, 'Predictions'].values
        plt.scatter(y_true_removal, predictions_removal)
        plt.scatter(y_true_removal, y_true_removal)
        plt.show()

    def plot_multiclassification_removal(self, removal_id):
        y_true_removal = self.result_df.loc[self.result_df['RemovalID'] == removal_id, 'RUL'].values
        size_removal = len(y_true_removal)
        n_classes = 8
        eol_dic = {0: 'Far away', 1: 'Less than 300 days',
                   2:  'Less than 250 days', 3: 'Less than 200 days',
                   4: 'Less than 150 days', 5: 'Less than 100 days',
                   6: 'Less than 50 days', 7: 'Less than 20 days'
                   }
        for i in range(n_classes):
            feature_name = 'Predictions' + str(i)
            predictions_removal = self.result_df.loc[self.result_df['RemovalID'] == removal_id, feature_name].values
            index = 0
            for j, m in zip([300, 250, 200, 150, 100, 50, 20], [250, 200, 150, 100, 50, 20, 0]):
                if size_removal >= j:
                    plt.axvline(j, color=self.colors[index])
                    plt.gca().axvspan(j, m, alpha=0.01, color=self.colors[index])
                if size_removal < j and j == 20:
                    plt.axvline(j, color='red')
                    plt.gca().axvspan(j, m, alpha=0.01, color=self.colors[index])
                index += 1
            plt.scatter(range(len(y_true_removal)), predictions_removal, c=self.colors[i], label=eol_dic[i])
        plt.gca().invert_xaxis()
        plt.legend()
        plt.show()

    def plot_multiclass_history(self, history):
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        history = history.history

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

    def plot_classification_removal(self, removal_id):
        y_true_removal = self.result_df.loc[self.result_df['RemovalID'] == removal_id, 'RUL'].values
        predictions_removal = self.result_df.loc[self.result_df['RemovalID'] == removal_id, 'Predictions'].values
        plt.scatter(range(len(y_true_removal)), y_true_removal, c='red')
        plt.scatter(range(len(y_true_removal)), predictions_removal, c='green')
        plt.show()

    def calculate_MAE(self):
        y_true = self.result_df['RUL'].values
        predictions = self.result_df['Predictions'].values
        error = np.mean(np.abs(y_true - predictions))
        return error

    def calculate_RMSE(self):
        y_true = self.result_df['RUL']
        predictions = self.result_df['Predictions']
        error = math.sqrt(np.mean(np.power(y_true - predictions, 2)))
        return error

    def remove_noise_Mvgavg(self, feature_name):
        removal_ids = np.unique(self.df['RemovalID'])
        self.df['MovAvg' + feature_name] = 0
        for removal_id in removal_ids:
            signal_removal = self.df.loc[self.df['RemovalID'] == removal_id, feature_name].values
            denoised_signal = Utils.moving_average(signal_removal, 20)
            self.df.loc[self.df['RemovalID'] == removal_id, 'MovAvg' + feature_name] = denoised_signal
            #self.plot_denoised_signal(signal_removal, denoised_signal)

    def remove_noise_ACD(self, feature_name):
        removal_ids = np.unique(self.df['RemovalID'])
        self.df['ACD' + feature_name] = 0
        for removal_id in removal_ids:
            signal_removal = self.df.loc[self.df['RemovalID'] == removal_id, feature_name].values
            acd_signal = ACD.increase_monotonicity(signal_removal)
            self.df.loc[self.df['RemovalID'] == removal_id, 'ACD' + feature_name] = acd_signal

    def remove_noise_ACD_xtrain(self, data):
        for col in ['Taxi_Mean', 'Taxi_Mean_Varies']:
            removal_ids = np.unique(data['RemovalID'])
            data['ACD' + col] = 0
            for removal_id in removal_ids:
                signal_removal = data.loc[data['RemovalID'] == removal_id, col].values
                acd_signal = ACD.increase_monotonicity(signal_removal)
                data.loc[data['RemovalID'] == removal_id, 'ACD' + col] = acd_signal

    def remove_noise_ACD_xtest(self, data):
        for col in ['Taxi_Mean', 'Taxi_Mean_Varies']:
            data['ACD' + col] = data['Accum' + col]
        return data

    def remove_noise_Accum(self, feature_name):
        removal_ids = np.unique(self.df['RemovalID'])
        self.df['Accum' + feature_name] = 0
        for removal_id in removal_ids:
            signal_removal = self.df.loc[self.df['RemovalID'] == removal_id, feature_name].values
            denoised_signal = Utils.moving_quantile(signal_removal, 20)
            self.df.loc[self.df['RemovalID'] == removal_id, 'Accum' + feature_name] = denoised_signal
            #self.plot_denoised_signal(signal_removal, denoised_signal)

    def plot_denoised_signal(self, signal, denoised_signal):
        plt.scatter(range(len(signal)), signal)
        plt.plot(denoised_signal)
        plt.show()
