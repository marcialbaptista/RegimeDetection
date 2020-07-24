# -----------------------------------------------------------
# demonstrates how to write ms excel files using python-openpyxl
#
# (C) 2020 Marcia Baptista, Lisbon, Portugal
# Released under GNU Public License (GPL)
# email marcia.lbaptista@gmail.com
# -----------------------------------------------------------

import pandas as pd
import numpy as np
from skimage import measure
from sklearn.cluster import KMeans
from minisom import MiniSom

###########################################
#
# Auxiliary reading functions
#
###########################################


def read_pandas_array(pd_array):
    '''
    joins the data of different C-MAPSS files in a single dataframe
    :param pd_array: list of pandas array
    :return: dataframe
    '''
    frames = []
    for i in range(len(pd_array)):
        frames.append(pd_array[i])
    return pd.concat(frames, ignore_index=True)

###########################################
#
# Features
#
###########################################


# features of C-MAPSS different data sets
feature_names = ['unit_number', 'time', 'altitude', 'mach_number', 'throttle_resolver_angle',
                 'T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 'Nf', 'Nc', 'epr', 'Ps30', 'phi',
                 'NRf', 'NRc', 'BPR', 'farB', 'htBleed', 'Nf_dmd', 'PCNfR_dmd', 'W31', 'W32']


# features that correspond only the sensor names
sensor_names = ['T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 'Nf', 'Nc', 'epr', 'Ps30', 'phi',
                'NRf', 'NRc', 'BPR', 'farB', 'htBleed', 'Nf_dmd', 'PCNfR_dmd', 'W31', 'W32']


# features that correspond to operational conditions
op_condition_features = ['altitude', 'mach_number', 'throttle_resolver_angle']


dataset2 = [
    pd.read_csv('data/train_FD002.txt', sep='\s+', names=feature_names)
]

dataset4 = [
    pd.read_csv('data/train_FD004.txt', sep='\s+', names=feature_names)
]


#############################################
#
# K-means clustering of regimes using
# operating features (op_condition_features)
#
#############################################

def detect_regimes(dataset_id, dimension=20):
    df = read_pandas_array(dataset2)

    if dataset_id == 2:
        df = read_pandas_array(dataset2)
    elif dataset_id == 4:
        df = read_pandas_array(dataset4)
    else:
        print("C-MAPSS has only two datasets with more than one operating mode: FD002 and FD004.\n",
              "Please choose dataset id 2 or 4")
        return 0

    engine_ids = np.unique(df["unit_number"])[:2]
    print("Number of engine units in dataset " + str(dataset_id) + ": ", len(engine_ids))

    nr_clusters = 6
    match_percentages, bmus_indexes = [], []

    for engine_id in engine_ids:
        print("-----------------------------------------")
        print("Clustering regimes of engine", engine_id)
        df_unit = df.loc[df["unit_number"] == engine_id, op_condition_features]
        data = df_unit.to_numpy()

        # find the grouth truth (GT) regimes
        k_means = KMeans(n_clusters=nr_clusters, random_state=0).fit(df_unit.loc[:, op_condition_features])

        # build the SOM
        x_dim = dimension
        y_dim = dimension

        som = MiniSom(x_dim, y_dim, len(data[0]), sigma=0.3, neighborhood_function="gaussian", learning_rate=0.5, random_seed=1)
        som.train_random(data, num_iteration=50000)

        # find best matching units (active neurons)
        bmus = []
        bmus_index = []
        print("Finding the Best Maching Units (BMUs)")
        for row in data:
            w = som.winner(row)
            bmus.append(w)
            bmus_index.append(w[0]*10 + w[1])

        # perform clustering of BMUs
        print("Clustering the BMUs")
        im = np.zeros((x_dim, y_dim))
        for i in range(x_dim):
            for j in range(y_dim):
                im[i, j] = 0

        for point in bmus:
            im[y_dim - point[1] - 1, point[0]] = 1

        blobs_labels = measure.label(im, background=0, neighbors=4)

        regime_clusters = dict()
        for i in range(x_dim):
            for j in range(y_dim):
                if im[y_dim - j - 1, i] != 0:
                    regime_clusters[(i, j)] = blobs_labels[y_dim - j - 1, i]

        bmus_index = []
        for w in bmus:
            cluster_index = regime_clusters[w]
            bmus_index.append(cluster_index)


        # Equalize the regimes
        bmus_index_orig = bmus_index
        regimes_found = np.unique(bmus_index)
        print("Found ",len(regimes_found), "SOM regimes found")
        k_means_labels = k_means.labels_
        bmus_index = np.array(bmus_index)
        regimes_GT = np.unique(k_means_labels)
        found_matches = [] # clusters GT that were already found

        # Assign the GT clusters to the found clusters
        for regime_found in regimes_found:
            #print("Regime found", regime_found)
            max = 0
            match_regime = 0
            for regime_GT in regimes_GT:
                match_samples = (np.logical_and(k_means_labels == regime_GT, bmus_index == regime_found))
                nr_assigned_samples_regime_GT = np.count_nonzero(match_samples)
                if max <= nr_assigned_samples_regime_GT:
                    match_regime = regime_GT
                    max = nr_assigned_samples_regime_GT

            if not (match_regime in found_matches):
                bmus_index[bmus_index == regime_found] = match_regime + 10

            if match_regime in found_matches:
                print("Regime", regime_found, np.unique(bmus_index))
                bmus_index[bmus_index == regime_found] = 98
                print(np.unique(bmus_index))

            found_matches.append(match_regime)

        print(np.unique(bmus_index))

        for regime_GT in regimes_GT:
            k_means_labels[k_means_labels == regime_GT] = regime_GT + 10

        # Calculate the matches between the GT clusters and the found clusters
        matches = np.equal(k_means_labels, bmus_index)
        match_percentage = 100 * (np.count_nonzero(matches) / len(matches))
        match_percentages.append(match_percentage)
        bmus_indexes.append(bmus_index)

        print("Match (%): ", match_percentage)

        print("SOM regimes", np.unique(bmus_index))
        print("K-mean regimes", np.unique(k_means_labels))

    return bmus_indexes, match_percentages

bmus_indexes, match_percentages = detect_regimes(dataset_id=2, dimension=20)