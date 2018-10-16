import numpy as np
np.set_printoptions(threshold=np.nan)
import tensorflow as tf
import os
import csv

class EEG():

    batch_index = 0

    X_list = []
    Y_list = []
    X_train_list = []
    Y_train_list = []
    X_valid_list = []
    Y_valid_list = []
    X_test_list = []
    Y_test_list = []

    # Normalization of X
    def normalize(self, X):
        col_max = np.max(X)
        col_min = np.min(X)
        return np.divide(X - col_min, col_max - col_min)

    # Shuffle X and Y with the same order
    def reshuffle(self, X, Y):
        num_data = len(X)
        perm_indices = np.arange(num_data)
        np.random.shuffle(perm_indices)
        X = X[perm_indices]
        Y = Y[perm_indices]
        return X, Y

    # Read data from file, shuffle, separate them into train, validation, test lists and normalize them
    def input_data(self, path):

        for file in os.listdir(path):

            filename = path+'/'+file

            with open(filename, newline='') as csvfile:
                filereader = csv.reader(csvfile)
                next(filereader) # skip header
                data = np.float32([r for r in filereader])

            for row in data:
                EEG.X_list.append(row[:14])
                if row[14] == 0.0:
                    label = [1, 0]
                else:
                    label = [0, 1]
                EEG.Y_list.append(label)

        train_index = np.random.choice((len(EEG.X_list)-64), int(round((len(EEG.X_list)-64) * 0.8)), replace=False)
        print(len(train_index))
        valid_index = np.array(list(set(range((len(EEG.X_list)-64))) - set(train_index)))
        print(len(valid_index))
        test_index = np.array(list(set(range(len(EEG.X_list))) - set(train_index) - set(valid_index)))

        EEG.X_list = np.asarray(EEG.X_list)
        EEG.Y_list = np.asarray(EEG.Y_list)
        # Comment next line if you want to run RNN.py
        EEG.X_list, EEG.Y_list = self.reshuffle(EEG.X_list, EEG.Y_list)

        EEG.X_train_list = np.asarray(EEG.X_train_list)
        EEG.Y_train_list = np.asarray(EEG.Y_train_list)
        
        EEG.X_valid_list = np.asarray(EEG.X_valid_list)
        EEG.Y_valid_list = np.asarray(EEG.Y_valid_list)

        EEG.X_test_list = np.asarray(EEG.X_test_list)
        EEG.Y_test_list = np.asarray(EEG.Y_test_list)

        EEG.X_train_list = EEG.X_list[train_index]
        EEG.X_train_list = self.normalize(EEG.X_train_list)
        EEG.Y_train_list = EEG.Y_list[train_index]

        EEG.X_valid_list = EEG.X_list[valid_index]
        EEG.X_valid_list = self.normalize(EEG.X_valid_list)
        EEG.Y_valid_list = EEG.Y_list[valid_index]

        EEG.X_test_list = EEG.X_list[test_index]
        EEG.X_test_list = self.normalize(EEG.X_test_list)
        EEG.Y_test_list = EEG.Y_list[test_index]

    # Get next data batch of size 'batch_size' 
    def next_batch(self, batch_size):
        n_train_examples = len(EEG.X_train_list)
        if EEG.batch_index + batch_size < n_train_examples:
            X_train_batch = EEG.X_train_list[EEG.batch_index:EEG.batch_index + batch_size]
            Y_train_batch = EEG.Y_train_list[EEG.batch_index:EEG.batch_index + batch_size]
            EEG.batch_index = EEG.batch_index + batch_size
            return X_train_batch, Y_train_batch
        else:
            return None, None

    def reset_batch_index(self):
        EEG.batch_index = 0
