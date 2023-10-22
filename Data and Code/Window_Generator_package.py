import pandas as pd
import numpy as np
import os

import warnings
warnings.filterwarnings("ignore")

class NotRunError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message

class Window_Generator(object):

    def __init__(self, train_series, test_series, window_length = 5):
        """
        Obtain 3D series for Deep Learning Usage on a time series.

        Parameters:
            train_series: the 1D series of train set
            test_series: the 1D series of test set
            window_length: the window_length is the timestep in the 3D series (samples, timestep, features)

        Note:
            Here, the features are 1 only, which is jut the series previous values.
        """  
        if (train_series.ndim != 1 or test_series.ndim != 1):
            raise ValueError("The input series should be 1D.")
        self.train_series = train_series
        self.test_series = test_series
        self.window_length = window_length
        self.train_index = train_series.index
        self.test_index = test_series.index
        self.train_length = len(train_series)
        self.test_length = len(test_series)
        self.total_length = self.train_length + self.test_length
        # This variable is used to determine whether the project have ran the window_generation method
        self.if_window_generator = False

    def standardization(self, show = None):
        
        # This method will standardize the train and test series
        # Train series will share the same mean and standard deviation
        train_mean = np.full(shape = (len(self.train_series)), fill_value = np.mean(self.train_series))
        train_std = np.full(shape = (len(self.train_series)), fill_value = np.std(self.train_series))
        
        # Test series will have different mean and standard deviation in each time step
        test_mean = np.zeros(shape=(len(self.test_series)))
        test_std = np.zeros(shape=(len(self.test_series)))
        train_length = len(self.train_series)
        full_series = pd.concat([self.train_series, self.test_series], axis = 0).values
        for i in range(1, len(self.test_series) + 1):
            test_mean[i - 1] = np.mean(full_series[: train_length + i])
            test_std[i - 1] = np.std(full_series[: train_length + i])
        
        # Standardize the train and test set
        train_series_standard = self.train_series.copy()
        train_series_standard = (train_series_standard - train_mean) / train_std

        test_series_standard = self.test_series.copy()
        test_series_standard = (test_series_standard - test_mean) / test_std

        self.train_series_standard = train_series_standard
        self.test_series_standard = test_series_standard
        self.train_mean = train_mean
        self.train_std = train_std
        self.test_mean = test_mean
        self.test_std = test_std

        if (show != None):
            output = (f"train_series_shape: {train_series_standard.shape}\n"
                      f"test_series_shape: {test_series_standard.shape}")
            print(output)

        return train_series_standard, test_series_standard, train_mean, train_std, test_mean, test_std

    def window_generation(self, transform = True, show = None):
        """
        Combined the two series together
        Here this method will directly used to the standardized series

        Parameters:
            transform: reshape the series into 3D

        """  
        # Combined the two series together
        full_series = pd.concat([self.train_series_standard, self.test_series_standard], axis = 0).values.tolist()
        X = []
        Y  = []

        # Create lag dataset based on the window_length
        for i in range(self.total_length - self.window_length):
            X.append(full_series[i: i + self.window_length])
            Y.append(full_series[i + self.window_length])
        
        X = np.array(X)
        Y = np.array(Y)
        # print(X.shape)
        # print(Y.shape)
        X_train = X[ : self.train_length - self.window_length]
        X_test = X[self.train_length - self.window_length : ]
        Y_train = Y[ : self.train_length - self.window_length]
        Y_test = Y[self.train_length - self.window_length : ]

        if transform:
        # return 3D List as the input to the Deep Learning Algorithm
            X_train = X_train.reshape([X_train.shape[0], X_train.shape[1], 1])
            Y_train = Y_train.reshape([Y_train.shape[0], 1, 1])
            X_test = X_test.reshape([X_test.shape[0], X_test.shape[1], 1])
            Y_test = Y_test.reshape([Y_test.shape[0], 1, 1])
        
        if (show != None):
            output = (f"X_train: {X_train.shape}\n"
                      f"Y_train: {Y_train.shape}\n"
                      f"X_test: {X_test.shape}\n"
                      f"Y_test: {Y_test.shape}")
            print(output)

        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test

        self.if_window_generator = True
        return X_train, X_test, Y_train, Y_test

    def data_getter(self):
        if(self.if_window_generator == False):
            raise NotRunError('You should run the window_generation method first')

        return self.X_train, self.X_test, self.Y_train, self.Y_test
    
    def denormalize_train(self, train_series = None):
        if train_series == None:
            train_series = self.train_series_standard
        
        if (len(train_series) != len(self.train_mean)):
            raise ValueError('The Lengths don\'t match')
        
        train_denormalized = train_series.copy()
        train_denormalized = (train_denormalized * self.train_std) + self.train_mean

        return train_denormalized
    
    def denormalize_test(self, test_series = None):
        if test_series == None:
            test_series = self.test_series_standard
        
        if (len(test_series) != len(self.test_mean)):
            raise ValueError('The Lengths don\'t match')
        
        test_denormalized = test_series.copy()
        test_denormalized = (test_denormalized * self.test_std) + self.test_mean

        return test_denormalized