# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping
from sklearn.utils import shuffle
from tcn import TCN
from tensorflow.keras.layers import Dropout, Dense, LSTM, GRU, Bidirectional
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
import math
import time
from bayes_opt import BayesianOptimization

def BOmodel(time_step, units, units_1):

    time_step = int(time_step)
    units = int(units)
    units_1 = int(units_1)

    data = pd.read_csv("./data/all_data.csv", header=0)

    x = data.iloc[0:len(data), 5:10].values
    y = data.iloc[0:len(data), -1].values
    future_num = 5

    def Z_ScoreNormalization(x, mean, sigma):
        x = (x - mean) / sigma
        return x

    for i in range(future_num):
        mean = np.average(x[:, i])
        sigma = np.std(x[:, i])
        for j in range(len(data)):
            x[j, i] = Z_ScoreNormalization(x[j, i], mean, sigma)

    train_volume = 43163
    val_volume = 17437
    test_volume = 16367

    train_x = x[0:train_volume, :]
    val_x = x[train_volume:train_volume+val_volume, :]
    test_x = x[-test_volume:, :]
    train_y = y[0:train_volume]
    val_y = y[train_volume:train_volume+val_volume]
    test_y = y[-test_volume:]

    x_train = []
    y_train = []
    x_val = []
    y_val = []
    x_test = []
    y_test = []

    for i in range(time_step, len(train_x) + 1):
        x_train.append(train_x[i - time_step:i, 0:future_num])
        y_train.append(train_y[i - 1])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], time_step, future_num))
    x_train, y_train = np.array(x_train), np.array(y_train)

    for i in range(time_step, len(val_x) + 1):
        x_val.append(val_x[i - time_step:i, 0:future_num])
        y_val.append(val_y[i - 1])
    x_val, y_val = np.array(x_val), np.array(y_val)
    x_val = np.reshape(x_val, (x_val.shape[0], time_step, future_num))
    x_val, y_val = np.array(x_val), np.array(y_val)

    for i in range(time_step, len(test_x) + 1):
        x_test.append(test_x[i - time_step:i, 0:future_num])
        y_test.append(test_y[i - 1])
    x_test, y_test = np.array(x_test), np.array(y_test)
    x_test = np.reshape(x_test, (x_test.shape[0], time_step, future_num))
    x_test, y_test = np.array(x_test), np.array(y_test)

    model = tf.keras.Sequential([
        LSTM(units, input_shape=(time_step, future_num), return_sequences=False),
        Dense(units_1),
        Dense(1)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mean_squared_error', metrics=['accuracy'])


    cp_callback = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(x_train, y_train, batch_size=48, epochs=100, validation_data=(x_val, y_val), validation_freq=1,
                        callbacks=cp_callback)

    predicted_data = model.predict(x_test)

    predicted_wave = []
    for i in predicted_data:
        for j in i:
            predicted_wave.append(j)

    predicted_wave_height = []
    for i in range(len(predicted_wave)):
        predicted_wave_height.append(predicted_wave[i])

    mse = mean_squared_error(predicted_wave_height, y_test)
    return 1-mse

def LSTMmodel(time_step, units, units_1):

    time_step = int(time_step)
    units = int(units)
    units_1 = int(units_1)

    data = pd.read_csv("./data/all_data.csv", header=0)

    x = data.iloc[0:len(data), 5:10].values
    y = data.iloc[0:len(data), -1].values
    future_num = 5

    def Z_ScoreNormalization(x, mean, sigma):
        x = (x - mean) / sigma
        return x

    for i in range(future_num):
        mean = np.average(x[:, i])
        sigma = np.std(x[:, i])
        for j in range(len(data)):
            x[j, i] = Z_ScoreNormalization(x[j, i], mean, sigma)

    train_volume = 43163
    val_volume = 17437
    test_volume = 16367

    train_x = x[0:train_volume, :]
    val_x = x[train_volume:train_volume+val_volume, :]
    test_x = x[-test_volume:, :]
    train_y = y[0:train_volume]
    val_y = y[train_volume:train_volume+val_volume]
    test_y = y[-test_volume:]

    x_train = []
    y_train = []
    x_val = []
    y_val = []
    x_test = []
    y_test = []

    for i in range(time_step, len(train_x) + 1):
        x_train.append(train_x[i - time_step:i, 0:future_num])
        y_train.append(train_y[i - 1])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], time_step, future_num))
    x_train, y_train = np.array(x_train), np.array(y_train)

    for i in range(time_step, len(val_x) + 1):
        x_val.append(val_x[i - time_step:i, 0:future_num])
        y_val.append(val_y[i - 1])
    x_val, y_val = np.array(x_val), np.array(y_val)
    x_val = np.reshape(x_val, (x_val.shape[0], time_step, future_num))
    x_val, y_val = np.array(x_val), np.array(y_val)

    for i in range(time_step, len(test_x) + 1):
        x_test.append(test_x[i - time_step:i, 0:future_num])
        y_test.append(test_y[i - 1])
    x_test, y_test = np.array(x_test), np.array(y_test)
    x_test = np.reshape(x_test, (x_test.shape[0], time_step, future_num))
    x_test, y_test = np.array(x_test), np.array(y_test)

    model = tf.keras.Sequential([
        LSTM(units, input_shape=(time_step, future_num), return_sequences=False),
        Dense(units_1),
        Dense(1)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mean_squared_error', metrics=['accuracy'])


    cp_callback = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(x_train, y_train, batch_size=48, epochs=100, validation_data=(x_val, y_val), validation_freq=1,
                        callbacks=cp_callback)

    predicted_data = model.predict(x_test)

    predicted_wave = []
    for i in predicted_data:
        for j in i:
            predicted_wave.append(j)

    predicted_wave_height = []
    for i in range(len(predicted_wave)):
        predicted_wave_height.append(predicted_wave[i])

    # 评价模型
    mse = mean_squared_error(predicted_wave_height, y_test)
    rmse = math.sqrt(mean_squared_error(predicted_wave_height, y_test))
    mae = mean_absolute_error(predicted_wave_height, y_test)
    mape = np.mean(np.abs((y_test - predicted_wave_height) / y_test))
    print('mse: %.4f' % mse)
    print('rmse: %.4f' % rmse)
    print('mae: %.4f' % mae)
    print('mape: %.4f' % mape)
    from scipy.stats import pearsonr
    r, p = pearsonr(y_test, predicted_wave_height)
    print("R: %.4f" % r)
    print("R2: %.4f" % r2_score(y_test, predicted_wave_height))

lstm_bo = BayesianOptimization(
    BOmodel,
    {
        'time_step':(2, 128),
        'units':(2, 128),
        'units_1':(2, 128),
    }
)
lstm_bo.maximize()
optimal = lstm_bo.max
params = optimal['params']
time_step = params['time_step']
units = params['units']
units_1 = params['units_1']

LSTMmodel(time_step, units, units_1)



