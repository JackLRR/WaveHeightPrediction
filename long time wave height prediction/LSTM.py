# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from tensorflow.keras.layers import Dropout, Dense, LSTM, BatchNormalization
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
import math
from scipy.stats import pearsonr
from keras import regularizers
from tensorflow.python.keras.callbacks import EarlyStopping

dataset = pd.read_csv("../41008.csv", header=0)

# 预测间隔
time_interval = 24

# 自定义变量
n_feature = 1   #特征数
time_step = 24   #时间步
cell = 24        #神经元数
learning_rate = 0.001
batch_size = 64
epochs = 100

data = dataset.iloc[:len(dataset), 5:6].values

# 重组数据
x = []
y = []
for i in range(time_step, len(data)-time_interval+1):
    x.append(data[i-time_step:i])
    y.append(data[i+time_interval-1])
x = np.array(x)
y = np.array(y)

# 数据集比例
train_volume = int(len(x) * 0.64)
val_volume = int(len(x) * 0.16)
test_volume = len(x) - train_volume - val_volume
test_num = test_volume

# 划分输入输出
train_x = x[:train_volume, :]
val_x = x[train_volume:train_volume+val_volume, :]
test_x = x[-test_volume:, :]
train_y = y[:train_volume]
val_y = y[train_volume:train_volume+val_volume]
test_y = y[-test_volume:]

# 将特征转换为三维
train_x, train_y = np.array(train_x), np.array(train_y)
train_x = np.reshape(train_x, (train_x.shape[0], time_step, n_feature))
val_x, val_y = np.array(val_x), np.array(val_y)
val_x = np.reshape(val_x, (val_x.shape[0], time_step, n_feature))
test_x, test_y = np.array(test_x), np.array(test_y)
test_x = np.reshape(test_x, (test_x.shape[0], time_step, n_feature))
test_y = np.array(test_y)

# 创建模型
model = tf.keras.Sequential([
     LSTM(cell, activation='tanh', input_shape=(time_step, n_feature), return_sequences=False),
     Dense(cell),
     Dense(1)
    ])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='mean_absolute_error',
                  metrics=['mae'])
cp_callback = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 训练模型
history = model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, validation_data=(val_x, val_y),
                        validation_freq=1, callbacks=cp_callback)
model.summary()

# 提取模型预测值
predicted_data = model.predict(test_x)

predicted_wave = []
for i in predicted_data:
    for j in i:
        predicted_wave.append(j)

predicted_wave_height = []
for i in range(len(predicted_wave)):
    predicted_wave_height.append(predicted_wave[i])

# 评价模型
mse = mean_squared_error(predicted_wave_height, test_y)
rmse = math.sqrt(mean_squared_error(predicted_wave_height, test_y))
mae = mean_absolute_error(predicted_wave_height, test_y)
R, p = pearsonr(test_y, predicted_wave_height)
R2 = r2_score(test_y, predicted_wave_height)

# 评价模型
mse = mean_squared_error(predicted_wave_height, test_y)
rmse = math.sqrt(mean_squared_error(predicted_wave_height, test_y))
mae = mean_absolute_error(predicted_wave_height, test_y)
mape = np.mean(np.abs((test_y - predicted_wave_height) / test_y))
R, p = pearsonr(test_y, predicted_wave_height)
R2 = r2_score(test_y, predicted_wave_height)

print('mse: %.4f' % mse)
print('rmse: %.4f' % rmse)
print('mae: %.4f' % mae)
print('mape: %.4f' % mape)
print("R: %.4f" % R)
print("R2: %.4f" % R2)

# 预测效果图
fig = plt.figure(figsize=(20, 5))
plt.plot(test_y[0:800], color='red', label='real')
plt.plot(predicted_wave_height[0:800], color='blue', label='EMD-LSTM')
plt.title('2014.11.30 2:50-2015.1.2 9:50')
plt.xlabel('Time Serise')
plt.ylabel('Wave height')
plt.legend()
plt.show()

