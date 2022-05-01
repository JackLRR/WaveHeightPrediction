# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from keras.layers import Dense
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
import math
from tcn import TCN
from tensorflow.python.keras.callbacks import EarlyStopping

dataset = pd.read_csv("../41008.csv", header=0)

# 预测间隔
time_interval = 24

# 自定义变量
n_feature = 1   #特征数
time_step = 24   #时间步
learning_rate = 0.001
batch_size = 64
epochs = 100

data = dataset.iloc[:len(dataset), 5:6].values

x = []
y = []

for i in range(time_step, len(data)-time_interval+1):
    x.append(data[i-time_step:i])
    y.append(data[i+time_interval-1])

x = np.array(x)
y = np.array(y)
a = int(y.shape[0])
x = np.reshape(x, (a, time_step))

# 数据集比例
train_volume = int(len(data)*0.64)
val_volume = int(len(data)*0.16)
test_volume = len(data)-train_volume-val_volume

train = data[:train_volume, :]
val = data[train_volume:train_volume+val_volume, :]
test = data[-test_volume:, :]


# 划分数据集
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
model = tf.keras.models.Sequential([
    TCN(nb_filters=24, input_shape=(24, 1)),
    Dense(24),
    Dense(1)
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='mean_squared_error',  metrics=['mae'])
cp_callback = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 训练模型
history = model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, validation_data=(val_x, val_y),
                        validation_freq=1, callbacks=cp_callback)
model.summary()

train_loss = history.history['loss']
val_loss = history.history['val_loss']

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
mape = np.mean(np.abs((test_y - predicted_wave_height) / test_y))
R, p = pearsonr(test_y, predicted_wave_height)
R2 = r2_score(test_y, predicted_wave_height)

print('mse: %.6f' % mse)
print('rmse: %.6f' % rmse)
print('mae: %.6f' % mae)
print('mape: %.4f' % mape)
print("R:", R)
print("R2:", R2)

# loss曲线图
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# 预测效果图
fig = plt.figure(figsize=(20, 5))
plt.plot(test_y[500:1000], color='red', label='real_wave_height')
plt.plot(predicted_wave_height[500:1000], color='blue', label='predicted_wave_height')
plt.title('Wave Height Prediction')
plt.xlabel('Time')
plt.ylabel('Wave Height')
plt.legend()
plt.show()
