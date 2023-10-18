import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional

# 读取CSV文件
data = pd.read_csv('data.csv')

# 提取需要的特征列
features = data[['date', 'dew', 'temp', 'press', 'wnd_spd', 'snow', 'pollution']]

# 归一化处理数据
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(features.values[:, -1:])  # 只保留pollution列

# 定义时间步长
time_steps = 10

# 构建时间序列数据集
X = []
y = []
for i in range(time_steps, len(scaled_data)):
    X.append(scaled_data[i-time_steps:i])
    y.append(scaled_data[i])

X = np.array(X)
y = np.array(y)

# 划分训练集和测试集
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 构建BiLSTM模型
model = Sequential()
model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(time_steps, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.summary()
# 训练模型
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# 模型评估
loss = model.evaluate(X_test, y_test)

# 反归一化处理
y_test = scaler.inverse_transform(y_test)
y_pred = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred)

# 可视化损失函数
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 反归一化后可视化真实值与预测值
plt.plot(y_test, label='True')
plt.plot(y_pred, label='Predicted')
plt.xlabel('Time')
plt.ylabel('Pollution')
plt.legend()
plt.show()

# 打印模型评价结果
# 计算评估指标
train_rmse = np.sqrt(mean_squared_error(y_train[0], train_predictions[:, 0]))
train_mae = mean_absolute_error(y_train[0], train_predictions[:, 0])
train_r2 = r2_score(y_train[0], train_predictions[:, 0])
test_rmse = np.sqrt(mean_squared_error(y_test[0], test_predictions[:, 0]))
test_mae = mean_absolute_error(y_test[0], test_predictions[:, 0])
test_r2 = r2_score(y_test[0], test_predictions[:, 0])
print('Test Loss:', loss)
