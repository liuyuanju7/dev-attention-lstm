import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense

# 加载数据
data = pd.read_csv('data.csv')

# 提取特征和目标列
features = data.iloc[:, 1:7].values
target = data.iloc[:, -1].values

# 归一化处理
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)
scaled_target = scaler.fit_transform(target.reshape(-1, 1))

# 定义函数将时间序列数据转换为监督学习数据集
def series_to_supervised(data, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(data)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        if out_end_ix > len(data):
            break
        seq_x, seq_y = data[i:end_ix, :], data[end_ix:out_end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# 定义参数
n_steps_in = 5  # 输入的时间步长
n_steps_out = 4  # 输出的时间步长

# 将数据转化为监督学习数据集
X, y = series_to_supervised(np.concatenate((scaled_features, scaled_target), axis=1), n_steps_in, n_steps_out)

# 划分训练集和测试集
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 定义模型
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps_in, X_train.shape[2])))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')

# 训练模型
history = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# 可视化损失函数
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

# 预测
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# 反归一化
y_train_pred = scaler.inverse_transform(y_train_pred)
y_test_pred = scaler.inverse_transform(y_test_pred)
y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# 可视化真实值和预测值
plt.plot(y_train_actual, label='Actual')
plt.plot(y_train_pred, label='Predicted')
plt.title('Training Set: Actual vs Predicted')
plt.xlabel('Time')
plt.ylabel('Pollution')
plt.legend()
plt.show()

plt.plot(y_test_actual, label='Actual')
plt.plot(y_test_pred, label='Predicted')
plt.title('Testing Set: Actual vs Predicted')
plt.xlabel('Time')
plt.ylabel('Pollution')
plt.legend()
plt.show()

# 输出模型评价结果
train_rmse = np.sqrt(mean_squared_error(y_train_actual, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_pred))
print("Train RMSE: %.3f" % train_rmse)
print("Test RMSE: %.3f" % test_rmse)
