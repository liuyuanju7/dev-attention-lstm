import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from datetime import timedelta

# 设置随机种子，以确保结果可重复
np.random.seed(0)

# 定义时间范围和特征列
start_date = pd.Timestamp('2022-01-01')
end_date = start_date + timedelta(days=299)
features = ['NewIssueCount', 'IssueSolvedAvgTime', 'MRCount', 'DevDevelopCycleRatio', 'DemandDevScale', 'DemandThroughput']

# 生成随机数据
data = pd.DataFrame(columns=['Date'] + features)
data['Date'] = pd.date_range(start=start_date, end=end_date, freq='D')
data[features] = np.round(np.random.rand(len(data), len(features)) * 20, 1)  # 将数据范围调整为0-20之间，并保留1位小数

# 提取需预测的目标列
target_col = 'DemandThroughput'
target_data = data[target_col].values.reshape(-1, 1)

# 数据标准化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(target_data)

# 定义函数将数据集转换为监督学习问题
def create_dataset(data, lookback):
    X, y = [], []
    for i in range(len(data)-lookback):
        X.append(data[i:i+lookback, 0])
        y.append(data[i+lookback, 0])
    return np.array(X), np.array(y)

# 设置时间步长（lookback）和训练/测试集划分比例
lookback = 10
train_ratio = 0.8

# 划分训练集和测试集
train_size = int(len(scaled_data) * train_ratio)
train_data = scaled_data[:train_size, :]
test_data = scaled_data[train_size-lookback:, :]

# 创建训练集和测试集
X_train, y_train = create_dataset(train_data, lookback)
X_test, y_test = create_dataset(test_data, lookback)

# 调整输入数据的形状（样本数，时间步长，特征数）
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# 构建LSTM模型
model = Sequential()
model.add(LSTM(units=50, input_shape=(lookback, 1)))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# 训练模型
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# 模型预测
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# 反标准化预测结果
train_predictions = scaler.inverse_transform(train_predictions)
y_train = scaler.inverse_transform([y_train])
test_predictions = scaler.inverse_transform(test_predictions)
y_test = scaler.inverse_transform([y_test])

# 计算评估指标
train_rmse = np.sqrt(mean_squared_error(y_train[0], train_predictions[:, 0]))
train_mae = mean_absolute_error(y_train[0], train_predictions[:, 0])
train_r2 = r2_score(y_train[0], train_predictions[:, 0])
test_rmse = np.sqrt(mean_squared_error(y_test[0], test_predictions[:, 0]))
test_mae = mean_absolute_error(y_test[0], test_predictions[:, 0])
test_r2 = r2_score(y_test[0], test_predictions[:, 0])

# 可视化损失函数
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'])
plt.show()

# 可视化真实值与预测值
plt.plot(y_test[0])
plt.plot(test_predictions[:, 0])
plt.title('Actual vs Predicted')
plt.xlabel('Time')
plt.ylabel('Demand Throughput')
plt.legend(['Actual', 'Predicted'])
plt.show()

print('Train RMSE:', train_rmse)
print('Train MAE:', train_mae)
print('Train R2:', train_r2)
print('Test RMSE:', test_rmse)
print('Test MAE:', test_mae)
print('Test R2:', test_r2)
