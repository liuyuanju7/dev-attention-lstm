import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional

# 读取数据
data = pd.read_csv('./dataset/pollution.csv')

# 提取需要的列并进行归一化处理
dataset = data[['pollution']].values.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# 将数据集划分为训练集和测试集
train_size = int(len(dataset) * 0.7)
train, test = dataset[:train_size], dataset[train_size:]

# 创建用于时序预测的数据集
def create_dataset(dataset, time_steps=1):
    X, Y = [], []
    for i in range(len(dataset)-time_steps):
        X.append(dataset[i:i+time_steps])
        Y.append(dataset[i+time_steps])
    return np.array(X), np.array(Y)

time_steps = 10  # 可灵活控制的时间步长
train_X, train_Y = create_dataset(train, time_steps)
test_X, test_Y = create_dataset(test, time_steps)

# 构建LSTM模型
model = Sequential()
model.add(Bidirectional(LSTM(50, input_shape=(time_steps, 1))))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
# 训练模型
history = model.fit(train_X, train_Y, epochs=10, batch_size=32, validation_data=(test_X, test_Y), verbose=2)
model.summary()
# 预测
train_predict = model.predict(train_X)
test_predict = model.predict(test_X)

# 反归一化处理
train_predict = scaler.inverse_transform(train_predict.reshape(-1, 1))
train_Y = scaler.inverse_transform(train_Y.reshape(-1, 1))
test_predict = scaler.inverse_transform(test_predict.reshape(-1, 1))
test_Y = scaler.inverse_transform(test_Y.reshape(-1, 1))


# 评估模型
# 评估模型
train_rmse = np.sqrt(mean_squared_error(train_Y.reshape(-1), train_predict.reshape(-1)))
train_rmae = mean_absolute_error(train_Y.reshape(-1), train_predict.reshape(-1))
train_r2 = r2_score(train_Y.reshape(-1), train_predict.reshape(-1))

test_rmse = np.sqrt(mean_squared_error(test_Y.reshape(-1), test_predict.reshape(-1)))
test_rmae = mean_absolute_error(test_Y.reshape(-1), test_predict.reshape(-1))
test_r2 = r2_score(test_Y.reshape(-1), test_predict.reshape(-1))

print("Train RMSE:", train_rmse)
print("Train RMAE:", train_rmae)
print("Train R2:", train_r2)
print("Test RMSE:", test_rmse)
print("Test RMAE:", test_rmae)
print("Test R2:", test_r2)


# 可视化损失函数
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 可视化注意力权重
attention_weights = model.layers[0].get_weights()[0]
plt.bar(range(time_steps), attention_weights[:, 0])
plt.xlabel('Time Step')
plt.ylabel('Attention Weight')
plt.show()

# 可视化真实值与预测值
plt.plot(train_Y[0], label='Actual')
plt.plot(train_predict[:,0], label='Predicted')
plt.xlabel('Time')
plt.ylabel('Pollution')
plt.legend()
plt.show()

plt.plot(test_Y[0], label='Actual')
plt.plot(test_predict[:,0], label='Predicted')
plt.xlabel('Time')
plt.ylabel('Pollution')
plt.legend()
plt.show()
