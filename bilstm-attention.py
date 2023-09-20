import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential, Model
from keras.layers import Bidirectional, LSTM, Dense, Input, Concatenate, Activation, Dot

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

# 构建BiLSTM模型
input_shape = (time_steps, 1)
hidden_units = 50

# 输入层
input_layer = Input(shape=input_shape)

# 双向LSTM层
lstm_layer = Bidirectional(LSTM(hidden_units, return_sequences=True))(input_layer)

# 注意力机制
attention = Dense(1, activation='tanh')(lstm_layer)
attention = Activation('softmax')(attention)
attention = Dot(axes=1)([attention, lstm_layer])

# 输出层
output_layer = Dense(1)(attention)

# 构建模型
model = Model(inputs=input_layer, outputs=output_layer)
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()
# 训练模型
history = model.fit(train_X, train_Y, epochs=50, batch_size=32, validation_data=(test_X, test_Y), verbose=2)

# 预测
train_predict = model.predict(train_X)
test_predict = model.predict(test_X)

# 反归一化处理
train_predict = scaler.inverse_transform(train_predict.reshape(-1, 1))
train_Y = scaler.inverse_transform(train_Y.reshape(-1, 1))
test_predict = scaler.inverse_transform(test_predict.reshape(-1, 1))
test_Y = scaler.inverse_transform(test_Y.reshape(-1, 1))

# 评估模型
train_rmse = np.sqrt(mean_squared_error(train_Y, train_predict))
train_rmae = mean_absolute_error(train_Y, train_predict)
train_r2 = r2_score(train_Y, train_predict)

test_rmse = np.sqrt(mean_squared_error(test_Y, test_predict))
test_rmae = mean_absolute_error(test_Y, test_predict)
test_r2 = r2_score(test_Y, test_predict)

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
attention_weights = model.layers[3].get_weights()[0]
plt.bar(range(time_steps), attention_weights[:, 0])
plt.xlabel('Time Step')
plt.ylabel('Attention Weight')
plt.show()

# 反归一化处理后可视化真实值与预测值
plt.plot(train_Y, label='Actual')
plt.plot(train_predict, label='Predicted')
plt.xlabel('Time')
plt.ylabel('Pollution')
plt.legend()
plt.show()

plt.plot(test_Y, label='Actual')
plt.plot(test_predict, label='Predicted')
plt.xlabel('Time')
plt.ylabel('Pollution')
plt.legend()
plt.show()
