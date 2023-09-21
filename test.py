import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 读取CSV文件
data = pd.read_csv('./dataset/pollution-simple.csv')

# 提取 pollution 列的数据
pollution_data = data['pollution'].values.reshape(-1, 1)

# 创建 MinMaxScaler 对象并进行归一化
scaler = MinMaxScaler(feature_range=(0, 20))
scaled_data = scaler.fit_transform(pollution_data)

# 将归一化后的数据四舍五入为整数
rounded_data = scaled_data.round().astype(int)


# 将归一化后的数据替换原始数据中的 pollution 列
data['pollution'] = rounded_data

# 将修改后的数据保存到新的CSV文件
data.to_csv('normalized_data.csv', index=False)
