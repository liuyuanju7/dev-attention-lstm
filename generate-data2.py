import pandas as pd
import numpy as np

# 设置数据长度和每段上升数据的条数
data_length = 1200
segment_length = 20

# 生成第一个数据
data1 = pd.DataFrame({
    'Value': np.zeros(data_length)
})

# 生成第二个数据，基于第一个数据的上升趋势
data2 = pd.DataFrame({
    'Value': np.zeros(data_length)
})

for i in range(data_length // segment_length):
    # 生成每段上升数据
    segment_data = np.random.randint(0, 21, size=segment_length)

    # 更新第一个数据
    data1['Value'][i * segment_length:(i + 1) * segment_length] = segment_data

    # 更新第二个数据，基于第一个数据的上升趋势
    trend = np.random.normal(0, 5)  # 随机生成趋势
    data2['Value'][i * segment_length:(i + 1) * segment_length] = segment_data + trend

# 保存为CSV文件
data1.to_csv('data1.csv', index=False)
data2.to_csv('data2.csv', index=False)
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
data1 = pd.read_csv('data1.csv')
data2 = pd.read_csv('data2.csv')

# 绘制数据
plt.plot(data1['Value'], label='Data 1')
plt.plot(data2['Value'], label='Data 2')

# 添加图例和标签
plt.legend()
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Comparison of Data 1 and Data 2')

# 显示图形
plt.show()
