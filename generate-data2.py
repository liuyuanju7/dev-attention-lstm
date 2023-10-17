import pandas as pd
import numpy as np

# 设置数据长度和周期
data_length = 1193
cycle_length = np.random.randint(15, 21)

# 生成周期性数据
data = pd.DataFrame({
    'Value': np.random.randint(0, 21, size=data_length)
})

# 添加周期性
data['Cycle'] = np.repeat(range(data_length // cycle_length + 1), cycle_length)[:data_length]

# 保存为CSV文件
data.to_csv('periodic_data.csv', index=False)
