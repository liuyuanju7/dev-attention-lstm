import numpy as np
import pandas as pd
from datetime import timedelta

# 设置随机种子，以确保结果可重复
np.random.seed(0)

# 定义时间范围和特征列
start_date = pd.Timestamp('2022-01-01')
end_date = start_date + timedelta(days=3650)
features = ['NewIssueCount', 'IssueSolvedAvgTime', 'MRCount', 'DevDevelopCycleRatio', 'DemandDevScale', 'DemandThroughput']

# 生成随机数据
data = pd.DataFrame(columns=['Date'] + features)
data['Date'] = pd.date_range(start=start_date, end=end_date, freq='D')

# 生成具有线性关系的数据
for feature in features:
    if feature == 'DemandThroughput':
        # 生成具有线性关系的特征列，并添加一些随机噪音
        x = np.linspace(0, 1, len(data))
        y = 2 * x + np.random.normal(0, 0.5, len(data))
        data[feature] = y
    else:
        # 生成随机的非目标特征列
        data[feature] = np.random.rand(len(data))

# 保存数据到CSV文件
data.to_csv('timeseries_data.csv', index=False)
