import numpy as np
import pandas as pd
from datetime import timedelta

# 设置随机种子，以确保结果可重复
np.random.seed(0)

# 定义时间范围和特征列
start_date = pd.Timestamp('2023-01-01')
end_date = start_date + timedelta(days=299)
features = ['NewIssueCount', 'IssueSolvedAvgTime', 'MRCount', 'DevDevelopCycleRatio', 'DemandDevScale', 'DemandThroughput']

# 生成随机数据
data = pd.DataFrame(columns=['Date'] + features)
data['Date'] = pd.date_range(start=start_date, end=end_date, freq='D')

# 生成具有线性关系的数据
for feature in features:
    x = np.linspace(0, 1, len(data))
    y = 5 * x + np.random.normal(0, 1, len(data))
    data[feature] = y

# 设置示例数据
example_data = pd.DataFrame({
    'Date': [pd.Timestamp('2023-01-01')],
    'NewIssueCount': [16],
    'IssueSolvedAvgTime': [5.19],
    'MRCount': [3],
    'DevDevelopCycleRatio': [80],
    'DemandDevScale': [23],
    'DemandThroughput': [6]
})

# 将示例数据添加到生成的数据中
data = pd.concat([example_data, data], ignore_index=True)

# 保存数据到CSV文件
data.to_csv('timeseries_data.csv', index=False)
