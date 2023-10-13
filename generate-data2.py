import pandas as pd
import numpy as np
from datetime import datetime, timedelta

input_dim = 6
num_samples = 1000

# Generate feature data
features = np.random.uniform(low=0, high=20, size=(num_samples, input_dim))
feature_names = ['dew', 'temp', 'press', 'wnd_spd', 'snow', 'pollution']

# Generate label data
weights = np.array([3, -2, 1, 0.5, -1.5, 2])  # Feature weights
bias = 1  # Bias
labels = np.dot(features, weights) + bias

# Scale labels to the range of 0-10
min_label = np.min(labels)
max_label = np.max(labels)
scaled_labels = (labels - min_label) / (max_label - min_label) * 10

# Generate date column
start_date = datetime.now().date()
dates = [start_date - timedelta(days=i) for i in range(num_samples)]

# Create DataFrame
data = pd.DataFrame(features, columns=feature_names)
data['date'] = dates
data['label'] = scaled_labels

# Save DataFrame to CSV
data.to_csv('dataset.csv', index=False)
