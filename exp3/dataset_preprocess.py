import pandas as pd

data = pd.read_csv('exp3_seeds_dataset.txt', header=None)
print(data.shape)

# separate dataset into data and label
data_columns = data.iloc[:, :7]
data_label = data.iloc[:, 7]

# save data and label to file
data_columns.to_csv('seeds_data.csv', index=False, header=False)
data_label.to_csv('seeds_label.csv', index=False, header=False)
