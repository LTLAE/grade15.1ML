import pandas as pd

housing_dataset = pd.read_csv('exp1_housing_one_hot.csv')

# pick 30% of the data for testing
test_dataset = housing_dataset.sample(frac=0.3)
# save them to a new file
test_dataset.to_csv('exp1_test_dataset.csv', index=False)
# remove test data from original dataset
housing_dataset.drop(test_dataset.index, inplace=True)
# save the remaining data to training dataset
housing_dataset.to_csv('exp1_training_dataset.csv', index=False)
