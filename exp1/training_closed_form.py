import pandas as pd
import numpy as np

housing_training_dataset = pd.read_csv('exp1_training_dataset.csv')
# Check if there are any missing values in the dataset
# print(housing_training_dataset.isnull().sum())
# Fill the missing values
housing_training_dataset.fillna(housing_training_dataset.mean(), inplace=True)

# input:  [longitude,latitude,housing_median_age,total_rooms,total_bedrooms,population,households,median_income, ocean_proximity]
# output: [median_house_value]

input_features = housing_training_dataset[['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'NEAR BAY', '<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'ISLAND']]
target_feature = housing_training_dataset['median_house_value']

# Add a column of 1 before the first column in input features
input_features = np.c_[np.ones(input_features.shape[0]), input_features]

# Find beta, the vector of coefficients
# beta = (input^T * input)^-1 * input^T * target
beta = np.linalg.pinv(input_features.T.dot(input_features)).dot(input_features.T).dot(target_feature)
print(beta)

# Save beta to file
np.savetxt('beta.csv', beta)
