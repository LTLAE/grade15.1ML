import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score

housing_test_dataset = pd.read_csv('exp1_test_dataset.csv')
# Check if there are any missing values in the dataset
# print(housing_training_dataset.isnull().sum())
# Fill the missing values
housing_test_dataset.fillna(housing_test_dataset.mean(), inplace=True)

# Extract features
test_features = housing_test_dataset[['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'NEAR BAY', '<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'ISLAND']]
test_target = housing_test_dataset['median_house_value']
# Add a column of 1 before the first column in input features
test_features = np.c_[np.ones(test_features.shape[0]), test_features]

# Import beta from file
beta = np.loadtxt('beta.csv')

# Predict the target values
test_predictions = test_features.dot(beta)

# Calculate R^2
r2 = r2_score(test_target, test_predictions)
print(r2)

features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'NEAR BAY', '<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'ISLAND']
coefficients = beta[1:]

plt.figure(figsize=(10, 6))
plt.barh(features, coefficients, color=['blue' if coef > 0 else 'red' for coef in coefficients])
plt.title("Positive and negative coefficients of the features")
plt.show()