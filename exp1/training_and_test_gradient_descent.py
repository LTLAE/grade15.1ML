import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

housing_training_dataset = pd.read_csv('exp1_training_dataset.csv')
# Check if there are any missing values in the dataset
# print(housing_training_dataset.isnull().sum())
# Remove rows with NaN
housing_training_dataset = housing_training_dataset.dropna()

# input:  [longitude,latitude,housing_median_age,total_rooms,total_bedrooms,population,households,median_income, ocean_proximity]
# output: [median_house_value]

input_features = housing_training_dataset[['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'NEAR BAY', '<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'ISLAND']]
target_feature = housing_training_dataset['median_house_value']

# add a column of ones to the input features
input_features = np.c_[np.ones(input_features.shape[0]), input_features]

# standardize using sklearn.preprocessing.StandardScaler()
input_features = StandardScaler().fit_transform(input_features)
target_scaler = StandardScaler()
target_feature = target_scaler.fit_transform(target_feature.values.reshape(-1, 1)).flatten()

def compute_cost(features, target, theta):
    m = len(target)
    return (1 / (2 * m)) * np.sum((input_features.dot(theta) - target) ** 2)


def gradient_descent(features, target, theta, learning_rate, iterations):
    cost_history = []  # record the cost in each iteration
    for _ in range(iterations):
        delta = input_features.dot(theta) - target  # delta = predicted - actual
        gradients = (1 / len(target)) * input_features.T.dot(delta)  # calculate the gradients
        theta -= learning_rate * gradients  # update theta
        cost_history.append(compute_cost(features, target, theta))  # record this time's cost
    return theta, cost_history


the_theta = np.zeros(input_features.shape[1])
learning_rate = 0.01
iterations = 2000

theta_optimal, cost_history = gradient_descent(input_features, target_feature, the_theta, learning_rate, iterations)
print(theta_optimal)

# save theta_optimal to a file
np.savetxt('theta_optimal.csv', theta_optimal)

# cost convergence graph drawing code by chatGPT
plt.plot(cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost (MSE)")
plt.title("Cost Convergence Over Iterations")
plt.show()

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

# Standardize the test features
test_features = StandardScaler().fit_transform(test_features)
test_target = target_scaler.transform(test_target.values.reshape(-1, 1)).flatten()


# Import beta from file
theta = np.loadtxt('theta_optimal.csv')

# Predict the target values
test_predictions = test_features.dot(theta)

# Calculate R^2
r2 = r2_score(test_target, test_predictions)
print(r2)

features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'NEAR BAY', '<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'ISLAND']
coefficients = theta_optimal[1:]

plt.figure(figsize=(10, 6))
plt.barh(features, coefficients, color=['blue' if coef > 0 else 'red' for coef in coefficients])
plt.title("Positive and negative coefficients of the features")
plt.show()
