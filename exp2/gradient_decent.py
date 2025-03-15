import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

iris_training_dataset = pd.read_csv('iris_training_dataset.csv')
train_features = iris_training_dataset[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
train_target = iris_training_dataset['class']

# add a column of ones to the input features
train_features = np.c_[np.ones(train_features.shape[0]), train_features]


# sigmoid function, compute cost and gradient
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def compute_cost(features, target, theta):
    h = sigmoid(features.dot(theta))
    return (-1 / len(target)) * (target.dot(np.log(h)) + (1 - target).T.dot(np.log(1 - h)))


def compute_gradient(features, target, theta):
    h = sigmoid(features.dot(theta))
    return (1 / len(target)) * features.T.dot(h - target)


# train 3 models for 3 classes, separating *this* class with others
def this_vs_rest(features, target, num_classes, learning_rate, iterations):
    m, n = features.shape
    all_theta = np.zeros((num_classes, n))

    for timer in range(num_classes):
        # this class: 1, rest: 0
        y_binary = np.where(target == timer, 1, 0)
        theta = np.zeros(n)

        # gradient decent
        for _ in range(iterations):
            gradient = compute_gradient(features, y_binary, theta)
            theta -= learning_rate * gradient

        # save all theta(s)
        all_theta[timer] = theta
    return all_theta


# train the model
learning_rate = 0.01
iterations = 1000
num_classes = 3

all_theta = this_vs_rest(train_features, train_target, num_classes, learning_rate, iterations)

print("All_theta:\n", all_theta)

# use all_theta to predict
iris_test_dataset = pd.read_csv('iris_test_dataset.csv')
test_features = iris_test_dataset[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
test_features = np.c_[np.ones(test_features.shape[0]), test_features]
test_target = iris_test_dataset['class']

probabilities = sigmoid(test_features.dot(all_theta.T))
# predictions of each column is stored in predictions[]
predictions = np.argmax(probabilities, axis=1)

# calculate accuracy
accuracy = np.mean(predictions == test_target) * 100
print("Accuracy: ", accuracy)

# calculate F1score
f1 = f1_score(test_target, predictions, average='macro')
print("F1 Score: ", f1)
