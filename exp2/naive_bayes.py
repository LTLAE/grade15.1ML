import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

iris_training_dataset = pd.read_csv('iris_training_dataset.csv')
train_features = iris_training_dataset[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
train_target = iris_training_dataset['class']

# calculate class prior probabilities
class_counts = train_target.value_counts()
class_priors = {class_id: count / len(train_target) for class_id, count in class_counts.items()}
print("Class priors: ", class_priors)

# calculate class means and variances
class_means = train_features.groupby(train_target).mean()
class_variances = train_features.groupby(train_target).var()
print("Class means: \n", class_means)
print("Class variances: \n", class_variances)

# use means and variances to predict
iris_test_dataset = pd.read_csv('iris_test_dataset.csv')
test_features = iris_test_dataset[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
test_features = pd.DataFrame(test_features, columns=['bias', 'sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
test_target = iris_test_dataset['class']

# calculate probabilities
predictions = []    # used to store the predicted class labels
for _, sample in test_features.iterrows():
    class_probabilities = {}
    # calculate the posterior probability for each class
    for class_label, class_prior in class_priors.items():
        class_prob = class_prior

        # calculate the likelihood each feature in class
        for feature in train_features.columns:
            mean = class_means.loc[class_label, feature]
            var = class_variances.loc[class_label, feature]
            class_prob *= 1 / np.sqrt(2 * np.pi * var) * np.exp(-0.5 * (sample[feature] - mean) ** 2 / var)

        class_probabilities[class_label] = class_prob

    # pick result with the highest probability
    predicted_class = max(class_probabilities, key=class_probabilities.get)
    predictions.append(predicted_class)

# calculate accuracy
acc = accuracy_score(test_target, predictions)
print("Accuracy: ", acc)

# calculate f1 score
f1 = f1_score(test_target, predictions, average='weighted')
print("F1 score: ", f1)
