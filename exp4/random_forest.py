import pandas as pd
import numpy as np
from collections import Counter

# random seed to reproduce the result
np.random.seed(114514)

# read training set and process
training_set = pd.read_csv('training_set.csv', header=None)
training_set = training_set.iloc[:, 1:]
training_target = training_set.iloc[:, 0]
training_features = training_set.iloc[:, 1:]

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        if len(np.unique(y)) == 1 or (self.max_depth and depth >= self.max_depth):
            return Counter(y).most_common(1)[0][0] # return most voted

        # ramdomly select a feature
        feature_idx = np.random.choice(X.shape[1], 1, replace=False)[0]
        thresholds = np.unique(X[:, feature_idx])

        if len(thresholds) == 1:
            return Counter(y).most_common(1)[0][0]  # return most voted

        # select best threshold
        threshold = np.median(X[:, feature_idx])
        left_idx = X[:, feature_idx] <= threshold
        right_idx = X[:, feature_idx] > threshold

        if sum(left_idx) == 0 or sum(right_idx) == 0:
            return Counter(y).most_common(1)[0][0]

        return {
            'feature': feature_idx,
            'threshold': threshold,
            'left': self._build_tree(X[left_idx], y[left_idx], depth + 1),
            'right': self._build_tree(X[right_idx], y[right_idx], depth + 1)
        }

    def predict_row(self, row, node):
        if isinstance(node, dict):
            if row[node['feature']] <= node['threshold']:
                return self.predict_row(row, node['left'])
            else:
                return self.predict_row(row, node['right'])
        else:
            return node

    def predict(self, X):
        return np.array([self.predict_row(row, self.tree) for row in X])

# main random forest
class RandomForestClassifier:
    def __init__(self, n_estimators=10, max_depth=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_estimators):
            # generate bootstrap sample
            idxs = np.random.choice(len(X), len(X), replace=True)
            X_sample, y_sample = X[idxs], y[idxs]

            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        # predictions of all trees
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        # vote
        return np.array([Counter(tree_predictions[:, i]).most_common(1)[0][0] for i in range(X.shape[0])])

# train
rf = RandomForestClassifier(n_estimators=10, max_depth=5)
rf.fit(training_features.values, training_target.values)

# predict
test_set = pd.read_csv('test_set.csv', header=None)
test_target = test_set.iloc[:, 0]
test_features = test_set.iloc[:, 1:]

predictions = rf.predict(test_features.values)
print(predictions)
predictions_binary = (predictions < 11.8).astype(int)
print("Predictions:\n", predictions_binary)

# calculate acc
acc = np.mean(predictions_binary == test_target)
print("Accuracy:", acc)

# calculate AUC
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(test_target, predictions_binary)
print("AUC:", auc)

