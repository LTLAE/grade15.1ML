import pandas as pd

iris_dataset = pd.read_csv('bezdekIris.data')

# flower class -> num
# Iris-setosa -> 0
# Iris Iris-versicolor -> 1
# Iris Virginica -> 2

iris_dataset['class'] = iris_dataset['class'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})

# pick 20% -> testing set
# don't separate them randomly, select them in each class
setosa_test = iris_dataset.iloc[:50].sample(frac=0.2, random_state=114514)
versicolor_test = iris_dataset.iloc[50:100].sample(frac=0.2, random_state=114514)
virginica_test = iris_dataset.iloc[100:150].sample(frac=0.2, random_state=114514)

test_dataset = pd.concat([setosa_test, versicolor_test, virginica_test])
test_dataset.to_csv('iris_test_dataset.csv', index=False)

# save remaining -> training set
iris_dataset.drop(test_dataset.index, inplace=True)
iris_dataset.to_csv('iris_training_dataset.csv', index=False)
