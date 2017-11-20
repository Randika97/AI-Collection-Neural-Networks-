import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()

# for this removed one sample from each type of flower
test_data = [0, 50, 100]

# training data
train_target = np.delete(iris.target, test_data)
train_data = np.delete(iris.data, test_data, axis=0)

# testing data
test_target = iris.target[test_data]
test_data = iris.data[test_data]

# making a decision tree clissifier
cli = tree.DecisionTreeClassifier()
# trainig the clissifier using our training data
cli.fit(train_data, train_target)

# print our testing data
print(test_target)

# printing predicted data
print(cli.predict(test_data))

#Vizualitation code

