from sklearn.datasets import load_iris

#simply loading the data set into our program
iris = load_iris()

#feature contains the features of the data set
features =  iris.feature_names
#target contains the labels
labels = iris.target_names

print("features ",features)
print("labels ",labels)

#printing features of the data set
print(iris.data[0])

#print labels of the data set
print(iris.target[0])

#print whole data set using a for loop
for index in range(len(iris.target)):
    print("Example %d: label %s, features %s" % (index, iris.target[index], iris.data[index]))