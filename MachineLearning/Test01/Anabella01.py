from sklearn import tree


#creating two varibles to create train my classifier

#taking features as the input for the classifier
#features = [[140, "smooth"], [140, "smooth"], [170, "Bumpy"], [150, "Bumpy"]]
features = [[140, 1], [140, 1], [170, 0], [150, 0]]

#taking labels as the output we want
#labels = ["apple", "apple", "orange", "orange"]
labels = [0, 0, 1, 1]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
ans = clf.predict([[150, 0]])

if ans == 1:
    print("Orange")
if ans == 0 :
    print("Apple")

