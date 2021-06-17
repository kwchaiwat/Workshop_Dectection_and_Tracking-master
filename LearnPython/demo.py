from sklearn import tree
import imp

#[height, weight, shoe size]
X = [[181,80,44], [177,70,43], [160,60,38], [154,54,37],[167,65,40],[190,90,47],
     [175,64,39],[167,50,37],[159,55,38],[171,75,42],[181,85,43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male',
     'male', 'female', 'female', 'male','male', ]

clf = tree.DecisionTreeClassifier()

clf = clf.fit(X,Y)

prediction = clf.predict([[190,70,43]])

print(prediction)