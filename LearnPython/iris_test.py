from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import imp

iris = datasets.load_iris() #ดึงข้อมูล iris มา

#วิธีเข้าถึง bunch iris.data หรือ iris['data']
#iris.target ไว้ดู target
#iris.data.shape iris.target.shape ไว้ดูข้อมูลว่ามีกี่ row กี่ colum

#การ Train/Test Splitting
#Train for ML Learn
#Test for ML ทำนาย target

x = iris.data
y = iris.target

# test_size = 0.2 คือจำนวนข้อมูลที่ใช้ test = 20% train = 80%
# random_state ค่า seed ที่ random กลุ่มข้อมูล
# stratify การกำหนดการกระจายของข้อมูลที่ split ให้มีการกระจายเหมือน original dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=12, stratify=y)

#ใช้ตัวแบบ K-nearest Neighbors
MrRobot = KNeighborsClassifier(n_neighbors=7)
MrRobot.fit(x_train, y_train)

#prediction
prediction = MrRobot.predict(x_test)
print(prediction)

#Check Accuracy
print(MrRobot.score(x_test, y_test))
print(x_test)
print(y_test)