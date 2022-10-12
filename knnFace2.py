from cgi import test
from email.mime import image
import cv2
import os
import numpy as np
#import pandas as pd

from sklearn.neighbors import KNeighborsClassifier

# import hình ảnh làm dữ liệu

image_path = 'archive'
train = []
validate = []
ytest = []

#whatelse là các thư mục s1,s2,.....
for whatelse in os.listdir(image_path):
    i = 0
    whatelse_path = os.path.join(image_path,whatelse)
    for sub_watelse in os.listdir(whatelse_path):
        # vào trong lấy hình các thư mục s1, s2,.... 
            i+=1
            img_path = os.path.join(whatelse_path,sub_watelse) 
            # đọc các hình trong thư mục
            img = cv2.imread(img_path,0)
            img = np.reshape(img,-1)
            
            # thêm dữ liệu vào trong train, validate, test
            # train có dạng [([10,4,...,3,2], "archive\\s1"), ( giống ngoặc đầu),]
            if (i == 9 ):
                validate.append((img,whatelse_path))
            elif (i == 10):
                ytest.append((img,whatelse_path))
            else:
                train.append((img,whatelse_path))
                


# chia train từ ([10,4,...,3,2], "archive\\s1") sang x_train = [10,4,...,3,2] và y_train = "archive\\s1"
x_train =[]
y_train =[]
for i in train:
    x_train.append(i[0])
    y_train.append(i[1])

# giống x_train, y_train bên trên
x_validate= []
y_validate= []
for i in validate:
    x_validate.append(i[0])
    y_validate.append(i[1])


# gọi mô hình KNN và huấn luyện
classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(x_train, y_train)
#print(validate[7])

# Dự đoán trên tập validate

y_pred = classifier.predict(x_validate)
#print('predict knn:',y_pred)

#tính độ 9 xác 
from sklearn.metrics import accuracy_score
ac = accuracy_score(y_validate,y_pred)
print('do 9 xac knn : ', ac)

from sklearn import svm
clf = svm.SVC()
clf.fit(x_train, y_train)
#y_pred2 = clf.predict(np.reshape(validate[7][0],(1,-1)))
#print('predict svm:',y_pred2)
y_pred2 = clf.predict(x_validate)
ac2 = accuracy_score(y_validate,y_pred2)
print('do 9 xac svm : ', ac2)


from sklearn import linear_model
logr = linear_model.LogisticRegression()
logr.fit(x_train,y_train)
y_pred3 = logr.predict(x_validate)
ac3 = accuracy_score(y_validate,y_pred3)
print('do 9 xac logic regr : ', ac3)


