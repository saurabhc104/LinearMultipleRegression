# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 01:10:26 2017

@author: Saurabh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation
from matplotlib import style
style.use('ggplot')
df = pd.read_excel("D:\Python_practice\Blood_Pressure.xls")

print(df.head())
print()
print()

x1 = np.array(df.X3)
x2 = np.array(df.X2)
y = np.array(df.X1)

Scaled_X1 = preprocessing.scale(x1)
Scaled_X2 = preprocessing.scale(x2)
z = pd.DataFrame()
z['x1'] = Scaled_X1
z['x2'] = Scaled_X2
Y = preprocessing.scale(y)

#Taking 75% of Scaled_X and Scaled_Y for training and rest 25% for testing purpose
X_train , X_test ,  Y_train , Y_test = cross_validation.train_test_split(z,Y,test_size=0.25 )


#Now here comes linear regression
clf = LinearRegression()
clf.fit(X_train,Y_train)  #Training


#For 3D plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_train.x1,X_train.x2,Y_train, c='b', marker='*')
ax.set_xlabel('X2')
ax.set_ylabel('X3')
ax.set_zlabel('Pressure')
ax.scatter(X_train.x1,X_train.x2,clf.predict(X_train), c='g', marker='+')


accuracy = clf.score(X_test,Y_test)
print("Accuracy:",accuracy*100,"%")


X, Y = np.meshgrid(X_train.x1, X_train.x2)
ax.plot_surface(X,Y,clf.predict(X_train)) #perfect plane is not gonna printed..but you can visualize it.


