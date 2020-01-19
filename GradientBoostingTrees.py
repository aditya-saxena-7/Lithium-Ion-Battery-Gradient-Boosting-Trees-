# -*- coding: utf-8 -*-
"""
Created on Fri Dec  15 21:58:04 2019

@author: Aditya Saxena
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf


dataset = pd.read_csv('Lithium-ion-battery-dataset.csv')
X = dataset.iloc[:, [6,7,8,9,13,14]].values
y = dataset.iloc[:, 5].values
X = np.nan_to_num(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.ensemble import GradientBoostingClassifier

lr_list = [0.02,0.05,0.075,0.1,0.25,0.5,0.75,1]

for learning_rate in lr_list:
    gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate = learning_rate)
    gb_clf.fit(X_train,y_train.values.ravel())
    
    
    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(gb_clf.score(X_train, y_train)))
    print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(X_test, y_test)))
    
    
    
gb_clf2 = GradientBoostingClassifier(n_estimators=20, learning_rate=0.5, max_features=2, max_depth=2, random_state=0)
gb_clf2.fit(X_train, y_train)
predictions = gb_clf2.predict(X_test)

from sklearn.metrics import confusion_matrix

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

#SINCE LEARNING RATE = 0.1 SHOWS THE BEST OPTIMIZATION RESULT AMONG OTHER LEARNING RATES

from sklearn.ensemble import GradientBoostingClassifier

gb_clf2 = GradientBoostingClassifier(n_estimators=20, learning_rate=0.1, max_features=2, max_depth=2, random_state=0)
gb_clf2.fit(X_train, y_train)
predictions = gb_clf2.predict(X_test)

from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min() - 1, stop= X_set[:,0].max() + 1, step= 0.01),
                     np.arange(start = X_set[:,1].min() - 1, stop= X_set[:,1].max() + 1, step= 0.01))
#plt.contourf(X1,X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
 #            alpha = 0.75, cmap = ListedColormap(('red','green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set ==j,0], X_set[y_set == j,1],
                c = ListedColormap(('red','green'))(i), label =j)
plt.title('Gradient-Boosting-Trees(Test)')
plt.xlabel('Cycle Number')
plt.ylabel('Cycle-Prediction')
plt.legend()
plt.show()  

from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min() - 1, stop= X_set[:,0].max() + 1, step= 0.01),
                     np.arange(start = X_set[:,1].min() - 1, stop= X_set[:,1].max() + 1, step= 0.01))
#plt.contourf(X1,X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
 #            alpha = 0.75, cmap = ListedColormap(('red','green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set ==j,0], X_set[y_set == j,1],
                c = ListedColormap(('red','green'))(i), label =j)
plt.title('Gradient-Boosting-Trees(Train)')
plt.xlabel('Cycle Number')
plt.ylabel('Cycle-Prediction')
plt.legend()
plt.show()  



