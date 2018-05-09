
# coding: utf-8

# In[8]:


import numpy as np
from sklearn import preprocessing,cross_validation,neighbors
import pandas as pd
df=pd.read_csv("breast_cancer.csv")
#Replace ? with -99999 in the dataset to recognize as outlier
df.replace('?',-99999,inplace=True)
# drop id column from the dataset
df.drop(['id'],1,inplace=True)
#storing features of the dataset
x=np.array(df.drop(['class'],1))

#storing labels of the dataset
y=np.array(df['class'])

#we are spliting data set into train and test set(20%)
x_train,x_test,y_train,y_test=cross_validation.train_test_split(x,y,test_size=0.2)

#loading KNN algorith to clf
clf=neighbors.KNeighborsClassifier()
#train the model using training data set
clf.fit(x_train,y_train)
#measures the accuracy of the model
predict=clf.predict(x_test)
accuracy=clf.score(x_test,y_test)
print(accuracy)

