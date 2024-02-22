#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing modules and libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')

import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder


# In[2]:


columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width', 'Species']
df = pd.read_csv('iris.data', names = columns)


# In[3]:


df.head()


# In[4]:


#basic statistical analysis
df.describe()


# In[5]:


#rows x columns
df.shape


# In[6]:


#column(feature) name
df.columns


# In[7]:


#feature type
df.info()


# In[8]:


type(df)


# In[9]:


#groupby size
df.groupby('Species').size()


# In[10]:


#visualize entire dataset
sns.pairplot(df, hue='Species');


# In[11]:


#separate features and target
data = df.values
X = data[:,0:4]#(features)
Y = data[:,4]#(target)


# In[12]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)#scaling X

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(Y)#encoding Y


# In[13]:


#calculating average of some columns using y
Y_Data = np.array([np.average(X[:, i][Y==j].astype('float32')) for i in range (X.shape[1]) for j in (np.unique(Y))])
#reshaping to 4x3 matrix
Y_Data_reshaped = Y_Data.reshape(4, 3)
#changing 4x3 to 3x4 matrix
Y_Data_reshaped = np.swapaxes(Y_Data_reshaped, 0, 1)
#x axis
X_axis = np.arange(len(columns)-1)
#setting width for bar graph
width = 0.25


# In[14]:


#bar plot
plt.bar(X_axis, Y_Data_reshaped[0], width, label = 'Setosa')
plt.bar(X_axis+width, Y_Data_reshaped[1], width, label = 'Versicolour')
plt.bar(X_axis+width*2, Y_Data_reshaped[2], width, label = 'Virginica')
plt.xticks(X_axis, columns[:4])
#labeling
plt.xlabel("Features")
plt.ylabel("Value in cm.")
#legend
plt.legend(bbox_to_anchor=(1.3,1))
#display
plt.show()


# In[15]:


#splitting data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)


# In[16]:


#intialize a model using support vector classifier and fit the model to a training data
svn = SVC()
svn.fit(X_train, y_train)


# In[17]:


#predicted variables are stored in predictions
predictions = svn.predict(X_test)


# In[18]:


accuracy_score(y_test, predictions)


# In[19]:


#evaluation metrices
print(classification_report(y_test, predictions))


# In[20]:


X_new = np.array([[3, 2, 1, 0.2], [  4.9, 2.2, 3.8, 1.1 ], [  5.3, 2.5, 4.6, 1.9 ]])
#Prediction of the species from the input vector
prediction = svn.predict(X_new)
print("Prediction of Species: {}".format(prediction))


# In[21]:


with open('SVM.pickle', 'wb') as f:
    pickle.dump(svn, f)


# In[22]:


with open('SVM.pickle', 'rb') as f:
    model = pickle.load(f)


# In[23]:


#make predictions on new data
model.predict(X_new)


# In[24]:


from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(X_train,y_train)


# In[25]:


training_prediction = log_reg.predict(X_train)
training_prediction


# In[26]:


test_prediction = log_reg.predict(X_test)
test_prediction


# In[ ]:




