#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


df = pd.read_csv('C:/Users/admin/Documents/heart.csv')
df.head(3)


# In[8]:


df.info()


# In[9]:


print('Number of rows in the dataset: ',df.shape[0])
print('Number of columns in the dataset: ',df.shape[1])


# In[10]:


df.isnull().sum()


# In[11]:


df.describe()


# In[12]:


sns.set_style('whitegrid')


# In[13]:


plt.figure(figsize=(14,8))
sns.heatmap(df.corr(), annot = True, cmap='coolwarm',linewidths=.1)
plt.show()


# In[14]:


sns.distplot(df['thalach'],kde=False,bins=30,color='violet')


# In[15]:


sns.distplot(df['chol'],kde=False,bins=30,color='red')
plt.show()


# In[16]:


sns.distplot(df['trestbps'],kde=False,bins=30,color='blue')
plt.show()


# In[17]:


plt.figure(figsize=(15,6))
sns.countplot(x='age',data = df, hue = 'target',palette='GnBu')
plt.show()


# In[18]:


plt.figure(figsize=(8,6))
sns.scatterplot(x='chol',y='thalach',data=df,hue='target')
plt.show()


# In[19]:


plt.figure(figsize=(8,6))
sns.scatterplot(x='trestbps',y='thalach',data=df,hue='target')
plt.show()


# In[20]:


X= df.drop('target',axis=1)
y=df['target']


# In[21]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.3,random_state=42)


# In[22]:


from sklearn.preprocessing import StandardScaler


# In[23]:


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_train = pd.DataFrame(X_train_scaled)

X_test_scaled = scaler.transform(X_test)
X_test = pd.DataFrame(X_test_scaled)


# In[24]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
knn =KNeighborsClassifier()
params = {'n_neighbors':list(range(1,20)),
    'p':[1, 2, 3, 4,5,6,7,8,9,10],
    'leaf_size':list(range(1,20)),
    'weights':['uniform', 'distance']
         }


# In[25]:


model = GridSearchCV(knn,params,cv=3, n_jobs=-1)


# In[57]:


from sklearn.metrics import classification_report


# In[58]:


print(classification_report(y_test,predict))


# In[59]:


from sklearn.metrics import roc_auc_score,roc_curve


# In[60]:


#Get predicted probabilites from the model
y_probabilities = model.predict_proba(X_test)[:,1]


# In[61]:


#Create true and false positive rates
false_positive_rate_knn,true_positive_rate_knn,threshold_knn = roc_curve(y_test,y_probabilities)


# In[62]:


#Plot ROC Curve
plt.figure(figsize=(10,6))
plt.title('Revceiver Operating Characterstic')
plt.plot(false_positive_rate_knn,true_positive_rate_knn)
plt.plot([0,1],ls='--')
plt.plot([0,0],[1,0],c='.5')
plt.plot([1,1],c='.5')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()


# In[64]:


#Calculate area under the curve
roc_auc_score(y_test,y_probabilities)


# In[65]:


from sklearn.linear_model import LogisticRegression
log = LogisticRegression()


# In[66]:


params = {'penalty':['l1','l2'],
         'C':[0.01,0.1,1,10,100],
         'class_weight':['balanced',None]}
log_model = GridSearchCV(log,param_grid=params,cv=10)


# In[67]:


log_model.fit(X_train,y_train)

# Printing best parameters choosen through GridSearchCV
log_model.best_params_


# In[68]:


predict = log_model.predict(X_test)


# In[69]:


from sklearn.metrics import accuracy_score
print('Accuracy Score: ',accuracy_score(y_test,predict))
print('Using Logistic Regression we get an accuracy score of: ',
      round(accuracy_score(y_test,predict),5)*100,'%')


# In[70]:


from sklearn.metrics import recall_score,precision_score,classification_report,roc_auc_score,roc_curve
print(classification_report(y_test,predict))


# In[74]:


#Get predicted probabilites
target_probailities_log = log_model.predict_proba(X_test)[:,1]


# In[75]:


#Create true and false positive rates
log_false_positive_rate,log_true_positive_rate,log_threshold = roc_curve(y_test,
                                                             target_probailities_log)


# In[76]:


#Plot ROC Curve
sns.set_style('whitegrid')
plt.figure(figsize=(10,6))
plt.title('Reciver Operating Characterstic Curve')
plt.plot(log_false_positive_rate,log_true_positive_rate)
plt.plot([0,1],ls='--')
plt.plot([0,0],[1,0],c='.5')
plt.plot([1,1],c='.5')
plt.ylabel('True positive rate')
plt.xlabel('False positive rate')
plt.show()


# In[77]:


#Calculate area under the curve
roc_auc_score(y_test,target_probailities_log)


# In[78]:


from sklearn.tree import DecisionTreeClassifier
dtree= DecisionTreeClassifier(random_state=7)


# In[79]:


#Setting parameters for GridSearchCV
params = {'max_features': ['auto', 'sqrt', 'log2'],
          'min_samples_split': [2,3,4,5,6,7,8,9,10,11,12,13,14,15], 
          'min_samples_leaf':[1,2,3,4,5,6,7,8,9,10,11]}
tree_model = GridSearchCV(dtree, param_grid=params, n_jobs=-1)


# In[80]:


tree_model.fit(X_train,y_train)
#Printing best parameters selected through GridSearchCV
tree_model.best_params_


# In[81]:


predict = tree_model.predict(X_test)


# In[82]:


from sklearn.metrics import accuracy_score
print('Accuracy Score: ',accuracy_score(y_test,predict))
print('Using Decision Tree we get an accuracy score of: ',
      round(accuracy_score(y_test,predict),5)*100,'%')


# In[83]:


from sklearn.metrics import classification_report,roc_auc_score,roc_curve


# In[84]:


print(classification_report(y_test,predict))


# In[85]:


cnf_matrix = confusion_matrix(y_test,predict)
cnf_matrix


# In[86]:


class_names = [0,1]
fig,ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks,class_names)
plt.yticks(tick_marks,class_names)

#create a heat map
sns.heatmap(pd.DataFrame(cnf_matrix), annot = True, cmap = 'YlGnBu',
           fmt = 'g')
ax.xaxis.set_label_position('top')
plt.tight_layout()
plt.title('Confusion matrix for Decision Tree Model', y = 1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()


# In[87]:


#Get predicted probabilites
target_probailities_tree = tree_model.predict_proba(X_test)[:,1]


# In[88]:


#Create true and false positive rates
tree_false_positive_rate,tree_true_positive_rate,tree_threshold = roc_curve(y_test,
                                                             target_probailities_tree)


# In[89]:


#Plot ROC Curve
sns.set_style('whitegrid')
plt.figure(figsize=(10,6))
plt.title('Reciver Operating Characterstic Curve')
plt.plot(tree_false_positive_rate,tree_true_positive_rate)
plt.plot([0,1],ls='--')
plt.plot([0,0],[1,0],c='.5')
plt.plot([1,1],c='.5')
plt.ylabel('True positive rate')
plt.xlabel('False positive rate')
plt.show()


# In[90]:


#Calculate area under the curve
roc_auc_score(y_test,target_probailities_tree)


# In[91]:


#Plot ROC Curve
sns.set_style('whitegrid')
plt.figure(figsize=(10,6))
plt.title('Reciver Operating Characterstic Curve')
plt.plot(false_positive_rate_knn,true_positive_rate_knn,label='k-Nearest Neighbor')
plt.plot(log_false_positive_rate,log_true_positive_rate,label='Logistic Regression')
plt.plot(tree_false_positive_rate,tree_true_positive_rate,label='Decision Tree')
plt.plot([0,1],ls='--')
plt.plot([0,0],[1,0],c='.5')
plt.plot([1,1],c='.5')
plt.ylabel('True positive rate')
plt.xlabel('False positive rate')
plt.legend()
plt.show()


# In[ ]:




