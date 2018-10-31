
# coding: utf-8

# In[1]:


import pandas as pd

customer = pd.read_csv("customer.csv" , index_col = 0)

customer.head()


# In[2]:


customer.replace(("Male" , "Female") , (0,1) , inplace = True)
customer.replace(("No" , "Yes") , (0,1) , inplace = True)
customer.replace(("No internet service") , (2) , inplace = True)
customer.InternetService.replace(("Fiber optic" , "DSL") , (1,2) , inplace = True)
customer.Contract.replace(("Month-to-month" , "One year" , "Two year") , (0,1,2) , inplace = True)
customer.MultipleLines.replace(("No phone service") , (2) , inplace = True)
customer.PaymentMethod.replace(("Electronic check" , "Mailed check" , "Bank transfer (automatic)" , "Credit card (automatic)") , (0,1,2,3) , inplace = True)

# We are replacing the all the words into category values (0,1,2.. etc.)
# In[3]:


customer.TotalCharges.replace((" ") , (999999) , inplace = True)


# In[4]:


for n in range(0,customer.shape[0]):
    float(customer.TotalCharges[n])


# In[5]:


customer[customer.columns[0:17]] = customer[customer.columns[0:17]].astype("category")
customer.Churn = customer.Churn.astype("category")
customer.tenure = customer.tenure.astype("int64")


# In[6]:


X_test = customer[customer.TotalCharges == 999999].drop("TotalCharges" , axis = 1)
X_train = customer[customer.TotalCharges != 999999].drop("TotalCharges" , axis = 1)
y_test = customer.TotalCharges[customer.TotalCharges == 999999]
y_train = customer.TotalCharges[customer.TotalCharges != 999999]


# In[7]:


import numpy as np
from sklearn import linear_model

reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)


# In[8]:


print(abs(y_pred))


# In[9]:


abc = abs(y_pred)
#abc = range(1,11)
print(abc)


# In[10]:


customer.TotalCharges[customer.TotalCharges == 999999] = abc


# In[11]:


customer.TotalCharges = customer.TotalCharges.astype("float")

# There are missing values in the total charges column so we replaced the missing values with 999999 to be distinct and easy to look for in the column. After we performed linear regression to predict the total charges for the missing values.


# In[12]:


customer.info()


# In[13]:


import numpy as np


# In[14]:


#customer[customer.columns[0:6]] how to take columns in R


# In[15]:


customer.info()


# In[16]:


import numpy as np

x = customer.MonthlyCharges.values
y = customer.TotalCharges.values

np.correlate(x , y)

# We can find the correlation between the monthly charges and total charges, obviously it should be positive


# In[17]:


customer = customer.drop("MonthlyCharges" , axis = 1)


# In[18]:


import matplotlib.pyplot as plt


# In[19]:


customer.PhoneService.value_counts()


# In[20]:


customer.info() 


# In[21]:


z = [5,7,8,9,10,11,12,13]


# In[22]:


customer.MultipleLines.value_counts()


# In[23]:


#this will work but it will take a lot of time

#plt.subplot(2,1,1)

#plt.bar(customer.PhoneService[customer.PhoneService ==1 ] ,  len(customer.PhoneService[customer.PhoneService ==1]) , align = 'center')
#plt.bar(customer.PhoneService[customer.PhoneService ==0 ] ,  len(customer.PhoneService[customer.PhoneService ==0]) , align = 'center')

#plt.title("Phone Service") 

#plt.subplot(2,1,2)
#plt.bar(customer.MultipleLines[customer.MultipleLines ==2] , len(customer.MultipleLines[customer.MultipleLines == 2]))
#plt.bar(customer.MultipleLines[customer.MultipleLines ==1] , len(customer.MultipleLines[customer.MultipleLines == 1]))
#plt.bar(customer.MultipleLines[customer.MultipleLines ==0] , len(customer.MultipleLines[customer.MultipleLines == 0]))
#plt.title("Multiple Lines")

#plt.show()


# In[24]:


def function(x,p):
    a = customer[customer.columns[x]].value_counts()
    
    b = []
    
    for n in range(0,len(a)):
        b.append(a.index[n])
    
    c = p
    plt.subplot(2,4,8)
    plt.bar(b,a)
    plt.xticks(b,c)
    plt.title(customer.columns[x])
    plt.show()


# In[25]:


p = ["No" , "Yes"]
z = [5,7,8,9,10,11,12,13]
for n in z:
    function(n , p)

# We can graph all the category values to give ourselves a visual on the frequency of each choice
# I need to fix up the third value for the graphs
# Prediction for Churn , we can now finally do our machine 
# learning to predict if the customers will continue the service (0) or terminate their service (1)

#Logistic Regression


# In[29]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

X_customer = customer.drop("Churn" , axis = 1)
y_customer = customer.Churn

logreg = LogisticRegression()

X_train , X_test, y_train , y_test = train_test_split(X_customer, y_customer, test_size = 0.4 , random_state = 1234)


logreg.fit(X_train , y_train)
y_pred = logreg.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test , y_pred))
print(accuracy_score(y_test , y_pred))

# KNN
# In[30]:


from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

param_grid = {'n_neighbors' : np.arange(1,50)}

knn_cv = KNeighborsClassifier()

knn_cv = GridSearchCV(knn_cv, param_grid, cv = 5)

knn_cv.fit(X_train , y_train)

print(knn_cv.best_params_)

print(knn_cv.best_score_)

y_pred = knn_cv.predict(X_test)

print("accuracy:" , accuracy_score(y_test , y_pred))

# Decision Tree + Random Search CV
# In[31]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {'max_depth' : [3,None],
             'max_features' : np.arange(1,9),
             'min_samples_leaf':np.arange(1,9),
             "criterion" :["gini" , "entropy"]}

tree = DecisionTreeClassifier()
tree_cv = RandomizedSearchCV(tree , param_dist , cv = 5 , verbose = 1)

tree_cv.fit(X_train , y_train)
print(tree_cv.best_params_)
print(tree_cv.best_score_)
print("accuracy:", accuracy_score(y_test , y_pred))


# In[32]:


dt = DecisionTreeClassifier(max_depth=4, random_state=123)

# Fit the classifier to the training set
dt.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = dt.predict(X_test)

# Compute the accuracy of the predictions: accuracy
accuracy = float(np.sum(y_pred==y_test))/y_test.shape[0]
print(accuracy)

# SVC
# In[33]:


from sklearn.svm import SVC

clf = SVC()

clf.fit(X_train , y_train)

y_pred = clf.predict(X_test)

print(classification_report(y_test , y_pred))
print('accuracy:' , accuracy_score(y_test , y_pred))

# Keras
# In[34]:


from keras.layers import Dense
from keras.models import Sequential

predictors_customer = customer.drop(["Churn"] , axis = 1).values
target_customer = customer.Churn.values
n_cols = predictors_customer.shape[1]

model_data = Sequential()
model_data.add(Dense(100 , activation = 'relu' , input_shape = (n_cols,)))
for n in range (1,100):
    model_data.add(Dense(n,activation = 'relu'))
model_data.add(Dense(2, activation = 'softmax'))

model_data.compile(optimizer = 'adam' , loss = 'sparse_categorical_crossentropy' , 
                  metrics = ['accuracy'])
model_data.fit(predictors_customer , target_customer)

# XGBoost
# In[35]:


customer_1  = customer[customer.columns[0:19]].astype("int")


# In[36]:


X_customer_1 = customer_1.drop("Churn" , axis = 1)
y_customer_1 = customer_1.Churn


X_train , X_test, y_train , y_test = train_test_split(X_customer_1, y_customer_1, test_size = 0.4 , random_state = 1234)


# In[37]:


import xgboost as xgb
xg = xgb.XGBClassifier(objective='reg:logistic', n_estimators = 10, seed=1234)
xg.fit(X_train, y_train)

y_pred = xg.predict(X_test)

accuracy =  accuracy_score(y_test, y_pred)
print(accuracy)


# In[38]:


churn_dmatrix = xgb.DMatrix(data = X_customer_1, label = y_customer_1)

# Create the parameter dictionary: params
# Xgboost parameters
params = {'learning_rate': 0.05, 
              'max_depth': 4,
              'subsample': 0.9,        
              'colsample_bytree': 0.9,
              'objective': 'multi:softprob',
              'num_class': 3,
              'silent': 1, 
              'n_estimators':100, 
              'gamma':1,         
              'min_child_weight':4} 

#params = {"objective":"reg:logistic", "max_depth":3}
#params = {"objective":"multi:softprob", "max_depth":3, 'num_class': 3, 'eta': 0.3, 'silent': True}#, 'num_round' : 20}

# Perform cross-validation: cv_results
cv_results = xgb.cv(dtrain = churn_dmatrix, params = params, nfold=3, num_boost_round=5, metrics="mlogloss", as_pandas=True, seed=1234)

# Print cv_results
print(cv_results)

# Print the accuracy
#print(((1-cv_results["test-mlogloss-mean"]).iloc[-1]))
print("Accuracy:" ,cv_results['test-mlogloss-mean'].mean()*100)


# In[104]:


import warnings
warnings.filterwarnings("ignore")

params = {'learning_rate': [0.48 ,0.49,0.5] ,
              'max_depth': np.arange(0,4) ,
              'subsample': [0.9],        
              'colsample_bytree': [0.9],
              #'num_class': [2],
              'silent': [1], 
              'n_estimators':[10, 50 , 90 , 100], 
              'gamma':[0,1],         
              'min_child_weight':[1,3,4,6]} 

xg = xgb.XGBClassifier(objective='reg:logistic', seed=1234)

cv = GridSearchCV(xg , param_grid = params , cv = 5 , scoring = 'accuracy' )

cv.fit(X_train , y_train)

y_pred = cv.predict(X_test)

print(cv.best_params_)

print(cv.score(X_test , y_test))

print(classification_report(y_test , y_pred))


# In[101]:


warnings.filterwarnings("ignore")

params = {'learning_rate': range(0,1) ,
              'max_depth': range(0,4) ,
              'subsample': [0.9],        
              'colsample_bytree': [0.9],
              #'num_class': [2],
              'silent': [1], 
              'n_estimators':range(0,100), 
              'gamma':range(0,1),         
              'min_child_weight':range(1,6)} 

xg = xgb.XGBClassifier(objective='reg:logistic', seed=1234)

cv = RandomizedSearchCV(xg , param_distributions = params , cv = 5 , scoring = 'accuracy' , n_iter = 5)

cv.fit(X_train , y_train)

y_pred = cv.predict(X_test)

print(cv.best_params_)

print(cv.score(X_test , y_test))

print(classification_report(y_test , y_pred))

# We can do gridsearchcv and randomizedsearchcv with xgboost to see if we can get a better accuracy.