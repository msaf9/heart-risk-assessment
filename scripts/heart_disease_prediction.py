#!/usr/bin/env python
# coding: utf-8

# # Heart Risk Assessment Model
# 
# This project focuses on developing a heart disease prediction model using machine learning techniques. The objective is to accurately predict the presence of heart disease in patients based on various health metrics and attributes.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

import warnings
warnings.filterwarnings('ignore')


# # Step 1: Data Loading
# ## Read dataset file named 'heart_statlog_cleveland_hungary_final.csv'

# In[2]:


data = pd.read_csv('../data/heart_statlog_cleveland_hungary_final.csv')


# # Step 2: Data Preprocessing
# ## Ensure target column is binary and of integer type

# In[3]:


data['target'] = data['target'].astype(int)


# ## Check for missing values

# In[4]:


missing_values = data.isnull().sum()
print("Missing Values:")
print(missing_values)


# # Step 3: Data Encoding
# ## If there are categorical variables, encode them using one-hot encoding

# In[5]:


data = pd.get_dummies(data, columns=['sex', 'chest pain type', 'fasting blood sugar', 'resting ecg', 'exercise angina', 'ST slope'])


# ## Feature Scaling

# In[6]:


scaler = StandardScaler()
numeric_cols = ['age', 'resting bp s', 'cholesterol', 'max heart rate', 'oldpeak']
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])


# # Step 4: Exploratory Data Analysis (EDA)
# ## Plot histograms for all features

# In[7]:


data.hist(figsize=(12, 10))
plt.show()


# # Step 5: Splitting Data into Train and Test Sets
# ## Define features (X) and target (y)

# In[8]:


X = data.drop('target', axis=1)
y = data['target']


# ## Split the data into training and testing sets

# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # Step 6: Feature Selection
# ## Plot correlation matrix

# In[10]:


correlation_matrix = data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()


# ## Checking the correlation of 'cholesterol' with the target variable

# In[11]:


print("Correlation of 'cholesterol' with target:", correlation_matrix['target']['cholesterol'])


# ## Initialize the Random Forest model

# In[12]:


rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)


# ## Get feature importances

# In[13]:


feature_importances = pd.DataFrame(rf_model.feature_importances_, index=X_train.columns, columns=['importance']).sort_values('importance', ascending=False)
print("Feature Importances:\n", feature_importances)


# ## Select features based on importance

# In[14]:


selected_features = feature_importances[feature_importances['importance'] > 0.01].index.tolist()
print("Selected Features:", selected_features)


# ## Update feature set

# In[15]:


X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]


# # Step 7: Model Training
# ## Initialize and train a Logistic Regression model

# In[16]:


model = LogisticRegression()
model.fit(X_train, y_train)


# # Step 8: Model Evaluation
# ## Predictions on the test set

# In[17]:


y_pred = model.predict(X_test)


# ## Accuracy
# ### Evaluate the model

# In[18]:


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# # Step 9: Save the Model
# ## Save the trained model as a .pkl file

# In[19]:


with open('../models/heart_disease_model.pkl', 'wb') as file:
    pickle.dump(model, file)


# # Step 10: Load and Use the Model (Example)
# ## To demonstrate loading the model, uncomment the following lines

# In[23]:


# with open('../models/heart_disease_model.pkl', 'rb') as file:
#     loaded_model = pickle.load(file)
#     new_predictions = loaded_model.predict(X_test_selected)
#     print("Loaded Model Accuracy:", accuracy_score(y_test, new_predictions))


# In[ ]:




