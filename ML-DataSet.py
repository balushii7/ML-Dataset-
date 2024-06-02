#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[10]:


# Load the dataset
df = pd.read_csv('salaries.csv')

# Display the first few rows of the dataset
print(df.head())


# In[11]:


# Display the column names to verify
print(df.columns)


# In[12]:


# Check for missing values
print("Missing values in each column:\n", df.isnull().sum())


# In[13]:


# Handling outliers ( 'salary_in_usd' is a target)
Q1 = df['salary_in_usd'].quantile(0.25)
Q3 = df['salary_in_usd'].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df['salary_in_usd'] < (Q1 - 1.5 * IQR)) |(df['salary_in_usd'] > (Q3 + 1.5 * IQR)))]

# Display the column names again to verify after handling missing values
print(df.columns)


# In[14]:


# Verify the actual column names in the dataset
print("Columns in the dataset:", df.columns)

# Encode categorical variables
categorical_columns = ['experience_level', 'employment_type', 'job_title', 'salary_currency', 'employee_residence', 'company_location', 'company_size']
existing_categorical_columns = [col for col in categorical_columns if col in df.columns]
df = pd.get_dummies(df, columns=existing_categorical_columns, drop_first=True)


# In[15]:


# Define salary categories
df['salary_category'] = pd.qcut(df['salary_in_usd'], q=2, labels=['Low', 'High'])

# Define feature matrix and target vector
X = df.drop(['salary_in_usd', 'salary_category'], axis=1)
y = df['salary_category']

# For efficiency, use a smaller sample if the dataset is too large
if len(X) > 10000:
    X, _, y, _ = train_test_split(X, y, train_size=10000, random_state=42)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[1]:


# Train Decision Tree model
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)


# In[17]:


# Make predictions
y_pred_dt = dt.predict(X_test)

# Evaluate model
print("Decision Tree Classification Report:\n", classification_report(y_test, y_pred_dt))
print("Decision Tree Accuracy Score:", accuracy_score(y_test, y_pred_dt))


# In[18]:


# Display Confusion Matrix
cm_dt = confusion_matrix(y_test, y_pred_dt)
cm_df_dt = pd.DataFrame(cm_dt, index=['Actual Low', 'Actual High'], columns=['Predicted Low', 'Predicted High'])
print("Decision Tree Confusion Matrix:\n", cm_df_dt)


# In[19]:


# Train KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)


# In[20]:


# Make predictions
y_pred_knn = knn.predict(X_test)

# Evaluate model
print("K-Nearest Neighbors Classification Report:\n", classification_report(y_test, y_pred_knn))
print("K-Nearest Neighbors Accuracy Score:", accuracy_score(y_test, y_pred_knn))


# In[21]:


# Display Confusion Matrix
cm_knn = confusion_matrix(y_test, y_pred_knn)
cm_df_knn = pd.DataFrame(cm_knn, index=['Actual Low', 'Actual High'], columns=['Predicted Low', 'Predicted High'])
print("K-Nearest Neighbors Confusion Matrix:\n", cm_df_knn)


# In[22]:


# Save results to a file
results = {
    "Decision Tree": {
        "Classification Report": classification_report(y_test, y_pred_dt, output_dict=True),
        "Confusion Matrix": cm_df_dt.to_dict(),
        "Accuracy Score": accuracy_score(y_test, y_pred_dt)
    },
    "K-Nearest Neighbors": {
        "Classification Report": classification_report(y_test, y_pred_knn, output_dict=True),
        "Confusion Matrix": cm_df_knn.to_dict(),
        "Accuracy Score": accuracy_score(y_test, y_pred_knn)
    }
}

# Save the results to a JSON file
import json
with open('model_results.json', 'w') as f:
    json.dump(results, f)

