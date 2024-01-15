#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 20:40:24 2023

@author: Tejinder
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

KSI_G1 = pd.read_csv('/Users/user/Documents/Centennial college/SEM4/Supervised learning/Term project/KSI.csv') 

KSI_G1.head()
KSI_G1.describe()
KSI_G1.info()

KSI_G1.isnull().sum()

#Columns that have huge missing values and will not make much contribution to the model
drop1 = ['X','Y','INDEX_','OFFSET','ACCLOC','MANOEUVER','PEDTYPE','PEDACT','PEDCOND','CYCLISTYPE',
         'CYCACT','CYCCOND','HOOD_140','NEIGHBOURHOOD_140','DIVISION','ObjectId']

#First drop made
KSI_G1_CLEAN = KSI_G1.drop(columns = drop1)

#ACCLASS seems to be the target, Deal with it's missing values and change it to consist of 2 unique values only
#Let's drop the mssing rows first
KSI_G1_CLEAN.dropna(subset=['ACCLASS'], inplace=True)
KSI_G1_CLEAN.isnull().sum()
#Now change the unique values
KSI_G1_CLEAN['ACCLASS'] = np.where(KSI_G1_CLEAN['ACCLASS'] == 'Property Damage Only', 'Non-Fatal', KSI_G1_CLEAN['ACCLASS'])
KSI_G1_CLEAN['ACCLASS'] = np.where(KSI_G1_CLEAN['ACCLASS'] == 'Non-Fatal Injury', 'Non-Fatal', KSI_G1_CLEAN['ACCLASS'])
KSI_G1_CLEAN['ACCLASS'] = np.where(KSI_G1_CLEAN['ACCLASS'] == 'Non-Fatal', 0, KSI_G1_CLEAN['ACCLASS'])
KSI_G1_CLEAN['ACCLASS'] = np.where(KSI_G1_CLEAN['ACCLASS'] == 'Fatal', 1, KSI_G1_CLEAN['ACCLASS'])
KSI_G1_CLEAN['ACCLASS'] = KSI_G1_CLEAN['ACCLASS'].astype('int64')

KSI_G1_CLEAN['ACCLASS'].nunique()

#I see that the data can be basically broken in three categories
#   1. The location of the accident
#   2. The type of Vehicles involved
#   3. The reason that could have influenced the accident

geo_data = ['ACCNUM', 'YEAR', 'DATE', 'TIME',
       'LATITUDE', 'LONGITUDE', 'WARDNUM',
       'STREET1', 'STREET2','DISTRICT','ROAD_CLASS', 'LOCCOORD', 'HOOD_158','NEIGHBOURHOOD_158',
       'TRAFFCTL', 'VISIBILITY', 'LIGHT', 'RDSFCOND', 'ACCLASS',
       'IMPACTYPE', 'INVTYPE', 'INVAGE', 'INJURY', 'FATAL_NO', 'INITDIR',
       ]

involved = ['VEHTYPE', 'DRIVACT', 'DRIVCOND', 'PEDESTRIAN', 'CYCLIST',
'AUTOMOBILE', 'MOTORCYCLE', 'TRUCK', 'TRSN_CITY_VEH', 'EMERG_VEH',
'PASSENGER']

conditions = ['SPEEDING', 'AG_DRIV', 'REDLIGHT', 'ALCOHOL', 'DISABILITY']


#WE NEED TO EXPAND geo_data
# Convert the 'DATE' column to datetime type
KSI_G1_CLEAN['DATE'] = pd.to_datetime(KSI_G1_CLEAN['DATE'])

# Extract month, day, and day of week (as numbers 1 to 7)
KSI_G1_CLEAN['MONTH'] = KSI_G1_CLEAN['DATE'].dt.month
KSI_G1_CLEAN['DAY'] = KSI_G1_CLEAN['DATE'].dt.day
KSI_G1_CLEAN['DAYOFWEEK'] = KSI_G1_CLEAN['DATE'].dt.dayofweek + 1

# Drop the 'DATE' column
KSI_G1_CLEAN.drop(columns=['DATE'], inplace=True)


#Let us clean the TIME column
# Convert the 'Time' column to string data type
KSI_G1_CLEAN['TIME'] = KSI_G1_CLEAN['TIME'].astype(str)

# Function to convert time to consistent format 'HH:MM'
def convert_to_time_format(time_str):
    if '::' in time_str:
        time_str = time_str.replace('::', ':')
    if len(time_str) == 3:
        return f"{time_str[:1]:0>2}:{time_str[1:]:0>2}"
    elif len(time_str) == 4:
        return f"{time_str[:2]}:{time_str[2:]:0>2}"
    else:
        return time_str

# Convert the 'Time' column to consistent time format 'HH:MM'
KSI_G1_CLEAN['TIME'] = KSI_G1_CLEAN['TIME'].apply(convert_to_time_format)

# Replace '::' with ':' in the 'Time' column
KSI_G1_CLEAN['TIME'] = KSI_G1_CLEAN['TIME'].str.replace('::', ':')

# Function to convert time to consistent format 'HHMM'
def convert_to_time_format2(time_str):
    if ':' in time_str:
        time_str = time_str.replace(':', '')
    if len(time_str) == 1:
        return f"000{time_str}"
    elif len(time_str) == 2:
        return f"00{time_str}"
    elif len(time_str) == 3:
        return f"0{time_str}"
    else:
        return time_str

# Convert the 'Time' column to consistent time format 'HHMM'
KSI_G1_CLEAN['TIME'] = KSI_G1_CLEAN['TIME'].apply(convert_to_time_format2)


# Convert the 'Time' column to datetime with format '%H:%M'
KSI_G1_CLEAN['TIME'] = pd.to_datetime(KSI_G1_CLEAN['TIME'], format='%H%M')

# Extract hour and minute as separate columns
KSI_G1_CLEAN['HOUR'] = KSI_G1_CLEAN['TIME'].dt.hour
KSI_G1_CLEAN['MINUTE'] = KSI_G1_CLEAN['TIME'].dt.minute

# Drop the 'Time' column
KSI_G1_CLEAN.drop(columns=['TIME'], inplace=True)

geo_data = ['ACCNUM', 'YEAR', 'MONTH', 'DAY', 'DAYOFWEEK', 'HOUR','MINUTE',
       'LATITUDE', 'LONGITUDE', 'WARDNUM',
       'STREET1', 'STREET2','DISTRICT','ROAD_CLASS', 'LOCCOORD', 'HOOD_158','NEIGHBOURHOOD_158',
       'TRAFFCTL', 'VISIBILITY', 'LIGHT', 'RDSFCOND', 'ACCLASS',
       'IMPACTYPE', 'INVTYPE', 'INVAGE', 'INJURY', 'FATAL_NO', 'INITDIR',
       ]

KSI_G1_CLEAN[geo_data]

#
#
#
#
###############################################
########### Graphs and Visualizations##########
###############################################

# Example 1: Bar plot using pandas
KSI_G1_CLEAN['ROAD_CLASS'].value_counts().plot(kind='bar')
plt.xlabel('Road Class')
plt.ylabel('Count')
plt.title('Bar Plot: Road Class')
plt.show()

# Example 2: Scatter plot using matplotlib
plt.scatter(KSI_G1['LONGITUDE'], KSI_G1['LATITUDE'])
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Scatter Plot: Location')
plt.show()

# Example 3: Box plot using seaborn
sns.boxplot(x=KSI_G1['YEAR'], y=KSI_G1['ACCNUM'])
plt.xlabel('Year')
plt.ylabel('Accident Number')
plt.title('Box Plot: Year vs Accident Number')
plt.show()

# Example 4: Histogram using matplotlib
plt.hist(KSI_G1['LATITUDE'], bins=10)
plt.xlabel('Latitude')
plt.ylabel('Frequency')
plt.title('Histogram: Latitude Distribution')
plt.show()

#Number of Unique accidents by Year
Num_accident = KSI_G1.groupby('YEAR')['ACCNUM'].nunique()
plt.figure(figsize=(12,6))
plt.title("Accidents caused in different years")
plt.ylabel('Number of Accidents (ACCNUM)')

ax = plt.gca()
ax.tick_params(axis='x', colors='blue')
ax.tick_params(axis='y', colors='red')
my_colors = list('rgbkymc')   #red, green, blue, black, etc.
Num_accident.plot(
    kind='bar', 
    color='blue',
    edgecolor='black'
)
#Num_accident.plot(kind='bar',color= my_colors)
plt.show()

#Categorizing Fatal vs. non-Fatal Incident (non-unique i.e: one accident is counted depending upon involved parties)
sns.catplot(x='YEAR', kind='count', data=KSI_G1,  hue='ACCLASS')
#
#
#
#
drop2= ['WARDNUM', 'NEIGHBOURHOOD_158','HOOD_158',
        'STREET1', 'STREET2', 'ROAD_CLASS', 'LOCCOORD','TRAFFCTL',
        'IMPACTYPE', 'INVTYPE', 'INVAGE', 'INJURY',
        'FATAL_NO', 'INITDIR', 'VEHTYPE', 'DRIVACT', 'DRIVCOND','ACCNUM']

KSI_G1_CLEAN = KSI_G1_CLEAN.drop(columns = drop2)
KSI_G1_CLEAN.info()
KSI_G1_CLEAN.isnull().sum()

#A lot of columns can be simply cleaned by changing the nan values to 'No'
columns_with_two_unique_values = [col for col in KSI_G1_CLEAN.columns if KSI_G1_CLEAN[col].nunique() == 1 and 'Yes' in KSI_G1_CLEAN[col].unique()]

print(KSI_G1_CLEAN[columns_with_two_unique_values])
KSI_G1_CLEAN[columns_with_two_unique_values] = KSI_G1_CLEAN[columns_with_two_unique_values].replace(np.nan, 'No')


#AFTER MAJOR CLEANING
KSI_G1_CLEAN.isnull().sum()


KSI_G1_CLEAN= KSI_G1_CLEAN.dropna(subset=['DISTRICT'])
KSI_G1_CLEAN.isnull().sum()


# Get unique values and value counts for each column in df_filtered
for col in KSI_G1_CLEAN.columns:
    unique_values = KSI_G1_CLEAN[col].unique()
    value_counts = KSI_G1_CLEAN[col].nunique()
    print(f"Column '{col}':")
    print("Unique Values:", unique_values)
    #print("Value Counts:")
    print(value_counts)
    print("---------------------")

columns_with_yes_no = [col for col in KSI_G1_CLEAN.columns if KSI_G1_CLEAN[col].nunique() == 2 and 'Yes' in KSI_G1_CLEAN[col].unique()]

print(KSI_G1_CLEAN[columns_with_yes_no])
KSI_G1_CLEAN[columns_with_yes_no] = KSI_G1_CLEAN[columns_with_yes_no].replace('Yes', 1)
KSI_G1_CLEAN[columns_with_yes_no] = KSI_G1_CLEAN[columns_with_yes_no].replace('No',0)
KSI_G1_CLEAN[columns_with_yes_no] = KSI_G1_CLEAN[columns_with_yes_no].astype(int)
KSI_G1_CLEAN.info()

KSI_G1_CLEAN.isnull().sum()

KSI_G1_CLEAN.reset_index(drop=True, inplace=True)
from sklearn.model_selection import train_test_split

features = KSI_G1_CLEAN.drop('ACCLASS', axis=1).columns.tolist()
predict = 'ACCLASS'
X_tejinder = KSI_G1_CLEAN[KSI_G1_CLEAN.columns.difference([predict])]
y_tejinder = KSI_G1_CLEAN[predict]

np.random.seed(34)
X_train, X_test, y_train, y_test = train_test_split(X_tejinder, y_tejinder, test_size = 0.2, random_state = 34)

# Get the column names of X_train
column_names = X_train.columns.tolist()

# Define the path for saving the joblib file
save_path = '/Users/user/Documents/Centennial college/SEM4/Supervised learning/Term project/Further Changes/'

# Save the column names using joblib
joblib.dump(column_names, save_path + 'G1_column.pkl')

print("Column names saved successfully.")


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

cat_attribs = ['DISTRICT', 'LIGHT','VISIBILITY', 'RDSFCOND']
num_attribs = X_train.drop(columns=cat_attribs).columns.tolist()

num_pipeline = Pipeline([
    ('std_scaler', StandardScaler())
])

cat_pipeline = Pipeline([
        ("cat_imputer", SimpleImputer(strategy="most_frequent")),
        ("cat_one_hot", OneHotEncoder())
    ])

full_pipeline = ColumnTransformer(transformers=[
    ("cat", cat_pipeline, cat_attribs),
    ("num", num_pipeline, num_attribs)
])
X_train.info()
X_test.info()
# Apply the pipeline to the DataFrame
X_G1_transformed = full_pipeline.fit_transform(X_train)

# Convert the transformed data to a DataFrame
X_G1_transformed_pd = pd.DataFrame(X_G1_transformed)


X_test_transformed = full_pipeline.transform(X_test)

X_test_transformed_pd = pd.DataFrame(X_test_transformed)

#How is the class distributer
from collections import Counter

# Calculate the class distribution
class_distribution = Counter(y_train)

# Print the number of 0s and 1s
print("Number of 0s in y_train:", class_distribution[0])
print("Number of 1s in y_train:", class_distribution[1])


'''
##  SMOTING------ TO INCREASE FATAL INTO THE CODE ,, 
#AND TO MAKE MODEL MORE BETTER AS WE HAVE NON-FATAL : 10857 AND FATAL :1788
'''
from imblearn.over_sampling import SMOTE

counter = Counter(y_train)
print(counter)

oversample = SMOTE()
X_G1_transformed, y_train = oversample.fit_resample(X_G1_transformed, y_train)

counter = Counter(y_train)
print(counter)

X_G1_transformed_pd = pd.DataFrame(X_G1_transformed)

"""
Training 5 Models using loop and checking ACCURACY SCORE
"""

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Define the models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Neural Network": MLPClassifier(random_state=42),
    "SVC": SVC(max_iter=1000)
}

# Loop through each model
for model_name, model in models.items():
    print(f"Training and evaluating {model_name}...")
    
    # Fit the model to your training data
    model.fit(X_G1_transformed, y_train)

    # Make predictions on the train and test data
    y_train_pred = model.predict(X_G1_transformed)
    y_test_pred = model.predict(X_test_transformed)

    # Calculate the accuracy scores for train and test data
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print(f"Train accuracy score for {model_name} is: {train_accuracy:.3f}")
    print(f"Test accuracy score for {model_name} is: {test_accuracy:.3f}")
    print("------------------------")


"""
PRECISION, RECALL AND CONFUSION MATRIX for the 5 models
"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Loop through each model
for model_name, model in models.items():
    print(f"Evaluating {model_name}...")
    
    # Fit the model to your training data
    model.fit(X_G1_transformed, y_train)

    # Make predictions on the train and test data
    y_train_pred = model.predict(X_G1_transformed)
    y_test_pred = model.predict(X_test_transformed)

    # Calculate the metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)

    print(f"Train accuracy score: {train_accuracy:.3f}")
    print(f"Test accuracy score: {test_accuracy:.3f}")
    print(f"Test precision score: {test_precision:.3f}")
    print(f"Test recall score: {test_recall:.3f}")

    conf_matrix = confusion_matrix(y_test, y_test_pred)
    print("Confusion Matrix:")
    print(conf_matrix)
    print("------------------------")

#We have to cehck for overfitting for Random Forest

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Create a Random Forest classifier
clf_rf_G1 = RandomForestClassifier(n_estimators=100, random_state=42)

# Perform cross-validation with 5 folds
# 'X_G1_transformed' is your feature matrix and 'y_train' is your target labels
scores = cross_val_score(clf_rf_G1, X_G1_transformed, y_train, cv=10, scoring='accuracy')

# Print the cross-validation scores
print("Cross-Validation Scores:", scores)
print("Mean CV Score:", scores.mean())
print("Standard Deviation:", scores.std())

#The scores suggest that the model is not overfiiitng despite it's high training score

#Calculate RMSE
from sklearn.metrics import mean_squared_error

rf_predictions = clf_rf_G1.predict(X_G1_transformed)

forest_mse = mean_squared_error(y_train,rf_predictions)

forest_rmse = np.sqrt(forest_mse)

forest_rmse

"""
#########################
HYPERTUNING FINE MODEL
#########################
"""
from sklearn.model_selection import GridSearchCV

"""
GridSearchCV LOGISTIC REGRESSION
"""

clf_lr_G1 = LogisticRegression(max_iter=1000)
# Define the hyperparameter grid for Search
param_grid_lr = [
    {
        'penalty': ['l1', 'l2'],
        'C': [0.1, 1, 10],
        'solver': ['liblinear']
    }
]

# Create the GridSearchCV instance for Logistic Regression
grid_search_lr = GridSearchCV(clf_lr_G1, param_grid_lr, cv=5, scoring='accuracy')

# Fit the GridSearchCV instance to the data
grid_search_lr.fit(X_G1_transformed, y_train)

# Print the best hyperparameters and corresponding accuracy score
print("Best Hyperparameters for Logistic Regression:", grid_search_lr.best_params_)
print("Best Accuracy Score for Logistic Regression:", grid_search_lr.best_score_)
print("Best Estimator for Logistic Regression:", grid_search_lr.best_estimator_)

final_param_lr = grid_search_lr.best_params_
final_model_lr = grid_search_lr.best_estimator_

"""
GridSearchCV DECISION TREE
"""

clf_dt_G1 = DecisionTreeClassifier(random_state=42)
# Define the hyperparameter grid for Search
param_grid_dt = [
    {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
]

# Create the GridSearchCV instance for Decision Tree
grid_search_dt = GridSearchCV(clf_dt_G1, param_grid_dt, cv=5, scoring='accuracy')

# Fit the GridSearchCV instance to the data
grid_search_dt.fit(X_G1_transformed, y_train)

# Print the best hyperparameters and corresponding accuracy score
print("Best Hyperparameters for Decision Tree:", grid_search_dt.best_params_)
print("Best Accuracy Score for Decision Tree:", grid_search_dt.best_score_)
print("Best Estimator for Decision Tree:", grid_search_dt.best_estimator_)

final_param_dt = grid_search_dt.best_params_
final_model_dt = grid_search_dt.best_estimator_

"""
GridSearchCV RANDOM FOREST CLASSIFIER
"""
clf_rf_G1 = RandomForestClassifier(random_state=42)
# Define the hyperparameter grid for Search
param_grid = {
    'n_estimators': [50, 100],
    'max_features': [4, 6],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False]
}

# Create the GridSearchCV instance
grid_search = GridSearchCV(clf_rf_G1, param_grid, cv=5, scoring='accuracy')

y_train_pd = pd.DataFrame(y_train)
# Fit the RandomizedSearchCV instance to the data
grid_search.fit(X_G1_transformed, y_train)

# Print the best hyperparameters and corresponding accuracy score
print("Best Hyperparameters:", grid_search.best_params_)
print("Best Accuracy Score:", grid_search.best_score_)
print("Best Accuracy Score:", grid_search.best_estimator_)

final_param_rf = grid_search.best_params_
final_model_rf = grid_search.best_estimator_

"""
GridSearchCV Neural Networks
"""

clf_nn_G1 = MLPClassifier(random_state=42)
# Define the hyperparameter grid for Search
param_grid_nn = [
    {
        'hidden_layer_sizes': [(50,), (100,)],
        'activation': ['relu'],
        'alpha': [0.001, 0.01],
        'max_iter': [100, 200]
    }
]

# Create the GridSearchCV instance for Neural Network
grid_search_nn = GridSearchCV(clf_nn_G1, param_grid_nn, cv=5, scoring='accuracy')

# Fit the GridSearchCV instance to the data
grid_search_nn.fit(X_G1_transformed, y_train)

# Print the best hyperparameters and corresponding accuracy score
print("Best Hyperparameters for Neural Network:", grid_search_nn.best_params_)
print("Best Accuracy Score for Neural Network:", grid_search_nn.best_score_)
print("Best Estimator for Neural Network:", grid_search_nn.best_estimator_)

final_param_nn = grid_search_nn.best_params_
final_model_nn = grid_search_nn.best_estimator_

"""
GridSearchCV SVC
"""

# Define the hyperparameter grid for Search
param_grid_svc = [
    {
        'C': [0.1, 1],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
]

# Create the GridSearchCV instance for SVC
grid_search_svc = GridSearchCV(SVC(), param_grid_svc, cv=5, scoring='accuracy')

# Fit the GridSearchCV instance to the data
grid_search_svc.fit(X_G1_transformed, y_train)

# Print the best hyperparameters and corresponding accuracy score
print("Best Hyperparameters for SVC:", grid_search_svc.best_params_)
print("Best Accuracy Score for SVC:", grid_search_svc.best_score_)
print("Best Estimator for SVC:", grid_search_svc.best_estimator_)

final_param_svc = grid_search_svc.best_params_
final_model_svc = grid_search_svc.best_estimator_



#Final Testing
X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model_rf.predict(X_test_prepared)

final_mse = mean_squared_error(y_test,final_predictions)

final_rmse = np.sqrt(final_mse)

print(final_rmse)


joblib.dump(full_pipeline, "/Users/user/Documents/Centennial college/SEM4/Supervised learning/Term project/Further Changes/G1_pipeline.pkl")

joblib.dump(final_model_rf, "/Users/user/Documents/Centennial college/SEM4/Supervised learning/Term project/Further Changes/G1_model.pkl")
joblib.dump(final_model_lr, "/Users/user/Documents/Centennial college/SEM4/Supervised learning/Term project/Further Changes/G1_model_lr.pkl")
joblib.dump(final_model_dt, "/Users/user/Documents/Centennial college/SEM4/Supervised learning/Term project/Further Changes/G1_model_dt.pkl")
joblib.dump(final_model_nn, "/Users/user/Documents/Centennial college/SEM4/Supervised learning/Term project/Further Changes/G1_model_nn.pkl")
joblib.dump(final_model_svc, "/Users/user/Documents/Centennial college/SEM4/Supervised learning/Term project/Further Changes/G1_model_svc.pkl")













