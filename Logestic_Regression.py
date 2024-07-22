###############################################################################
# Logistic Regression- Advanced Template
###############################################################################

###############################################################################
# Import Required Packages
###############################################################################

import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import RFECV

###############################################################################
# Import Sample Data
###############################################################################

# Import
data_for_model = pickle.load(open('Data/abc_classification_modelling.p', 'rb'))                 # 'rb': read file

# Drop unnecessary columns
data_for_model.drop('customer_id', axis = 1, inplace = True)

# Shuffle data
data_for_model = shuffle(data_for_model, random_state = 42)

# Class balance (the proportion of zeros and ones in our data)
data_for_model['signup_flag'].value_counts(normalize = True)               # Checking if the data are balanced (gives us the proportion of ones and zeros)
"""
data_for_model['signup_flag'].value_counts()                               # gives us the number of ones and zeros
"""

###############################################################################
# Deal with Missing Values
###############################################################################

data_for_model.isna().sum()
data_for_model.dropna(how = 'any', inplace = True)

###############################################################################
# Deal with Outliers
###############################################################################

# Describe the data and compare mean, max and min to find the columns that possibly have outliers
outlier_investigation = data_for_model.describe()

outlier_columns = ['distance_from_store', 'total_sales', 'total_items']

# Boxplot approach
for column in outlier_columns:
    
    lower_quartile = data_for_model[column].quantile(0.25)
    upper_quartile = data_for_model[column].quantile(0.75)
    iqr = upper_quartile - lower_quartile
    iqr_extended = iqr * 2                                         # We used 2 instead of 1.5 to keep more data and remove less outliers
    max_border = upper_quartile + iqr_extended
    min_border = lower_quartile - iqr_extended
    
    outliers = data_for_model[(data_for_model[column] > max_border) | (data_for_model[column] < min_border)].index
    print(f'{len(outliers)} outliers were detected in column {column}')
    
    data_for_model.drop(outliers, inplace = True)

###############################################################################
# Split Input Variables and Output Variables
###############################################################################

X = data_for_model.drop(['signup_flag'], axis = 1)
y = data_for_model['signup_flag']

###############################################################################
# Split out Training and Test Sets
###############################################################################

X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.2, random_state = 42, stratify = y)        # stratify = y: our training and test set should have the same proportion of zeros and ones

###############################################################################
# Deal with Categorical Variables
###############################################################################

# Create a list of categorical variables                    
categorical_vars = ['gender']   

# Create and apply OneHotEncoder while removing the dummy variable
one_hot_encoder = OneHotEncoder(sparse = False, drop = 'first')               

# Apply fit_transform on training data
X_train_encoded = one_hot_encoder.fit_transform(X_train[categorical_vars])

# Apply transform on test data
X_test_encoded = one_hot_encoder.transform(X_test[categorical_vars])

# Get feature names to see what each column in the 'encoder_vars_array' presents
encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)

# Convert our result from an array to a DataFrame
X_train_encoded = pd.DataFrame(X_train_encoded, columns = encoder_feature_names)

# Concatenate (Link together in a series or chain) new DataFrame to our original DataFrame 
X_train = pd.concat([X_train.reset_index(drop = True),X_train_encoded.reset_index(drop = True)], axis = 1)    
 
# Drop the original categorical variable columns
X_train.drop(categorical_vars, axis = 1, inplace = True)           

X_test_encoded = pd.DataFrame(X_test_encoded, columns = encoder_feature_names)
X_test = pd.concat([X_test.reset_index(drop = True),X_test_encoded.reset_index(drop = True)], axis = 1)    
X_test.drop(categorical_vars, axis = 1, inplace = True)           

###############################################################################
# Feature Selection
###############################################################################

clf = LogisticRegression(random_state = 42, max_iter = 1000)           # max_iter: number of iterations to find the optimum regression line (default: 100)
feature_selector = RFECV(clf)                                          # We can determine number of chunks (default:5 meaning that it splits the data to 5 equal size chunks, runs the model over 4 chunks and validate it over the remaining one)

fit = feature_selector.fit(X_train,y_train)

# Finding the optimum number of variables
optimal_feature_count = feature_selector.n_features_
print(f'optimal number of features: {optimal_feature_count}')

# Dynamically updating X DataFrame to contain only the new variables
X_train = X_train.loc[:,feature_selector.get_support()]
X_test = X_test.loc[:,feature_selector.get_support()]

# Visualizing the results in case required
plt.plot(range(1, len(fit.cv_results_['mean_test_score']) + 1), fit.cv_results_['mean_test_score'], marker = 'o')
plt.ylabel('Model Score')
plt.xlabel('Number of Features')
plt.title(f"Feature Selection using RFE \n Optimal number of features is {optimal_feature_count} (at score of {round(max(fit.cv_results_['mean_test_score']),4)})")
"""
\n: goes to next line
round(), 4: round to 4 decimal places
"""
plt.tight_layout()
plt.show()

###############################################################################
# Model Training
###############################################################################

clf = LogisticRegression(random_state = 42, max_iter = 1000) 
clf.fit(X_train, y_train)

###############################################################################
# Model Assessment
###############################################################################

y_pred_class = clf.predict(X_test)
y_pred_prob = clf.predict_proba(X_test)[:,1]         # probability of falling a data point in each class of 0 or 1
                                                     # Select class 1 only

# Classification Report
print(classification_report(y_test,y_pred_class))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_class)
print(conf_matrix)

plt.style.use('seaborn-poster')
plt.matshow(conf_matrix, cmap = 'coolwarm')     # cmap: color map
plt.gca().xaxis.tick_bottom()                   # transfer xaxis notes to the bottom
plt.title('Confusion Matrix')
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
for (i, j), corr_value in np.ndenumerate(conf_matrix):
    plt.text(j, i, corr_value, ha = 'center', va = 'center', fontsize = 20)     # horizontal and vertical alignments
plt.show()

# Accuracy (the number of correct classification out of all attempted classifications)
accuracy_score(y_test, y_pred_class) 

# Precision (of all observations that were predicted as positive, how many were actually positive)
precision_score(y_test, y_pred_class) 

# Recall (of all positive observations, how many did we predict as positive)
recall_score(y_test, y_pred_class) 

# F1-Score (the harmonic mean of precision and recall)
f1_score(y_test, y_pred_class) 

###############################################################################
# Finding the optimal threshold
###############################################################################

thresholds = np.arange(0, 1, 0.01)

precision_scores = []
recall_scores = []
f1_scores = []

for threshold in thresholds:
    
    pred_class = (y_pred_prob >= threshold) * 1                                    # puts 1 if the () is True and 0 if it is False
    
    precision = precision_score(y_test, pred_class, zero_division = 0)             # zero_division = 0: we won't get error when the threshhold is too high or too low and no data points fall into one of these classes
    precision_scores.append(precision)
    
    recall = recall_score(y_test, pred_class)             
    recall_scores.append(recall)
    
    f1 = f1_score(y_test, pred_class)             
    f1_scores.append(f1)

# Finding max F1-score

max_f1 = max(f1_scores)
max_f1_idx = f1_scores.index(max_f1)

# Plotting all scores
plt.style.use('seaborn-poster')
plt.plot(thresholds, precision_scores, label = 'Precision', linestyle = '--')
plt.plot(thresholds, recall_scores, label = 'Recall', linestyle = '--')
plt.plot(thresholds, f1_scores, label = 'F1', linewidth = 5)
plt.title(f'Finding the Optimal Threshold for Classification Model \n Max F1: {round(max_f1,2)}(Threshold = {round(thresholds[max_f1_idx], 2)})')
plt.xlabel('Threshold')
plt.ylabel('Assessment Score')
plt.legend(loc = 'lower left')
plt.tight_layout()
plt.show()

# Finding class for optimal threshold
optimal_threshold = thresholds[max_f1_idx]
y_pred_class_threshold = (y_pred_prob >= optimal_threshold) * 1  




