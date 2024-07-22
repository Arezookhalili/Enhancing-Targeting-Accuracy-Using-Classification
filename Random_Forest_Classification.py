###############################################################################
# Random Forest for Classification - ABC Grocery Task
###############################################################################


###############################################################################
# Import Required Packages
###############################################################################

import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance

###############################################################################
# Import Sample Data
###############################################################################

# Import
data_for_model = pickle.load(open('Data/abc_classification_modelling.p', 'rb'))                 # 'rb': read file

# Drop unnecessary columns
data_for_model.drop('customer_id', axis = 1, inplace = True)

# Shuffle data
data_for_model = shuffle(data_for_model, random_state = 42)

# Class balance

data_for_model['signup_flag'].value_counts(normalize = True)               # Checking if the data are balanced

###############################################################################
# Deal with Missing Values
###############################################################################

data_for_model.isna().sum()
data_for_model.dropna(how = 'any', inplace = True)

###############################################################################
# Split Input Variables and Output Variables
###############################################################################

X = data_for_model.drop(['signup_flag'], axis = 1)
y = data_for_model['signup_flag']

###############################################################################
# Split out Training and Test Sets
###############################################################################

X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.2, random_state = 42, stratify = y)

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
# Model Training
###############################################################################

clf = RandomForestClassifier(random_state = 42, n_estimators = 500, max_features = 5)        # n_estimators: number of trees
                                                                                             # max_features: number of variables offered for splitting in each splitting point
                                                                                             # If we do not enter these parameters, the defaults will be applied (100, auto)
clf.fit(X_train, y_train)

###############################################################################
# Model Assessment
###############################################################################

y_pred_class = clf.predict(X_test)
y_pred_prob = clf.predict_proba(X_test)[:,1]         # class probability: probability of falling a data point in each class of 0 or 1
                                                     # Select class 1 only

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

# Feature importance (based on minimum gini score: tells us the importance of each input variable in the predictive power of our random forest model)

feature_importance = pd.DataFrame(clf.feature_importances_)
feature_names = pd.DataFrame(X.columns)
feature_importance_summary = pd.concat([feature_names, feature_importance], axis = 1)
feature_importance_summary.columns = ['input_variable', 'feature_importance']
feature_importance_summary.sort_values(by = 'feature_importance', inplace = True)

plt.barh(feature_importance_summary['input_variable'],feature_importance_summary['feature_importance'])        # Horizontal bar plot
plt.title('Feature Importance of Random Forest')
plt.xlabel('Feature Importance')
plt.tight_layout()
plt.show()

# Permutation importance (based on the decrease seen when we randomize the values of each input variable)

result = permutation_importance(clf, X_test, y_test, n_repeats = 10, random_state = 42)      # n_repeats: How many times we want to apply random shuffling on each input variable

permutation_importance = pd.DataFrame(result['importances_mean'])                                  # importances_mean: average of data we got over n_repeats of random shuffling
permutation_names = pd.DataFrame(X.columns)
permutation_importance_summary = pd.concat([feature_names, permutation_importance], axis = 1)
permutation_importance_summary.columns = ['input_variable', 'permutation_importance']
permutation_importance_summary.sort_values(by = 'permutation_importance', inplace = True)

plt.barh(permutation_importance_summary['input_variable'],permutation_importance_summary['permutation_importance'])        # Horizontal bar plot
plt.title('Permutation Importance of Random Forest')
plt.xlabel('Permutation Importance')
plt.tight_layout()
plt.show()












# Finding the best max depth

max_depth_list = list(range(1,15))
accuracy_scores = []

for depth in max_depth_list:
    
    clf = DecisionTreeClassifier(max_depth = depth, random_state = 42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = f1_score(y_test,y_pred)
    accuracy_scores.append(accuracy)
    
max_accuracy = max(accuracy_scores)
max_accuracy_idx = accuracy_scores.index(max_accuracy)
optimal_depth = max_depth_list[max_accuracy_idx]

# Plot of max depths
plt.plot(max_depth_list, accuracy_scores)
plt.scatter(optimal_depth, max_accuracy, marker = 'x', color = 'red')
plt.title(f'Accuracy (F1 Score) by Max Depth \n Optimal Tree Depth: {optimal_depth} (Accuracy: {round(max_accuracy, 4)})')
plt.xlabel('Max Depth of Decision Tree')
plt.ylabel('Accuracy (F1 Score)')
plt.tight_layout()
plt.show()

# Refit the model with max depth that gives us much more explainable model with good accuracy
"""
as the best max_depth = 9, we can run the code again to get better results)
"""

# Plot our model

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(25,15))
tree = plot_tree(clf,
                 feature_names = X.columns,
                 filled = True,
                 rounded = True,
                 fontsize = 24)