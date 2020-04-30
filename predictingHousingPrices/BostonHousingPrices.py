#part 1 - GETTING SET UP

# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)
# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

#important imports
import pandas as pd
import numpy as np
import os
from six.moves import urllib # support URL download
import pandas as pd
np.random.seed(42)


#Part 2 - GETTING THE DATA.

#get the housing prices in link to raw data on my github
url = 'https://raw.githubusercontent.com/adssoccer1/machineLearning2019/master/housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataSet = pd.read_csv(url, delim_whitespace=True, names=names)

#Part 3 - EXPLORING THE DATA. Comment and uncomment where desired.

#dataSet.head(10)
#dataSet.info()
#dataSet.describe()

#histogram
#import matplotlib.pyplot as plt
#dataSet.hist(bins=50, figsize=(20,15))
#plt.show()

#Part 4 - CLEANING AND GOING DEEPER INTO THE DATA.


# Use corr() to see the correlations with MEDV to see which features are relevant.
corr_matrix = dataSet.corr()
corr_matrix["MEDV"].sort_values(ascending=False)

#we see LSTAT and RM are parituclarly important features - lets vizualize them.
dataSet.plot(kind="scatter", x="LSTAT", y="RM", alpha=0.1)

#now we split the training and test set
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(dataSet, test_size=0.2, random_state=42)

# Create the features and labels for training
housing = train_set.drop("MEDV", axis=1) # drop labels for training set
housing_labels = train_set["MEDV"].copy()

# Check to see if there's any missing value in the data with .isnull()
sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()

#Part 5 - TRAIN THE DATA USING LINEAR REGRESSION, DECISION TREES AND RANDOM FOREST

#using linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing, housing_labels)
print(lin_reg)


#use rmse
from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
print("mean squared error: " , lin_mse)
lin_rmse = np.sqrt(lin_mse)
print("root mean squared error: " , lin_rmse)
from sklearn.metrics import mean_absolute_error
lin_mae = mean_absolute_error(housing_labels, housing_predictions)
print("mean squared error: " , lin_mae)

#use cross validatoin scores for linear regression
from sklearn.model_selection import cross_val_score
lin_scores = cross_val_score(lin_reg, housing, housing_labels, scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
display_scores(lin_rmse_scores)

#using decision tree
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing, housing_labels)

housing_predictions = tree_reg.predict(housing)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)

#use cross validation scores.
from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing, housing_labels, scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
print()
print("Decision tree scores below: ")
display_scores(tree_rmse_scores)

#finally use random forest
print()
print("Random Forest scores below: ")
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(housing, housing_labels)

#use cross validation scores
forest_scores = cross_val_score(forest_reg, housing, housing_labels, scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)

print("\n", "So random forest seems to produce the smallest error.")

# part 6 - FINE TUNING THE MODEL.

from sklearn.model_selection import GridSearchCV

param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of ? rounds of training
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(housing, housing_labels)

grid_search.best_params_
grid_search.best_estimator_
cvres = grid_search.cv_results_
# zip() is to combine the column "mean_test_score" with struct "params"
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

pd.DataFrame(grid_search.cv_results_)

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {
    'n_estimators': randint(low=1, high=200),
    'max_features': randint(low=1, high=8),
}

forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
rnd_search.fit(housing, housing_labels)

cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances

# finally using on test set
final_model = grid_search.best_estimator_
X_test = test_set.drop("MEDV", axis=1)
y_test = test_set["MEDV"].copy()
final_predictions = final_model.predict(X_test)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print("final_mse: ", final_mse)
print("final_rmse: ", final_rmse)

print("So final rmse is: ", final_rmse, "!!!!")
