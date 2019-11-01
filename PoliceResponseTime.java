

#So we want to get CallTime and find difference between OnSceneDate/Time to calculate the responsiveness time. We use 
#We train on dataSetTwo which has the Zone, priority, and CallTime as columns.
#DataSet is the original data that is maintained be used as a reference when necessary. 

import pandas as pd

#get the data prices in link to raw data on my github
url = 'https://raw.githubusercontent.com/adssoccer1/machineLearning2019/master/Police_Calls_for_Service1.csv'
names = ['CallType', 'Zone', 'CaseDisposition', 'priority', 'CallTime', 'EntryDate/Time', 'DispatchDate/Time', 'EnRouteDate/Time', 'OnSceneDate/Time']
dataSet = pd.read_csv(url, names=names)


#drop uneeded columns
dataSetTwo = dataSet.drop("CallType", axis=1)
dataSetTwo = dataSetTwo.drop("CaseDisposition", axis=1)
dataSetTwo = dataSetTwo.drop("EntryDate/Time", axis=1)
dataSetTwo = dataSetTwo.drop("DispatchDate/Time", axis=1)
dataSetTwo = dataSetTwo.drop("EnRouteDate/Time", axis=1)
dataSetTwo = dataSetTwo.drop("OnSceneDate/Time", axis=1)

#Helper Function: 
def get_sec(time_str):
    """Get Seconds from time."""
    counter = 0
    for x in time_str:
      if x == ":":
        counter += 1
    if counter == 2:
      h, m, s = time_str.split(':')
      return int(h) * 3600 + int(m) * 60 + int(s)
    elif counter == 1:
      h, m = time_str.split(':')
      return int(h) * 3600 + int(m) * 60
    else: 
      m = time_str
      return int(m) * 60
#Helper Function     
def getAverage(reponseTime):
  sum = 0 
  counter = 0
  for seconds in responseTime:
    if seconds > 0:
      sum += seconds
      counter += 1
  return sum / counter
    
#get a list of calltime converted to seconds
CallTime = dataSet["CallTime"].copy()
CallTimeList = [] #list of call times. 
for instance in CallTime:
  if type(instance) is float: 
    CallTimeList.append(-1111)
  else:
    splitInstance = instance.split()
    time = splitInstance[1]
    time = get_sec(time)#using the helper function
    CallTimeList.append(time) 
print("CallTimeList: ",CallTimeList[0:9])

#get a list of on scene time converted to seconds
OnSceneTime = dataSet["OnSceneDate/Time"].copy()
OnSceneTimeList = [] #list of arrival times. 
for instance in OnSceneTime:
  if type(instance) is float: 
    OnSceneTimeList.append(-1111)
  else:
    splitInstance = instance.split()
    time = splitInstance[1]
    time = get_sec(time)#using the helper function
    OnSceneTimeList.append(time)
print(OnSceneTimeList[0:9])


#do subtraction and make new y array with response time.

#get new responseTime list by subracting time called(in CallTimeList) vs time of arrival(in OnSceneTimeList) 
responseTime = []
for index in range(len(CallTimeList)):
  temp = OnSceneTimeList[index] - CallTimeList[index]
  if OnSceneTimeList[index] == -1111 or CallTimeList[index] == -1111:
    responseTime.append(-1111)
  elif temp < 0: 
    responseTime.append(-1111)
  else: 
    responseTime.append(OnSceneTimeList[index] - CallTimeList[index])

#For every -1111 or negative number we put into responseTime we need to replace with the average of all the other responseTime enteries
#so go through the response times and impute missing values with the average response time. 
average = getAverage(responseTime)#using the helper function
for index in range(len(responseTime)):
  if responseTime[index] <= 0:
    responseTime[index] = average
print(responseTime[0:19])

#prepare DataSetTwo
dataSetTwo["CallTime"] = CallTimeList
print(dataSetTwo.head(10))

#add response time to the dataSetTwo
df = pd.DataFrame(dataSetTwo)
df["responseTime"] = responseTime

#some data vizualization 
corr_matrix = dataSetTwo.corr()
print(corr_matrix)

dataSetTwo.plot(kind="scatter", x="CallTime", y="responseTime", alpha=0.1)
from sklearn.model_selection import train_test_split
import numpy as np

#helper function 
def display_scores(scores):
    print("Cross validation Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std(), "\n")



#test validity of dataSetTwo - we should return false and true
#print(np.any(np.isnan(dataSetTwo)))
#print(np.all(np.isfinite(dataSetTwo)))



#split dataSetTwo and responseTime to prepare for standard scaling and train_test_split
responseTime = dataSetTwo["responseTime"].copy()
dataSetTwo = dataSetTwo.drop("responseTime", axis=1)

#standard scaling the values from dataSet (columns Zone, priority, and CallTime )
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(dataSetTwo)
StandardScaler(copy=True, with_mean=True, with_std=True)
scaler.mean_
dataSetTwo = scaler.transform(dataSetTwo)

#check the dataSet looks okay
print(dataSetTwo)

#split dataSetTwo  after using standard scalar 
X_train, X_test, y_train, y_test = train_test_split(dataSetTwo, responseTime, test_size=0.2, random_state=42)

#train a linear regression model and print results
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)
lin_mse = mean_squared_error(y_test, y_pred)
print("LINEAR REGRESSION")
print("mean squared error: " , lin_mse)
lin_rmse = np.sqrt(lin_mse)
print("root mean squared error: " , lin_rmse)
lin_mae = mean_absolute_error(y_test, y_pred)
print("mean absolute error: " , lin_mae, "\n")
#cross validation on the linear regression model
from sklearn.model_selection import cross_val_score
lin_scores = cross_val_score(lin_reg, X_train, y_train, scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)


#train a decision tree model and print results
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(X_train, y_train)
y_pred = tree_reg.predict(X_test)
tree_mse = mean_squared_error(y_test, y_pred)
print("DECISION TREE")
print("mean squared error: " , tree_mse)
tree_rmse = np.sqrt(tree_mse)
print("root mean squared error: " , tree_rmse)
tree_mae = mean_absolute_error(y_test, y_pred)
print("mean absolute error: " , tree_mae, "\n")
#cross validation on the decision tree model
tree_scores = cross_val_score(tree_reg, X_train, y_train, scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-tree_scores)
display_scores(tree_rmse_scores)


#train a random forest model and print results
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(X_train, y_train)
y_pred = forest_reg.predict(X_test)
forest_mse = mean_squared_error(y_test, y_pred)
print("RANDOM FOREST")
print("mean squared error: " , forest_mse)
forest_rmse = np.sqrt(forest_mse)
print("root mean squared error: " , forest_rmse)
forest_mae = mean_absolute_error(y_test, y_pred)
print("mean absolute error: " , forest_mae, "\n")
#cross validation on the random forest model
forest_scores = cross_val_score(forest_reg, X_train, y_train, scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)


