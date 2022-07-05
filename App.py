import pandas as pd
import os
import numpy as np
from numpy.random import seed
from numpy.random import randn
from sklearn import linear_model
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.graphics.gofplots import qqplot
import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import seaborn as sns
sns.set(style="whitegrid")
import math
import csv
import time
import scipy.stats as stats
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score, mean_absolute_error

# os.chdir('C:\Users\user\Desktop\PythonProject')

dataset = pd.read_csv("train_data.csv")
prod_price = pd.read_csv("product_prices.csv")
date_week=pd.read_csv("date_to_week_id_map.csv")

df=pd.merge(prod_price,date_week, on=['week_id'], how='inner')
dataset=pd.merge(dataset,df, on=['date','product_identifier','outlet'], how='inner')

col = ['category_of_product', 'state']
dataset[col] = dataset[col].astype('category')

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
dataset['state_encoded'] = LabelEncoder().fit_transform(dataset['state'])
dataset['cat_prod_encoded'] = LabelEncoder().fit_transform(dataset['category_of_product'])
dataset['Month'] = pd.to_datetime(dataset['date']).dt.month
dataset = dataset.drop(columns=["date","week_id","state","category_of_product"])

from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

def knn():
     knn = KNeighborsRegressor(n_neighbors=10)
     return knn
    
def extraTreesRegressor():
     clf = ExtraTreesRegressor(n_estimators=100,max_features='auto', verbose=1, n_jobs=1)
     return clf
def randomForestRegressor():
     clf = RandomForestRegressor(n_estimators=100,max_features='log2', verbose=1)
     return clf

def svm():
     clf = SVR(kernel='rbf', gamma='auto')
     return clf

def predict_(m, test_x):
     return pd.Series(m.predict(test_x))

def model_():
     return extraTreesRegressor()


def train_(train_x, train_y):
     m = model_()
     m.fit(train_x, train_y)
     return m

def train_and_predict(train_x, train_y, test_x):
     m = train_(train_x, train_y)
     return predict_(m, test_x), m
    
def calculate_error(test_y, predicted, weights):
     return mean_absolute_error(test_y, predicted, sample_weight=weights)

from sklearn.model_selection import KFold

kf = KFold(n_splits=5)
splited = []

for name, group in dataset.groupby(["outlet", "department_identifier"]):
     group = group.reset_index(drop=True)
     trains_x = []
     trains_y = []
     tests_x = []
     tests_y = []
     if group.shape[0] <= 5:
         f = np.array(range(5))
         np.random.shuffle(f)
         group['fold'] = f[:group.shape[0]]
         continue
     fold = 0
     for train_index, test_index in kf.split(group):
         group.loc[test_index, 'fold'] = fold
         fold += 1
     splited.append(group)

splited = pd.concat(splited).reset_index(drop=True)

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error

best_model = None
error_cv = 0
best_error = np.iinfo(np.int32).max
for fold in range(1):
     dataset_train = splited.loc[splited['fold'] != fold]
     dataset_test = splited.loc[splited['fold'] == fold]
     train_y = dataset_train['sales']
     train_x = dataset_train.drop(columns=['sales', 'fold'])
     test_y = dataset_test['sales']
     test_x = dataset_test.drop(columns=['sales', 'fold'])
     print(dataset_train.shape, dataset_test.shape)
     predicted, model = train_and_predict(train_x, train_y, test_x)
     weights = test_x['outlet'].replace(True, 5).replace(False, 1)
     error = calculate_error(test_y, predicted, weights)
     error_cv += error     
     print(fold, error)
     if error < best_error:
         print('Find best model')
         best_error = error
         best_model = model
error_cv /= 5

if error < best_error:
         print('Find best model')
         best_error = error
         best_model = model

dataset_test = pd.read_csv("test_data.csv")
prod_price = pd.read_csv("product_prices.csv")
date_week=pd.read_csv("date_to_week_id_map.csv")

df=pd.merge(prod_price,date_week, on=['week_id'], how='inner')
dataset_test=pd.merge(dataset_test,df, on=['date','product_identifier','outlet'], how='inner')


col = ['category_of_product', 'state']
dataset_test[col] = dataset_test[col].astype('category')

dataset_test['state_encoded'] = LabelEncoder().fit_transform(dataset_test['state'])
dataset_test['cat_prod_encoded'] = LabelEncoder().fit_transform(dataset_test['category_of_product'])


dataset_test['Month'] = pd.to_datetime(dataset_test['date']).dt.month
dataset_test = dataset_test.drop(columns=["date","week_id","state","category_of_product","id"])

predicted_test = best_model.predict(dataset_test)
test=dataset_test

dataset_test['sales'] = predicted_test

dataset_test['id'] = dataset_test['outlet'].astype(str) + '_' +  dataset_test['department_identifier'].astype(str)
dataset_test = dataset_test[['id', 'sales']]
dataset_test = dataset_test.rename(columns={'id': 'Id', 'sales': 'sales'})

dataset_test.to_csv('mragpavkum.csv', index=False)

import pickle
pickle.dump(best_model, open( "model.pkl", "wb" ) )
model = open('model.pkl', 'rb')     
model= pickle.load(model)

print(model.predict([[74,11,111,3.43,1,2,3]]))



from flask import Flask, redirect, request, render_template

app = Flask(__name__)
model = open('model.pkl', 'rb')     
model= pickle.load(model)

@app.route('/')
def login():
    return render_template('page-login.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/handle_login')
def login_check():
    return render_template('index.html')

@app.route('/stock/login')
def stockLogin():
    return render_template('stock/login.html')

@app.route('/graph')
def viewGraph():
    return render_template('graph/graph.html')

@app.route('/calender')
def viewCalender():
    return render_template('calender.html')


# @app.route('/predict', methods=['GET','POST'])
# def predict():
#   rainfall = request.form['rainfall']
#   rainfall = float(rainfall)
#   humidity = request.form['humidity']
#   humidity = float(humidity)
#   rain_today = request.form['rain_today']
#   rain_today = Boolean(rain_today)
#   pressure = request.form['pressure']
#   pressure =float(pressure)

#   return render_template('index.html', prediction = "Tomorrow will be a Rainy day")

if __name__ == '__main__':
   app.run(debug=True)






































