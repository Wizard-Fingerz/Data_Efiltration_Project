from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,accuracy_score
from sklearn import metrics





def home(request):
    return render(request, 'home.html')


def predict(request):
    return render(request, 'predict.html')


def result(request):
    df = pd.read_csv('benefit_metrics_dataset/Data_exfiltration_Dataset_FINAL_UPDATED.csv')
    dataset = df.dropna()
    X = dataset.iloc[:, 0:23].values
    y = dataset.iloc[:, 23].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    

    # Random Forest
    rf = RandomForestClassifier()
    rf.fit(X_train.reshape(-1,1), y_train)
    rf_y_pred = rf.predict(X_test)
    
    # decision Tree
    tree = DecisionTreeClassifier()
    tree.fit(X_train, y_train)
    tree_y_pred = tree.predict(X_test)
    
    # Ada boost
    ada = AdaBoostClassifier(random_state = 1)
    ada.fit(X_train, y_train)
    ada_y_pred = ada.predict(X_test)
    
    # Support Vector
    svc =  SVC()
    svc.fit(X_train, y_train)
    svc_y_pred = svc.predict(X_test)
    
    # Gradient Boosting 
    gb = GradientBoostingClassifier()
    gb.fit(X_train, y_train)
    gb_y_pred = gb.predict(X_test)
    
    # SDG Classifier
    sgd =  SGDClassifier()
    sgd.fit(X_train, y_train)
    sgd_y_pred = sgd.predict(X_test)
    
    # Logistic Regression
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    lr_y_pred = lr.predict(X_test)
    
    
    # get value from html
    stime = float(request.GET['stime'])
    flgs = float(request.GET['flgs'])
    proto = float(request.GET['proto'])
    sport = float(request.GET['sport'])
    dport = float(request.GET['dport'])
    pkts = float(request.GET['pkts'])
    bytes = float(request.GET['bytes'])
    state = float(request.GET['state'])
    ltime = float(request.GET['ltime'])
    seq = float(request.GET['seq'])
    dur = float(request.GET['dur'])
    mean = float(request.GET['mean'])
    stddev = float(request.GET['stddev'])
    sum = float(request.GET['sum'])
    min = float(request.GET['min'])
    max = float(request.GET['max'])
    spkts = float(request.GET['spkts'])
    dpkts = float(request.GET['dpkts'])
    rate = float(request.GET['rate'])
    srate = float(request.GET['srate'])
    drate = float(request.GET['drate'])
    
    
    
    ada_y_pred = ada.predict(np.array([stime, flgs, proto, sport, dport, pkts, bytes, state, ltime, seq, dur, mean, stddev, sum, min, max, spkts, dpkts, rate, srate, drate]))
    # rf_y_pred = rf.predict(np.array([stime, flgs, proto, sport, dport, pkts, bytes, state, ltime, seq, dur, mean, stddev, sum, min, max, spkts, dpkts, rate, srate, drate]))
    pred = round(ada_y_pred[0])
    
    result = 'The predicted value is $'+str(pred)
    return render(request, 'predict.html', {'result2':result})