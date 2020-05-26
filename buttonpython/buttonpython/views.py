from django.shortcuts import render
import requests
import sys
from subprocess import call
import os


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def button(request):
    return render(request,'home.html')

def output(request):
    data = requests.get("https://www.google.com/")
    print(data.text)
    data = data.text
    return render(request,'home.html',{'data':data})

def external(request):
    inp = []
    for i in range(1,31):
        inp.append(request.POST.get('param'+str(i)))
        
    for i in range(1,31):
        inp[i-1]=int(inp[i-1])
      
    #out = run(sys.executable,['C:\\Users\\user\\Documents\\anaconda scripts\\test.py',inp],shell=False,stdout=PIPE)
    
    data = pd.read_csv(os.path.join(BASE_DIR, 'data.csv'))

    data['diagnosis'] = data['diagnosis'].apply(lambda x: '1' if x == 'M' else '0')
    data = data.set_index('id')

    Y = data['diagnosis'].values
    X = data.drop('diagnosis', axis=1).values

    X_train, X_test, Y_train, Y_test = train_test_split (X, Y, test_size = 0.20, random_state=21)

    # prepare the model
    '''with warnings.catch_warnings():
        warnings.simplefilter("ignore")'''
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    model = SVC(C=2.0, kernel='rbf')
    start = time.time()
    model.fit(X_train_scaled, Y_train)
    end = time.time()
    print( "Run Time: %f" % (end-start))
    
    
    inp2 = np.reshape(inp,(1,30))
    inp_scaled = scaler.transform(inp2)



    # estimate accuracy on test dataset
    ''''with warnings.catch_warnings():
        warnings.simplefilter("ignore")'''
    #X_test_scaled = scaler.transform(X_test)
    prediction = model.predict(inp_scaled)
    print(prediction)
    if prediction[0] is '1':
        value = "Malignant"
    else:
        value = "Benign"
    return render(request,'home.html',{'data1':value})
