import sys
sys.path.append("/home/maciej/repos/scikit-learn")

import cv2
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from dataset_io import *
import numpy as np
from joblib import dump, load
import timeit
import random
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt

def get_gradient_data(train_scenes, test_scenes):
    def get_samples(feature, scenes, y_val):
        X = [np.load(get_existing_clf_ds_filepath(feature, s)) for s in scenes]
        X = np.concatenate(X)
        X=[np.dstack(x) for x in X]
        X=[x.flatten() for x in X]
        y=np.full(len(X), y_val)
        return X,y

    train_pallet_X, train_pallet_y=get_samples("pallet_rectangles_gradient", train_scenes, 1)
    train_background_X, train_background_y=get_samples("background_rectangles_gradient", train_scenes, 0)

    test_pallet_X, test_pallet_y=get_samples("pallet_rectangles_gradient", test_scenes, 1)
    test_background_X, test_background_y=get_samples("background_rectangles_gradient", test_scenes, 0)

    X_train = np.vstack((train_pallet_X,train_background_X))
    X_test = np.vstack((test_pallet_X,test_background_X))
    y_train = np.hstack((train_pallet_y,train_background_y))
    y_test = np.hstack((test_pallet_y,test_background_y))
    
    return X_train, X_test, y_train, y_test
    
    
def classify_gradient(train_scenes, test_scenes, fold_name, persist=False):
    X_train, X_test, y_train, y_test = get_gradient_data(train_scenes, test_scenes)
    clf = RandomForestClassifier(128, n_jobs=-1) #128 is as fast as lower sizes and as accurate as greater sizes
    clf.fit(X_train, y_train)
    y_pred=clf.predict(X_test)

    acc=accuracy_score(y_test, y_pred)
    prec=precision_score(y_test, y_pred)
    rec=recall_score(y_test, y_pred)
    f1=f1_score(y_test, y_pred)
    
    if persist:
        dump(clf, "models/rand_forest_clf_{}.joblib".format(fold_name)) 
    #print("acc: ",acc," f1: ",f1," prec: ",prec," rec: ",rec)
    return acc, prec, rec, f1, clf

def get_color_data(train_scenes, test_scenes, color):
    def get_samples(feature, scenes, y_val, color=None):
        X = [np.load(get_existing_clf_ds_filepath(feature, s, color), allow_pickle=True) for s in scenes]
        X = [x for x in X if len(x) > 0]
        X = np.concatenate(X)
        X = np.array([np.hstack(x) for x in X])
        X= np.concatenate(X, 0)
        y=np.full(len(X), y_val)
        return X,y

    print(1)
    train_pallet_X, train_pallet_y=get_samples("pallets_color", train_scenes, 1, color)
    train_background_X, train_background_y=get_samples("backgrounds_color", train_scenes, 0)

    print(2)
    test_pallet_X, test_pallet_y=get_samples("pallets_color", test_scenes, 1, color)
    test_background_X, test_background_y=get_samples("backgrounds_color", test_scenes, 0)

    print(3)
    train_background_X_indices=range(len(train_background_X))
    print(31)
    count=min(len(train_pallet_X)*2, len(train_background_X_indices))
    indices=random.sample(train_background_X_indices, count)
    print(32)
    train_background_X_indices=train_background_X[indices]
    
    print(4)
    test_background_X_indices=range(len(test_background_X))
    count=min(len(test_pallet_X)*2, len(test_background_X_indices))
    indices=random.sample(test_background_X_indices, count)
    test_background_X_indices=test_background_X[indices]
    
    print(5)
    X_train = np.vstack((train_pallet_X,train_background_X))
    X_test = np.vstack((test_pallet_X,test_background_X))
    y_train = np.hstack((train_pallet_y,train_background_y))
    y_test = np.hstack((test_pallet_y,test_background_y))
    print(6)
    
    return X_train, X_test, y_train, y_test

def classify_color(color, train_scenes, test_scenes, fold_name, persist):  
    X_train, X_test, y_train, y_test=get_color_data(train_scenes, test_scenes, color)
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    print(7)
    X_train=None
    y_train=None
    y_pred=clf.predict(X_test)
    print(8)
    X_test=None

    acc=accuracy_score(y_test, y_pred)
    f1=f1_score(y_test, y_pred)
    prec=precision_score(y_test, y_pred)
    rec=recall_score(y_test, y_pred)
    #cm=confusion_matrix(y_test, y_pred, labels=[True, False])
    print(9)

    if persist:
        dump(clf, 'models/naive_bayes_clf_{}_{}.joblib'.format(color, fold_name))
        print(10)
    #     print("acc: ",acc," f1: ",f1," prec: ",prec," rec: ",rec)
    #     print(cm)
    return acc, prec, rec, f1, clf

def classify_colors(train_scenes, test_scenes, fold_name, persist=False):
    res=[]
    for color in COLORS:
        print(color)
        acc, prec, rec, f1, _ = classify_color(color, train_scenes, test_scenes, fold_name, persist)
        res.append([acc, prec, rec, f1])
    return res