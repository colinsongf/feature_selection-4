# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 22:50:16 2016

@author: jianbo
"""
import math
from sklearn import ensemble
import pandas as pd
import numpy as np
from copy import deepcopy
 
def getsigma(y,f):
    n = len(f)
    mae = sum(abs(y-f))/n
    std = math.sqrt(2*mae*mae)
    count = 0
    mae = 0
    for i in range(n):
        if (abs(y[i] - f[i])) > 5*std:
            count = count + 1
        else:
            mae = mae + (f[i] - y[i])**2
    s = mae/(n - count)
    s = math.sqrt(s)
    s = max(s, 1e-8)
    return s
     
def feaSelect(x, y, method, th, param, flagCR = 'Classifier'):
    np.rand.seed(9001)
    if method in {'fspp', 'randomforest'}:
        print method, ' is running in feature selection'
    else:
        print '*** Wrong method ***'
        return None
    xcol = x.columns
    if flagCR == 'Regressor':
        clf = ensemble.RandomForestRegressor(**param)
    elif flagCR == 'Classifier':
        clf = ensemble.RandomForestClassifier(**param)
    clf.fit(x,y)
    if method == 'randomforest':
        fscore = clf.feature_importances_
    if method == 'fspp':
        if flagCR == 'Classifier':
            d = x.shape[1]
            n = x.shape[0]
            dhat = x.shape[1]
            order_s = np.random.permutation(n)
            xx = deepcopy(x)
            fscore = np.zeros(d)
            iset = range(d)
            for ell in range(d):
                print "RFE loop %d" % ell
                S = np.zeros(dhat)
                clf.fit(xx,y)
                f1 = clf.predict_proba(xx)
                xxcol = list(xx.columns)
                for j in range(dhat):
                    tx = deepcopy(xx)
                    txj = xx[xxcol[j]]
                    a = txj.iloc[order_s]
                    a = a.reset_index(drop = True)
                    tx[xxcol[j]] = a
                    f2 = clf.predict_proba(tx)
                    thi = abs(f1[:,1] - f2[:,1])
                    S[j] = sum(thi)
                index = np.argsort(S)
                xx = xx.drop(xxcol[index[0]], axis = 1)
                fscore[iset[index[0]]]  = (ell+1)*1000
                del(iset[index[0]])
                dhat = dhat -1
        if flagCR == 'Regressor':
            d = x.shape[1]
            n = x.shape[0]
            dhat = x.shape[1]
            order_s = np.random.permutation(n)
            xx = deepcopy(x)
            fscore = np.zeros(d)
            iset = range(d)
            for ell in range(d):
                print "RFE loop %d" % ell
                S = np.zeros(dhat)
                clf.fit(xx, y)
                f1 = clf.predict(xx)
                s1 = getsigma(y, f1)
                xxcol = list(xx.columns)
                for j in range(dhat):
                    tx = deepcopy(xx)
                    txj = xx[xxcol[j]]
                    a = txj.iloc[order_s]
                    a = a.reset_index(drop = True)
                    tx[xxcol[j]]  = a
                    f2 = clf.predict(tx)
                    thi = abs(f1 - f2)
                    s2 = getsigma(y, f2)
                    S[j] = sum(thi*thi)/(s2**2) + n(s1**2)/(s2**2) + n*math.log((s2**2)/(s1**2))
                index = np.argsort(S)
                xx = xx.drop(xxcol[index[0]], axis = 1)
                fscore[iset[index[0]]]  = (ell+1)*1000
                del(iset[index[0]])
                dhat = dhat - 1
    frank  = np.argsort(fscore)
    n = len(fscore)
    frankscore = np.zeros((n,2))
    for i in range(n):
        frankscore[i,0] = frank[n-i-1]
        frankscore[i,1] = fscore[frank[n-1-i]]
    idx = frankscore[:,1] >th
    fselect = frankscore[idx,0]
    xcolselect  = xcol[fselect.tolist()]
    x = x.ix[:,xcolselect]
    feaScore = pd.DataFrame(columns = ['idx', 'features','score'])
    aa = frankscore[:,0]
    feaScore['idx'] = aa.astype(int)
    feaScore['features'] = xcol[aa.astype(int)]
    feaScore['score'] = frankscore[:,1]
    return({'x':x, 'xcolselect':xcolselect, 'frankscore':frankscore, 'feaScore':feaScore})
    