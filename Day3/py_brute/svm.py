#!/usr/bin/env 
import numpy as np
import scipy as sp
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import fuzzywuzzy 
from fuzzywuzzy import fuzz,process
import re
import sklearn
import datetime
from sklearn.cross_validation import train_test_split,cross_val_score
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectPercentile,f_classif,chi2,f_regression
def processing(df):
    dummies_df = pd.get_dummies(df['City Group'])
    def add_CG(name):
        return 'CG_' + name
    dummies_df = dummies_df.rename(columns=add_CG)
    #print dummies_df.head()
    df = pd.concat([df, dummies_df.iloc[:,0]], axis=1)

    dummies_df = pd.get_dummies(df['Type'])
    def add_Type(name):
        return 'Type_' + name
    dummies_df = dummies_df.rename(columns=add_Type)
    df = pd.concat([df, dummies_df.iloc[:,0:3]], axis=1)

    # try to put in age as a column
    def add_Age(string):
        age=datetime.datetime.now()-datetime.datetime.strptime(string,"%m/%d/%Y")
        return age.days 

    df['Age']=df['Open Date'].map(add_Age)
    df=df.drop(['Id','Open Date','City','City Group','Type','revenue'],axis=1) 
    #scaler = StandardScaler().fit(df) 
    scaler = RobustScaler().fit(df) 
    df = scaler.transform(df)

    #print df.head()
    return df

def main():
    df_train=pd.read_csv("data/train.csv")
    df_test=pd.read_csv("data/test.csv")
    y=np.log(np.array(df_train["revenue"]))
    plt.hist(df_train['P28'])
    plt.show()
    return
    #test_id=df_test["Id"]
    df = pd.concat([df_train, df_test])
    df = processing(df)
    df_train = df[0:137,:]
    df_test = df[137:,:]
    X=df_train
    print X.shape,y.shape
    X_test=df_test
    model=SVR(kernel='linear')
    #samplesubmit.head()
    if(1):
        selector=SelectPercentile(f_regression,percentile=100)
        selector.fit(X,y)
        print selector.pvalues_
        scores=-np.log10(selector.pvalues_)
        scores/=scores.max()
        print scores
        feature_index_0=scores>0.1
    #model.C=1.e4
    #model.gamma=0.2
    X=X[:,feature_index_0]
    #print X.shape
    #model.fit(X,y)
    #y_pred=model.predict(X)
    #meanres=np.median(y-y_pred)
    #stdres=np.median(np.abs(y-y_pred-meanres))
    #print meanres,stdres
    #outlierindex=np.abs(y-y_pred-meanres)>5.*stdres
    #print len(y[outlierindex])
    #plt.plot(y-y_pred,'.')
    #plt.plot((y-y_pred)[outlierindex],'r.')
    #plt.show()
    #return
    #X=X[~outlierindex,:]
    #y=y[~outlierindex]
    #if(1):
    #    selector=SelectPercentile(f_regression,percentile=100)
    #    selector.fit(X,y)
    #    print selector.pvalues_
    #    scores=-np.log10(selector.pvalues_)
    #    scores/=scores.max()
    #    print scores
    #    feature_index_1=scores>0.1
    ##return
    if 0:
        #gammas=np.logspace(-7,7,50)
        #gammas=[0.1]
        #C_s=np.array([1])
        
            #return
        #model=SVR()
        #X=X[:,feature_index_1]
        C_s=np.logspace(-7,2,50)
        #C_s=np.logspace(0,5,50)
        scores_mean=[]
        scores_std=[]
        best_score=1.e20
        best_C=1
        best_gamma=1
        for C in C_s:
            if(1):
            #for gamma in gammas:
                model.C=C
                #model.gamma=gamma
                scores=cross_val_score(model,X,y,cv=10,scoring='mean_squared_error')
                scores_mean.append(np.sqrt(-np.mean(scores)))
                print np.sqrt(-np.mean(scores))
                #if np.sqrt(-np.mean(scores))<best_score:
                #    best_C=C
                #    best_gamma=gamma
                #    best_score=np.sqrt(-np.mean(scores))
                scores_std.append(np.std(scores))
        #plt.semilogx(gammas,scores_mean,'.') 
        #plt.plot(X[:,0],y-y_pred,'.')
        #plt.plot(X[:,1],y-y_pred,'.')
        #plt.plot(X[:,2],y-y_pred,'.')
        #plt.plot(X[:,3],y-y_pred,'.')
        plt.semilogx(C_s,np.array(scores_mean)/1.e6,'.')
        plt.show()
        #print best_C,best_gamma,best_score
        return 
    #
    #model=SVR()
    #X=X[:,feature_index_1]
    #X=X[:,feature_index_0]
    #model.gamma=0.1
    #model.C=1.5e3
    #model=SVR()
    #model.C=1.38949549437
    model.C=0.01
    #model.gamma=0.193069772888
    model.fit(X,y)
    #y_pred=model.predict(X)
    #plt.plot(y-y_pred,'.')
    #plt.show()
    #return
    X_test=X_test[:,feature_index_0]
    #X_test=X_test[:,feature_index_1]
    y_pred=model.predict(X_test)
    samplesubmit = pd.read_csv("data/sampleSubmission.csv")
    samplesubmit["Prediction"]=np.exp(y_pred)
    #samplesubmit.to_csv
    #samplesubmit.to_csv("data/submit_fistsvr.csv",index=False)
    #samplesubmit.to_csv("data/submit_linearsvr_fregression_age.csv",index=False)
    samplesubmit.to_csv("data/submit_svr_logrevenue.csv",index=False)
    #samplesubmit.to_csv("data/submit_linearsvr_fregression_outlierreject.csv",index=False)


    return

if __name__=='__main__':
    main()
