# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 08:04:23 2017

@author: salmon

This is an example of classifier program, we tried to use the KNN classifier 
but we never got better results than the M2. (We could only get get better 
results when using their classifier, just by modifying the value of alpha 
they had chosen.) The test for the class is included in the 'main'.

IMPORTANT: when you submit your solution to Codalab, the program run.py 
should be able to find your classifier. Currently it loads "classifier.py"
from the sample_code/ directory. If you do not want to modify run.py, 
copy all your code to sample_code/ and rename zClassifier.py to Classifier.py.
Alternatively, make sure that the path makes it possible
to find your code and that you import your own version of "Classifier":
in run.py, add
lib_dir2 = os.path.join(run_dir, "my_code")
path.append (lib_dir2)
and change 
from data_manager import DataManager 
from classifier import Classifier    
to
from zDataManager import DataManager 
from zClassifier import Classifier    
"""

import numpy as np
from sys import argv

from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.multiclass import OneVsRestClassifier
import pickle
from sklearn.svm import SVC

def ebar(score, sample_num):
    '''ebar calculates the error bar for the classification score (accuracy or error rate)
    for sample_num examples'''
    return np.sqrt(1.*score*(1-score)/sample_num)
    
    
class Classifier(BaseEstimator):
    def __init__(self):
        pass 
        
    def fit(self, X, y, alfa):
        print "dans classifier.fit :\t X: ",X.shape, "y: ", y.shape
        self.clf = BernoulliNB(alpha=alfa)
        self.clf = OneVsRestClassifier(self.clf).fit(X,y)
        ## self.clf = OneVsRestClassifier(SVC(kernel='linear')).fit(X,y)
        return self.clf

    def predict(self, X):
        ''' This is called to make predictions on test data. Predicted classes are output.'''
        return self.clf.predict(X)

    def predict_proba(self, X):
        ''' Similar to predict, but probabilities of belonging to a class are output.'''
        return self.clf.predict_proba(X) # The classes are in the order of the labels returned by get_classes
        
    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "w"))

    def load(self, path="./"):
        self = pickle.load(open(path + '_model.pickle'))
        return self
    
    def score(self,X,y):
        return self.score

class Classifier2(BaseEstimator):
    def __init__(self):
        pass 
    
    def fit(self, X, y, maxn, tr_samples):
        print "dans classifier2.fit : ", "X: ",X.shape, "y: ", y.shape
        self.clf = KNeighborsClassifier(n_neighbors=maxn, weights='distance').fit(X[:tr_samples],y[:tr_samples])
        return self.clf

    def predict(self, X):
        ''' This is called to make predictions on test data. Predicted classes are output.'''
        return self.clf.predict(X)

    def predict_proba(self, X):
        ''' Similar to predict, but probabilities of belonging to a class are output.'''
        return self.clf.predict_proba(X) # The classes are in the order of the labels returned by get_classes
        
    def score(self,X,y):
        return self.score

if __name__=="__main__":
    # We can use this to run this file as a script and test the Classifier
    if len(argv)==1: # Use the default input and output directories if no arguments are provided
        input_dir = "../public_data"
        output_dir = "../res"
    else:
        input_dir = argv[1]
        output_dir = argv[2];
        
    from zDataManager import DataManager # The class provided by binome 1
    
    basename = 'movies'
    D = DataManager(basename, input_dir) # Load data
    print D
    
    classifier_dict = {
            'KNN': Classifier2(),
            'BernoulliNB': Classifier()
            }
    
    for key in classifier_dict:
        myclassifier = classifier_dict[key]
        
        # Train
        Ytrue_tr = D.data['Y_train']
        X = D.data['X_train']
        ## print "here : ", D.data['X_train'].shape, "type: ", type(X)
        nums=X.shape[0]/2
        Ytrtr=Ytrue_tr[nums:]
        
        ## print "cest ici :\n",Ytrue_tr,"\n",len(Ytrue_tr), len(Ytrue_tr[0])
        nclass = len(Ytrue_tr[0,:])
        Yonehot_tr = np.zeros([len(Ytrtr),nclass])
        for i, item in enumerate(Ytrtr):
            # print i,'\t',item,len(item),item.shape
            for ii in item:
                if(ii==1.):    Yonehot_tr[i,ii.astype(int)]=1
        print '\n', len(Yonehot_tr),'\t', Yonehot_tr.shape
                       
        from libscores import f1_metric, f1_binary_score, f1_multiclass_score, f1_sklearn
        from sklearn.metrics import accuracy_score 
        
        alfa=0.000  # <=> no smoothing
        neigh=20
        # change the following line of code by what's next to it to see the
        # alternative (much longer though and less efficient)
        
        if(key!='KNN'):
            myclassifier.fit(X, Ytrue_tr, alfa)
        else:
            myclassifier.fit(X, Ytrue_tr, neigh, nums)       
                                  
        # Making classification predictions (the output is a vector of class IDs)
        Ypred_tr = myclassifier.predict(X)
        print "\n Ytrue_tr[:nums] \n",Ytrue_tr[:nums],"\n predictions: \n", Ypred_tr[:nums]
        Yprtr=Ypred_tr[nums:]
        print "\n complémentaire de Ytrue_tr[:nums] : \n",Ytrtr,"\n prédictions associées : \n", Yprtr
        Ypred_va = myclassifier.predict(D.data['X_valid'])
        Ypred_te = myclassifier.predict(D.data['X_test'])  
        
        # Making probabilistic predictions (each line contains the proba of belonging in each class)
        Yprob_tr = myclassifier.predict_proba(D.data['X_train'])
        Yprob_va = myclassifier.predict_proba(D.data['X_valid'])
        Yprob_te = myclassifier.predict_proba(D.data['X_test'])
        
        # Training success rate and error bar:
        # First the regular accuracy (fraction of correct classification   
        acc = accuracy_score(Ytrue_tr, Ypred_tr)
        ## print Yonehot_tr.shape[1],'\t',type(Yonehot_tr[1])
        ## print '''Yonehot_tr''','\n', len(Yonehot_tr), '\n','\n', Yprob_tr,'\n',len(Yprob_tr),type(Yprob_tr),type(Yprob_tr[0])
        f1m = f1_metric(Ytrue_tr, Ypred_tr, task='multilabel.classification')
        f1b=f1_binary_score(Ytrue_tr, Ypred_tr)
        f1ms=f1_multiclass_score(Ytrue_tr, Ypred_tr)
        f1sk=f1_sklearn(Ytrue_tr, Ypred_tr)
        print "classifier\tacc\tf1m\tf1b\tf1ms\tf1sk\talpha\tneigh"
        print "%s\t%5.2f+-%5.3f %5.2f\t%5.2f\t%5.2f\t%5.2f\t%5.3f\t%d" % (key, acc, ebar(acc, Ytrtr.shape[0]), f1m, f1b, f1ms, f1sk, alfa, neigh)
        
        print "here is a sample of the predictions:"
        for e in Yprtr[len(Yprtr)-25:]:
            print "%.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f" %(e[0],e[1],e[2],e[3],e[4],e[5],e[6],e[7],e[8],e[9],e[10],e[11],e[12],e[13])
    # Note: we do not know Ytrue_va and Ytrue_te
    # See zClassifierTest for a better evaluation using cross-validation
