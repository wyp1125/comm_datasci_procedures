import pandas as pd
import numpy as np
import argparse
import os
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report

parser = argparse.ArgumentParser(description='This program assesses machine learning models for a cross validation dataset')
parser.add_argument('-i', '--indir', type=str, required=True, help="directory for cross-validation")
parser.add_argument('-m', '--mlmodel', type=int, required=True, help="ml model choice - 0:Nearest Neighbors, 1:Linear SVM, 2:RBS SVM, 3:Decision Tree, 4:Random Forest, 5:Neural Net, 6:AdaBoost, 7:Naive Bayes")
args = parser.parse_args()

clsf_names=["Nearest Neighbors", "Linear SVM", "RBF SVM",
            "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
            "Naive Bayes"]

classifiers = [KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB()]

if args.mlmodel<0 or args.mlmodel>7:
    print("Classifier ID should be between 0 and 7")
    exit()

clsf = classifiers[args.mlmodel]
print("Chosen ML model: "+str(clsf))
i=0
trainxfile=os.path.join(args.indir,"train.X."+str(i))
trainyfile=os.path.join(args.indir,"train.Y."+str(i))
testxfile=os.path.join(args.indir,"test.X."+str(i))
testyfile=os.path.join(args.indir,"test.Y."+str(i))
ave_acc=0
while os.path.isfile(trainxfile):
  x_trn=pd.read_csv(trainxfile,header=0)
  y_trn=pd.read_csv(trainyfile,header=0).iloc[:,0]
  x_tst=pd.read_csv(testxfile,header=0)
  y_tst=pd.read_csv(testyfile,header=0).iloc[:,0]
  model=clsf.fit(x_trn,y_trn)
  y_pred=model.predict(x_tst)
  #print(testyfile)
  #print(x_tst.shape)
  #print(y_tst.shape)
  acc=float((y_pred==y_tst).sum())/float(len(y_tst))
  ave_acc=ave_acc+acc
  #print("\nModel accuracy: {0: .3f}%\n".format(acc*100))
  #print(classification_report(y_tst,y_pred))
  i=i+1
  trainxfile=os.path.join(args.indir,"train.X."+str(i))
  trainyfile=os.path.join(args.indir,"train.Y."+str(i))
  testxfile=os.path.join(args.indir,"test.X."+str(i))
  testyfile=os.path.join(args.indir,"test.Y."+str(i))
ave_acc=ave_acc/i
print(ave_acc)
