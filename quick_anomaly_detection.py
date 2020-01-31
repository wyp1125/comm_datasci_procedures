import os
import pandas as pd
import numpy as np
import argparse
from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score,precision_score,recall_score,f1_score,matthews_corrcoef

parser = argparse.ArgumentParser(description='This program quickly builds an outlier detection model on an imbalanced dataset for testing purpose')
parser.add_argument('-i', '--inprefix', type=str, required=True, help="prefix of feature & response files (.codedx & .codedy)")
parser.add_argument('-t', '--testsize', type=float, required=False, help="testing size (default: 0.2)")
parser.add_argument('-m', '--admodel', type=int, required=True, help="anomaly detection algorithm - 0:Robust covariance, 1:1class SVM, 2:Isolation Forest, 3:Local Outlier Factor")
args = parser.parse_args()
testratio=0.2
if args.testsize:
    testratio=args.testsize
feafile=args.inprefix+".codedx"
rspfile=args.inprefix+".codedy"
fea=pd.read_csv(feafile)
rsp=pd.read_csv(rspfile).iloc[:,0]
x_trn, x_tst, y_trn, y_tst = train_test_split(fea, rsp, test_size=testratio)

n_samples = len(x_trn)
cls_nums=np.array(np.unique(y_trn, return_counts=True)).T
ol_label=cls_nums[0,0]
ol_count=cls_nums[0,1]
il_label=cls_nums[1,0]
il_count=cls_nums[1,1]
if cls_nums[0,1]>cls_nums[1,1]:
    ol_label=cls_nums[1,0]
    ol_count=cls_nums[1,1]
    il_label=cls_nums[0,0]
    il_count=cls_nums[0,1]

outlier_fraction = ol_count/n_samples
print("Outlier fraction: {}".format(outlier_fraction))

clsf_names=["Robust covariance", "One-class SVM", "Isolation Forest","Local Outlier Factor"]
anomaly_algorithms = [EllipticEnvelope(contamination=outlier_fraction),
                      svm.OneClassSVM(nu=outlier_fraction, kernel="rbf",gamma=0.1),
                      IsolationForest(contamination=outlier_fraction,random_state=42),
                      LocalOutlierFactor(n_neighbors=35, contamination=outlier_fraction)]

if args.admodel<0 or args.admodel>3:
    print("Anomal detection algorithm ID should be between 0 and 3")
    exit()

clsf = anomaly_algorithms[args.admodel]
clsf.fit(x_trn)
if clsf_names[args.admodel] == "Local Outlier Factor":
    y_pred = clsf.fit_predict(x_tst)
else:
    y_pred = clsf.predict(x_tst)

y_pred_1=y_pred
for i in range(len(y_pred)):
    if y_pred[i]==-1:
        y_pred_1[i]=ol_label
    else:
        y_pred_1[i]=il_label

print(classification_report(y_tst,y_pred_1))
acc= accuracy_score(y_tst,y_pred_1)
print("Algorithm: {}".format(clsf_names[args.admodel]))
print("Accuracy: {}".format(acc))
prec= precision_score(y_tst,y_pred_1)
print("Precision: {}".format(prec))
rec= recall_score(y_tst,y_pred_1)
print("Recall: {}".format(rec))
f1= f1_score(y_tst,y_pred_1)
print("F1-Score: {}".format(f1))
MCC=matthews_corrcoef(y_tst,y_pred_1)
print("Matthews correlation coefficient: {}".format(MCC))

