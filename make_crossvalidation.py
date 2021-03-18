import pandas as pd
import numpy as np
import argparse
import os
from sklearn.utils import shuffle
parser = argparse.ArgumentParser(description='This program builds cross-validation dataset from .codex/y files')
parser.add_argument('-i', '--inprefix', type=str, required=True, help="prefix of feature and response files (.codedx & .codedy)")
parser.add_argument('-n', '--fold', type=int, required=True, help="n fold cross validation")
parser.add_argument("-o", '--outdir', type=str, required=True,  help="output directory")
args = parser.parse_args()
feafile=args.inprefix+".codedx"
rspfile=args.inprefix+".codedy"
fea=pd.read_csv(feafile)
rsp=pd.read_csv(rspfile)
m=fea.shape[0]
idx=list(range(0,m))
fea,rsp = shuffle(fea,rsp)
if not os.path.exists(args.outdir):
  os.makedirs(args.outdir)
n=int(args.fold)
size=int(m/n)
if size*n<m:
  size=size+1
for i in range(0,n):
  n_start=int(i*size)
  n_end=int(i*size+size)
  if n_end>m:
    n_end=m
  trainx=pd.DataFrame()
  trainy=pd.DataFrame()
  testx=pd.DataFrame() 
  testx=pd.DataFrame()
  trainx=fea.iloc[np.r_[0:n_start,n_end:m]]
  trainy=rsp.iloc[np.r_[0:n_start,n_end:m]]
  testx=fea.iloc[n_start:n_end]
  testy=rsp.iloc[n_start:n_end]
  trainxout=os.path.join(args.outdir,"train.X."+str(i))
  trainyout=os.path.join(args.outdir,"train.Y."+str(i))
  testxout=os.path.join(args.outdir,"test.X."+str(i))
  testyout=os.path.join(args.outdir,"test.Y."+str(i))
  trainx.to_csv(trainxout,index=False)
  trainy.to_csv(trainyout,index=False)
  testx.to_csv(testxout,index=False)
  testy.to_csv(testyout,index=False)
