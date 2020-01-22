import pandas as pd
import numpy as np
import os
import argparse
parser = argparse.ArgumentParser(description='This program takes in the output files of read_data.py, removes outliers,  and code the data')
parser.add_argument('-i', '--inprefix', type=str, required=True, help="prefix for input files")
parser.add_argument('-o', '--outprefix', type=str, required=True, help="prefix for output files")
parser.add_argument('-r', '--rmoutlier', action='store_true', help="remove outliers")
args = parser.parse_args()

c_outlier_cut=0.03
iqr_magnitude=1.5

rmol=args.rmoutlier
vardesfile=args.inprefix+".vardes"
if not os.path.isfile(vardesfile):
    print("Cannot find the .vardes file!")
    exit()
rawxfile=args.inprefix+".rawx"
if not os.path.isfile(rawxfile):
    print("Cannot find the .rawx file!")
    exit()
rawyfile=args.inprefix+".rawy"
if not os.path.isfile(rawyfile):
    print("Cannot find the .rawy file!")
    exit()
vardes=pd.read_csv(vardesfile,sep='\t')
n=len(vardes.index)
vardict={}
for i in range(0,n):
    vardict[vardes.iloc[i,0]]=vardes.iloc[i,4]
rawx=pd.read_csv(rawxfile)
m=len(rawx.index)
outlierrow=[0 for i in range(m)]
dumvar=[]
numvar=[]
for colname in rawx.columns:
    if vardict[colname]=='c':
        keepvar=1
        if rmol:
            outliers=[]
            valuecounts=rawx[colname].value_counts()
            for i in range(len(valuecounts.index)):
                if valuecounts[i]/m<c_outlier_cut:
                    outliers.append(valuecounts.index[i])
            if len(valuecounts.index)-len(outliers)<2:
                keepvar=0
            else:
                if len(outliers)>0:
                    for j in range(m):
                        if rawx[colname][j] in outliers:
                            outlierrow[j]=1
        if keepvar==1:
            dumvar.append(colname)

    if vardict[colname]=='n':
        numvar.append(colname)
        if rmol:
            q4=np.max(rawx[colname])
            q3=np.quantile(rawx[colname],0.75)
            q1=np.quantile(rawx[colname],0.25)
            q0=np.min(rawx[colname])
            print(str(q4)+"\t"+str(q3)+"\t"+str(q1)+"\t"+str(q0))
print(dumvar)
print(numvar)

