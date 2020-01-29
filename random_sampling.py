import os
import pandas as pd
import numpy as np
import argparse
import random
parser = argparse.ArgumentParser(description='This program takes in feature files (codedx & codedy) and generates a small testing set by random sampling')
parser.add_argument('-i', '--inprefix', type=str, required=True, help="prefix of input files")
parser.add_argument('-o', '--outprefix', type=str, required=True, help="prefix of output files")
parser.add_argument('-s', '--sampling', type=str, required=True, help="sampling approach: if integer and >1: number of rows; if float and <=1: fraction; default:1000")
args = parser.parse_args()

codedxfile=args.inprefix+".codedx"
if not os.path.isfile(codedxfile):
    print("Cannot find the .codedx file!")
    exit()
codedx=pd.read_csv(codedxfile)

codedyfile=args.inprefix+".codedy"
if not os.path.isfile(codedxfile):
    print("Cannot find the .codedy file!")
    exit()
codedy=pd.read_csv(codedyfile)

n=len(codedx.index)
print(n)
samn=1000
if n<samn:
    samn=n
if args.sampling:
    r=float(args.sampling)
    if r>1:
        samn=int(r)
    elif r>0:
        samn=int(n*r)
    else:
        print("samping option should be >0")
        exit()
idx=[i for i in range(0,n)]
selidx=random.sample(idx,samn)
selcodedx=codedx.iloc[selidx,:]
selcodedy=codedy.iloc[selidx,:]
selcodedxfile=args.outprefix+".codedx"
selcodedyfile=args.outprefix+".codedy"
selcodedx.to_csv(selcodedxfile,index=False)
selcodedy.to_csv(selcodedyfile,index=False)



