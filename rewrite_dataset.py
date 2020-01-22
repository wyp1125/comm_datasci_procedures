import pandas as pd
import numpy as np
sevdata=pd.read_csv("../dataset/Auto_Insurance_Claims_Severity.csv")
values=sevdata["Total Claim Amount"]
hcut=np.quantile(values,2/3)
lcut=np.quantile(values,1/3)
n=len(values)
sevlevel=[]
for i in range(0,n):
    level='L'
    if values[i]>lcut:
        level='M'
    if values[i]>hcut:
        level='H'
    sevlevel.append(level)
sevdata1=sevdata.drop(['Total Claim Amount'],axis=1)
sevdata1['Claim Serverity']=sevlevel
sevdata1.to_csv("../dataset/Auto_Insurance_Claims_Severity1.csv",index=False)
creditfraud=pd.read_csv("../dataset/Credit_Card_Fraud.csv")
values=creditfraud["Class"]
n=len(values)
clscode=[]
for i in range(0,n):
    code='N'
    if values[i]==1:
        code='Y'
    clscode.append(code)
creditfraud["Class"]=clscode
creditfraud.to_csv("../dataset/Credit_Card_Fraud1.csv",index=False)
