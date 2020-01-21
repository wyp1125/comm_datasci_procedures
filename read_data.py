import pandas as pd
import argparse
parser = argparse.ArgumentParser(description='This program reads a raw dataset and generates variable features.')
parser.add_argument('-i', '--infile', type=str, required=True, help="input data file in csv or xls format")
parser.add_argument('-o', '--outprefix', type=str, required=True, help="prefix for output files")
parser.add_argument('-y', '--yname', type=str, required=True, help="name of the response/outcome variable")
args = parser.parse_args()
filepath=args.infile
if filepath.endswith('.csv'):
    df=pd.read_csv(filepath)
elif filepath.endswith('.xlsx'):
    df=pd.read_excel(filepath)
else:
    print("The input file must be a '.csv' or '.xlsx' file")
    exit()
#filtering null colunms and rows with missing data
df1=df.dropna(axis=1,how='all')
df2=df1.dropna(axis=0,how='any')
nrows=df2.shape[0]
nvars=df2.shape[1]
colnames=df2.columns
varfea=df2.dtypes
havingy=0
print("Variable_name\tMode\tLevels\tPercentage\tType")
vardes="Variable_name\tMode\tLevels\tPercentage\tType"
feacolumns=[]
idcolumns=[]
ycolumns=[]
for i in range(0,len(varfea)):
    collevels=len(df2[varfea.index[i]].unique())
    perct=int(10000*collevels/nrows)/100
    if str(varfea[i])=="object":
        vartype="c"
        if collevels==1 or collevels>nrows*0.2:
            vartype="d"
    elif str(varfea[i])=="bool":
        vartype="c"
    else:
        vartype="n"
    if varfea.index[i]==args.yname:
        havingy=1
        vartype="y"+vartype
        modeltype=vartype
    if vartype=="c" or vartype=="n":
        feacolumns.append(varfea.index[i])
    if vartype=="d":
        idcolumns.append(varfea.index[i])
    if vartype[0]=="y":
        ycolumns.append(varfea.index[i])
    print(varfea.index[i]+"\t"+str(varfea[i])+"\t"+str(collevels)+"\t"+str(perct)+"\t"+vartype)
    vardes=vardes+"\n"+varfea.index[i]+"\t"+str(varfea[i])+"\t"+str(collevels)+"\t"+str(perct)+"\t"+vartype
if havingy==0:
    print("The name of the outcome variable specified did not match the data file. Please try again!")
    exit()
if modeltype=="yn":
    print("The outcome variable is numeric")
else:
    print("The outcome variable is categorical")
    level2count=df2[args.yname].value_counts()
    ncls=len(level2count)
    if ncls>2:
        print("This is a multi-class problem")
    else:
        print("This is a binary classification")
        cls1=int(level2count[0])
        cls2=int(level2count[1])
        print(str(level2count.index[0])+":"+str(level2count[0]))
        print(str(level2count.index[1])+":"+str(level2count[1]))
        rt=cls1/(cls1+cls2)
        if rt>0.1 and rt<0.9:
            print("This dataset should be regarded as balanced. Consider removing outliers in predictors")
        else:
            print("This dataset should be regarded as imbalanced. Better to implement anomaly detection algorithms")
with open(args.outprefix+".vardes","w") as of:
    of.write(vardes)
out_y=df2[ycolumns]
out_x=df2[feacolumns]
out_y.to_csv(args.outprefix+".rawy",index=False)
out_x.to_csv(args.outprefix+".rawx",index=False)
    


