import pandas as pd
from sklearn import ensemble
from sklearn import cross_validation
from sklearn import metrics
import xgboost as xgb
import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

df = train
liste = df.columns.values.tolist()
for s in liste :
    with open('describe.csv', 'a') as f:
        df[s].describe().to_csv(f, header=True)

# plt.figure()
# sns.distplot(np.log(df["SalePrice"]))
# plt.show()
#
# dico = {None: "0","Po": 1,"Fa": 2,"TA": 3,"Gd": 4,"Ex": 5}
# df["PoolQC"] = df["PoolQC"].map(dico)
# print df["PoolQC"].describe()

fig, ax = plt.subplots(figsize=(20,20),sharey=True)
sns.despine(left=True)
sns.color_palette("Set2", 10)
sns.set(color_codes=True)
plt.xlim(0,10000000)
plt.ylim(0,500)
sns.set(style="white", palette="muted")

num_cols = [c for c in df.columns if df.dtypes[c] in ['int64', 'float64']]
label_cols = df.columns.difference(num_cols)

# df = df.dropna()
# df = df.to_string()
# for i in num_cols :
#
#     f = sns.distplot(df[str(i)], kde=False)
# ax.set(xlabel='Valeurs', ylabel='Occurences')
# f.legend()
# fig.show()
