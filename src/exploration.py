import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind
from numpy import median
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV, Ridge, Lasso
from scipy.stats import skew
import seaborn as sns

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')


train = train[train['GrLivArea']<4000]
#train = train.drop(train[train['TotalBsmtSF'] == 0].index)
#train = train[train['TotalBsmtSF']<3200]
#train = train[train['1stFlrSF']<4000]
#train = train.drop(train[train['GarageArea'] == 0].index)
# print train.shape
#train = train[train['GarageArea']<1240]
# print "garage", train.shape
# train = train[train['LotArea']<100000]
# print "LotArea", train.shape
# train = train[train['MasVnrArea']<1160]
# print "MasVnrArea", train.shape
# train = train[train['BsmtFinSF1']<5000]
# print "BsmtFinSF1", train.shape
# #train = train[train['LotFrontage']<300]
# #print "LotFrontage", train.shape
# train = train[train['OpenPorchSF']<500]
# print "OpenPorchSF", train.shape




correlations = train.corr()
correlPrice = correlations.sort_values(by='SalePrice', ascending=False)
for feat in correlPrice[0:10].index:
    print correlations.sort_values(by=feat, ascending=False)[feat]
    print "-----------"
k = 20 #number of variables for heatmap
cols = correlations.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


highestCor = correlations['SalePrice'].index[1:10]
print highestCor
sns.set()
sns.pairplot(train[highestCor], size=2)
plt.show()
  
# for feat in correlations['SalePrice'].index:
#     train.plot.scatter(feat, 'SalePrice')
#     plt.show()



allData = pd.concat([train,test])


print train.shape
print test.shape
print allData.shape
isNa = allData.isna().sum()

#impute assuming NA == None
allData["PoolQC"] = allData["PoolQC"].fillna("None")
allData["Alley"] = allData["Alley"].fillna("None")
allData["MiscFeature"] = allData["MiscFeature"].fillna("None")
allData["Fence"] = allData["Fence"].fillna("None")
allData["FireplaceQu"] = allData["FireplaceQu"].fillna("None")
allData["GarageCond"] = allData["GarageCond"].fillna("None")
allData["GarageFinish"] = allData["GarageFinish"].fillna("None")
allData["GarageQual"] = allData["GarageQual"].fillna("None")
allData["GarageYrBlt"] = allData["GarageYrBlt"].fillna("None")
allData["GarageType"] = allData["GarageType"].fillna("None")
allData["BsmtCond"] = allData["BsmtCond"].fillna("None")
allData["BsmtExposure"] = allData["BsmtExposure"].fillna("None")
allData["BsmtQual"] = allData["BsmtQual"].fillna("None")
allData["BsmtFinType2"] = allData["BsmtFinType2"].fillna("None")
allData["BsmtFinType1"] = allData["BsmtFinType1"].fillna("None")
allData["GarageType"] = allData["GarageType"].fillna("None")
allData["MasVnrType"] = allData["MasVnrType"].fillna("None")

#impute with most frequent value
allData["Utilities"] = allData["Utilities"].fillna(allData["Utilities"].value_counts().idxmax())
allData["MSZoning"] = allData["MSZoning"].fillna(allData["MSZoning"].value_counts().idxmax())
allData["Functional"] = allData["Functional"].fillna(allData["Functional"].value_counts().idxmax())
allData["Electrical"] = allData["Electrical"].fillna(allData["Electrical"].value_counts().idxmax())
allData["Exterior1st"] = allData["Exterior1st"].fillna(allData["Exterior1st"].value_counts().idxmax())
allData["Exterior2nd"] = allData["Exterior2nd"].fillna(allData["Exterior2nd"].value_counts().idxmax())
allData["KitchenQual"] = allData["KitchenQual"].fillna(allData["KitchenQual"].value_counts().idxmax())
allData["SaleType"] = allData["SaleType"].fillna(allData["SaleType"].value_counts().idxmax())

#impute with median
allData["LotFrontage"] = allData["LotFrontage"].fillna(allData["LotFrontage"].median())
allData["MasVnrArea"] = allData["MasVnrArea"].fillna(allData["MasVnrArea"].median())
allData["TotalBsmtSF"] = allData["TotalBsmtSF"].fillna(allData["TotalBsmtSF"].median())
allData["GarageCars"] = allData["GarageCars"].fillna(allData["GarageCars"].median())
allData["GarageArea"] = allData["GarageArea"].fillna(allData["GarageArea"].median())
allData["BsmtUnfSF"] = allData["BsmtUnfSF"].fillna(allData["BsmtUnfSF"].median())
allData["BsmtFinSF2"] = allData["BsmtFinSF2"].fillna(allData["BsmtFinSF2"].median())
allData["BsmtFinSF1"] = allData["BsmtFinSF1"].fillna(allData["BsmtFinSF1"].median())
allData["BsmtFullBath"] = allData["BsmtFullBath"].fillna(allData["BsmtFullBath"].median())
allData["BsmtHalfBath"] = allData["BsmtHalfBath"].fillna(allData["BsmtHalfBath"].median())

# allData["YearBuilt-2"] = allData["YearBuilt"] ** 2
# allData["LotArea-2"] = np.sqrt(allData["LotArea"])
# allData["GrLivArea-2"] = allData["GrLivArea"] ** 2
# allData["GrLivArea-3"] = allData["GrLivArea"] ** 3
# allData["1stFlrSF-2"] = allData["1stFlrSF"] ** 2
# allData["1stFlrSF-3"] = allData["1stFlrSF"] ** 3
# allData["TotalSF"] = allData["1stFlrSF"] + allData["2ndFlrSF"] + allData["TotalBsmtSF"]
 
isNa = allData.isna().sum()
print "missing values:", isNa[isNa>0].sort_values(ascending=False)

numeric_feats = allData.dtypes[allData.dtypes != "object"].index

skewed_feats = allData[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.25]

skewed_feats = skewed_feats.index
#print len(skewed_feats), len(numeric_feats)
allData[skewed_feats] = np.log1p(allData[skewed_feats])
numeric_feats = numeric_feats.drop('SalePrice')
print numeric_feats

stdSc = StandardScaler()
allData[numeric_feats] = stdSc.fit_transform(allData[numeric_feats] )




allData = pd.get_dummies(allData)
trainFilled = allData[0:train.shape[0]]
testFilled = allData[train.shape[0]:]
print trainFilled.shape
print testFilled.shape

#box plot to detect outliers
#trainFilled.boxplot(column="TotalBsmtSF")
#plt.show()

#check skewness of SalePrice
#train['SalePrice'].hist(bins=20)
#plt.show()


correlations = trainFilled.corr().sort_values(by='SalePrice', ascending=False)
print correlations['SalePrice']
   
for feat in correlations['SalePrice'].index:
    trainFilled.plot.scatter(feat, 'SalePrice')
    plt.show()