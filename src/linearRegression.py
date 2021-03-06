import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, RobustScaler
from scipy.stats import ttest_ind
from numpy import median
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV, Ridge, Lasso
from scipy.stats import skew
import seaborn as sns
from sklearn.metrics import r2_score
from scipy.special import boxcox1p

nFolds = 10
skewThres = 0.

def rmse(actual, predictions):
    #return np.sqrt(np.sum(np.log(actual) - np.log(predictions))**2/len(actual))
    return np.sqrt(np.sum(actual - predictions)**2/len(actual))

def rmseCV(model, scoringFunction, X, y):
    return np.sqrt(-cross_val_score(model,X=X,y=y,cv=5, scoring=scoringFunction))

def nestedCrossValidation(X, y, model, parameterGrid, scoringFunction):
    numIterations = 3
    scores = np.zeros(numIterations)
    for i in range(numIterations):
        print("Iteration %d" %i)
        innerCV = KFold(n_splits=nFolds, shuffle=True, random_state=i)
        outerCV = KFold(n_splits=nFolds, shuffle=True, random_state=i+1+numIterations)
        gridSearch = GridSearchCV(estimator=model, param_grid=parameterGrid, scoring=scoringFunction, cv=innerCV, verbose=0)
        nestedScore = np.sqrt(-cross_val_score(gridSearch, X=X, y=y, cv=outerCV, scoring=scoringFunction))
        print(nestedScore)
        print("score: %.5f" % nestedScore.mean())
        scores[i] = nestedScore.mean()
    return scores

def crossValidation(X, y, model, parameterGrid, scoringFunction):
    numIterations = 1
    scores = np.zeros(numIterations)
    for i in range(numIterations):
        print("Iteration %d" %i)
        gridSearch = GridSearchCV(estimator=model, param_grid=parameterGrid, scoring=scoringFunction, cv=nFolds, verbose=0)
        gridSearch.fit(X,y)
        nestedScore = np.sqrt(-gridSearch.best_score_)
        print("score: %.5f" % nestedScore.mean())
        print("best alpha:", gridSearch.best_params_)
        scores[i] = nestedScore.mean()
    return scores

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

#print isNa[isNa>0].sort_values(ascending=False)
#housesWithPools = train[~train['PoolQC'].isna()]
#housesWithoutPools = train[train['PoolQC'].isna()]
#print " Avg price of houses with pools:", housesWithPools['SalePrice'].mean(), "without: ", housesWithoutPools['SalePrice'].mean()
#print ttest_ind(housesWithPools['SalePrice'], housesWithoutPools['SalePrice'])




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

#train = train.drop(columns=['GarageCars', 'TotRmsAbvGrd', 'GarageCars', '1stFlrSF', 'GarageYrBlt'])
#test = test.drop(columns=['GarageCars', 'TotRmsAbvGrd', 'GarageCars', '1stFlrSF', 'GarageYrBlt'])

correlations = train.corr().sort_values(by='SalePrice', ascending=False)
print correlations['SalePrice']

# for feat in correlations['SalePrice'].index:
#     train.plot.scatter(feat, 'SalePrice')
#     plt.show()

#allData = allData.drop(allData[allData['MasVnrArea'] == 0].index)
#allData = allData.drop(allData[allData['BsmtFinSF1'] == 0].index)
#allData = allData.drop(allData[allData['BsmtFinSF2'] == 0].index)
# allData = allData.drop(allData[allData['OpenPorchSF'] == 0].index)
# allData = allData.drop(allData[allData['WoodDeckSF'] == 0].index)
# allData = allData.drop(allData[allData['2ndFlrSF'] == 0].index)



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
# #allData["1stFlrSF-2"] = allData["1stFlrSF"] ** 2
# #allData["1stFlrSF-3"] = allData["1stFlrSF"] ** 3
# allData["TotalSF"] = allData["2ndFlrSF"] + allData["TotalBsmtSF"]
 
isNa = allData.isna().sum()
print "missing values:", isNa[isNa>0].sort_values(ascending=False)

numeric_feats = allData.dtypes[allData.dtypes != "object"].index
categoricFeats = allData.dtypes[allData.dtypes == "object"].index

skewed_feats = allData[numeric_feats].apply(lambda x: abs(skew(x.dropna()))) #compute skewness
skewed_feats = skewed_feats[skewed_feats > skewThres]
skewed_feats = skewed_feats.drop(['SalePrice'])
skewed_feats = skewed_feats.index

numeric_feats = numeric_feats.drop(['SalePrice', 'Id'])
print numeric_feats

allData["SalePrice"] = np.log1p(allData["SalePrice"])
#allData[skewed_feats] = np.log1p(allData[skewed_feats])
allData[skewed_feats] =boxcox1p(allData[skewed_feats], 0.25)
#for feat in skewed_feats:
#    allData[feat] = boxcox1p(allData[feat], 0.01)
    



stdSc = StandardScaler()
allData[numeric_feats] = stdSc.fit_transform(allData[numeric_feats] )

allData = pd.get_dummies(allData)
trainFilled = allData[0:train.shape[0]]
testFilled = allData[train.shape[0]:]




#scatter plots of variables strongly correlated to SalePrice
#trainFilled.plot.scatter(x='OverallQual', y='SalePrice')
#trainFilled.plot.scatter(x='GrLivArea', y='SalePrice')
#trainFilled.plot.scatter(x='GarageCars', y='SalePrice')
#trainFilled.plot.scatter(x='TotalBsmtSF', y='SalePrice')
#trainFilled.plot.scatter(x='MasVnrArea', y='SalePrice')


numRuns = 1
errors = [0]*numRuns
ridgeCV = RidgeCV(alphas=[21,22,23,24,25,26,27,28,29,30],scoring="neg_mean_squared_error", 
                  normalize=False, cv=nFolds)
ridge = Ridge(normalize=False)
alphasRidge = {"alpha":[13,14,15,16,17,20, 21, 22, 23, 24, 25]}

alphasLasso = {"alpha":[0.0005,0.001, 0.005]}
lasso = Lasso(max_iter=1000)
lassoCV = LassoCV(alphas=[0.0005,0.001, 0.005], normalize=False, cv=nFolds, max_iter=1000)
#lassoCV.fit(trainFilled.drop(columns='SalePrice'), trainFilled['SalePrice'])


testModel = ridge
testParams = alphasRidge
predModel = ridgeCV

Xtrain = trainFilled.drop(columns=['SalePrice', 'Id'])
ytrain = trainFilled['SalePrice']

testFilled = testFilled.drop(columns=['Id'])
print trainFilled.shape
print testFilled.shape

# error = nestedCrossValidation(Xtrain, ytrain, 
#                       testModel, testParams, "neg_mean_squared_error")
# print("mean error from nested CV: {0}, variance: {1}".format(np.mean(error), np.var(error)) )
  
errorNonNested = crossValidation(Xtrain, ytrain, 
                      testModel, testParams, "neg_mean_squared_error")
  
print("mean error from non-nested CV: %.5f " %np.mean(errorNonNested) )
#print("mean rmseCV: %.5f" % rmseCV(ridge, "neg_mean_squared_error", Xtrain, ytrain).mean())

predModel.fit(Xtrain, ytrain)



ytrain_pred = predModel.predict(Xtrain)
print "r2 score", r2_score(ytrain, ytrain_pred)
print sorted(ytrain_pred - ytrain)[0:20]

# plt.scatter(ytrain_pred, ytrain_pred - ytrain, c = "lightgreen", marker = "s", label = "Validation data")
# plt.title("Linear regression")
# plt.xlabel("Predicted values")
# plt.ylabel("Residuals")
# plt.legend(loc = "upper left")
# plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")
# plt.show()
# 
# plt.scatter(ytrain_pred, ytrain, c = "lightgreen", marker = "s", label = "Validation data")
# plt.title("Linear regression with Ridge regularization")
# plt.xlabel("Predicted values")
# plt.ylabel("Real values")
# plt.legend(loc = "upper left")
# plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")
# plt.show()

print "best alpha", predModel.alpha_
predictionsSub = predModel.predict(testFilled.drop(columns=['SalePrice']))
predictionsSub = np.expm1(predictionsSub)




 
#solution = pd.DataFrame({"id":test.Id, "SalePrice":predictionsSub})
#solution.to_csv("../data/submission.csv", index = False)

values = zip(test['Id'], predictionsSub) 
with open('../data/submissionLR.csv', 'w') as file:
    file.write('Id,SalePrice\n')
    for id,value in values:
        file.write(str(id)+","+str(value)+"\n")
