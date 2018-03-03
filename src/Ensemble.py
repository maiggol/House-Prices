import pandas as pd
from scipy.stats import pearsonr

predictionsLR = pd.read_csv('../data/submissionRidge.csv')
predictionsSVR = pd.read_csv('../data/submissionSVR.csv')
print predictionsLR["Id"].values
predictionsEnsemble = (predictionsLR['SalePrice'].values + predictionsSVR['SalePrice'].values)/2
solution = pd.DataFrame({"Id":predictionsLR["Id"].values, "SalePrice":predictionsEnsemble})
solution.set_index("Id")
solution.to_csv("../data/submissionEnsemble.csv", index = False)

print pearsonr(predictionsLR['SalePrice'].values, predictionsSVR['SalePrice'].values)
