import pandas as pd
import numpy as np
from fancyimpute import KNN,SoftImpute
import joblib


test=pd.read_csv("../input/test_data.csv",low_memory=False)


cat_cols = ['Term','Years in current job','Home Ownership','Purpose']

for c in cat_cols:
    test[c] = pd.factorize(test[c])[0]

#Imputing missing data with soft impute
updated_test_data=pd.DataFrame(data=SoftImpute().fit_transform(test[test.columns[3:19]],), columns=test[test.columns[3:19]].columns, index=test.index)

#Getting the dataset ready pd.get dummies function for dropping the dummy variables
test_data = pd.get_dummies(updated_test_data, drop_first=True)


gbm_pickle = joblib.load('../output/GBM_Model_version1.pkl')


y_pred = gbm_pickle.predict(test_data)


y_pred = gbm_pickle.predict_proba(test_data)

y_pred_1=np.where(y_pred ==0, 'Loan Approved', 'Loan Rejected')


test['Loan Status']=y_pred_1

test.to_csv('../output/Output_Test.csv',index=False)