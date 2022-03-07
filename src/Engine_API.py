from flask import Flask, request, Response
import pandas as pd
import numpy as np
from fancyimpute import SoftImpute
import joblib


gbm_pickle = joblib.load('../output/GBM_Model_version1.pkl')

app = Flask(__name__)

@app.route('/loan_eligibility/predict', methods=['POST'])
def predict():
    test_json = request.get_json()

    if test_json:
        if isinstance(test_json, dict):
            test = pd.DataFrame(test_json, index=[0])
        else:
            test = pd.DataFrame(test_json, columns=test_json[0].keys())

        cat_cols = ['Term', 'Years in current job', 'Home Ownership', 'Purpose']

        for c in cat_cols:
            test[c] = pd.factorize(test[c])[0]

        # Imputing missing data with soft impute
        updated_test_data = pd.DataFrame(data=SoftImpute().fit_transform(test[test.columns[3:19]], ),
                                         columns=test[test.columns[3:19]].columns, index=test.index)

        # Getting the dataset ready pd.get dummies function for dropping the dummy variables
        test_data = pd.get_dummies(updated_test_data, drop_first=True)

        # y_pred = gbm_pickle.predict(test_data)

        y_pred = gbm_pickle.predict_proba(test_data)

        y_pred_1 = np.where(y_pred == 0, 'Loan Approved', 'Loan Rejected')

        test['Loan Status'] = y_pred_1
        test = test.to_dict(orient='records')

        return test
    else:
        return Response('{}', status=200, mimetype='application/json')

if __name__ == '__main__':
    app.run('0.0.0.0')