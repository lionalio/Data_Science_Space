import os, sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path + "/common")

from eda import *
from learning import *

df = pd.read_csv('insurance_claims.csv')
#print(df)
#eda_ = EDA('insurance_claims.csv', label='fraud_reported')
#eda_.dump()
mapping = {
    'fraud_reported': {'N': 0, 'Y': 1},
}

preprocessings = {
    #'imputer': SimpleImputer(strategy='most_frequent'),
    #'mapping': mapping,
    #'dim_reduce': False,
    #'remove_high_corr': False,
    'encoder': None,
    'cat_encoder': 'CountEncoder',
    'preprocess': None,
}

classifiers = [
    LogisticRegression(solver='lbfgs'),
    DecisionTreeClassifier(),
    xgb.XGBClassifier(),  # Take insanely long time to run. Why???
    RandomForestClassifier()
]

parameters = [
    # Logistic Regression
    {
        'penalty': ['l2'],
        'C': (0.01, 10., 'log-uniform'),
        'max_iter': (100, 1000)
    },
    # Decision Tree
    {
        'criterion':['gini','entropy'],
        'max_depth': (5, 50)
    },
    # XGBoost
    {
        "learning_rate": (0.01, 1., 'log-uniform'),
        "max_depth": (3, 10),
        "min_child_weight": (1, 10),
        "gamma": (0.0, 1, 'uniform'),
        'eval_metric': ['mlogloss']
    },
    # Random Forest
    {
        'bootstrap': [True, False],
         'max_depth': (10, 100),
         'max_features': ['auto', 'sqrt'],
         'min_samples_leaf': (1, 10),
         'min_samples_split': (2, 10),
         'n_estimators':(100, 200, 500, 1000)
    }
]

drops = ['policy_number','policy_bind_date', 'incident_date','incident_location','auto_model', 
        'fraud_reported', '_c39']

detector = Learning('insurance_claims.csv', 
                        [f for f in df.columns if f not in drops], 
                        label_col='fraud_reported')
print(detector.df)
detector.set_methods_process(preprocessings)
detector.set_mapping(mapping)
detector.processing()
for clf, params in zip(classifiers, parameters):
    detector.set_methods(clf)
    detector.set_parameters(params)
    detector.run(method='BayesSearch')