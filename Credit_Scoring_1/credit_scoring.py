import os, sys

from sklearn.base import ClassifierMixin

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path + "/common")

from eda import *
from classification import *

import matplotlib
matplotlib.use('TkAgg')

df = pd.read_csv('dataset_31_credit-g.csv')
print(df)

#eda0 = EDA('dataset_31_credit-g.csv', label='class')
#eda0.dump()

features = df.columns
preprocessings = {
    'imputer': LabelEncoder(),
    'mapping': None,
    'dim_reduce': True,
    'remove_high_corr': True,
    'preprocess': None,
}

classifiers = [
    LogisticRegression(solver='lbfgs'),
    DecisionTreeClassifier(),
    xgb.XGBClassifier(),  # Take insanely long time to run. Why???
    RandomForestClassifier()
]


# Those are for grid search! For Bayesian search, it must use different declaration
'''
parameters = [
    {
        'penalty': ['l2'],
        'C': [0.1, 0.5, 1., 2.],
        'max_iter': [100, 200, 500, 1000]
    },
    {
        'criterion':['gini','entropy'],
        'max_depth':[5, 8, 10, 20, 50]
    },
    {
        "learning_rate": [0.05, 0.10, 0.15],
        "max_depth": [ 3, 5, 8],
        "min_child_weight": [ 1, 3, 5],
        "gamma":[ 0.0, 0.1, 0.5],
        'eval_metric': ['mlogloss']
    }
]
'''
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


detector = Classification('dataset_31_credit-g.csv', [f for f in df.columns if f != 'class'], 'class')
print(detector.cat_features)
print(detector.num_features)
detector.set_methods_process(preprocessings)
detector.processing()
for clf, params in zip(classifiers, parameters):
    detector.set_methods(clf)
    detector.set_parameters(params)
    detector.run(method='BayesSearch')