import os, sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path + "/common")

from eda import *
from classification import *

df = pd.read_csv('train.csv')

eda_ = EDA('train.csv', label='Loan_Status')
eda_.dump()

mapping = {

}

preprocessings = {
    'imputer': SimpleImputer(strategy='most_frequent'),
    'mapping': None,
    'dim_reduce': False,
    'remove_high_corr': False,
    'encoder': LabelEncoder(),
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

detector = Classification('train.csv', 
                        [f for f in df.columns if f != 'Loan_Status' and f != 'Loan_ID'], 
                        label_col='Loan_Status')
print(detector.df)
detector.set_methods_process(preprocessings)
detector.processing()
for clf, params in zip(classifiers, parameters):
    detector.set_methods(clf)
    detector.set_parameters(params)
    detector.run(method='BayesSearch')