import os, sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path + "/common")

from eda import *
from learning import *

df = pd.read_csv('water_potability.csv')
print(df)

#eda_ = EDA('water_potability.csv', label='Potability')
#eda_.dump()

preprocessings = {
    #'drop_na': True,
    'imputer': SimpleImputer(strategy='median'), #KNNImputer(n_neighbors=8, weights="uniform"),
    #'mapping': mapping,
    #'dim_reduce': False,
    #'remove_high_corr': False,
    'encoder': None,
    'cat_encoder': None,
    'resampling': SMOTE(random_state=42),
    'preprocess': StandardScaler(),
}

classifiers = [
    LogisticRegression(solver='lbfgs'),
    DecisionTreeClassifier(),
    xgb.XGBClassifier(),  # Take insanely long time to run. Why???
    LGBMClassifier(),
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
        "learning_rate": (0.01, 1., 'uniform'),
        "max_depth": (3, 10),
        "min_child_weight": (1, 10),
        "gamma": (0.0, 1, 'uniform'),
        'eval_metric': ['mlogloss']
    },
    # LightGBM
    {
        'reg_alpha': (0.001, 10.0, 'uniform'),
        'reg_lambda': (0.001, 10.0, 'uniform'),
        'num_leaves': (10, 500),
        'min_child_samples': (5, 100),
        'max_depth': (2, 6),
        'learning_rate': (0.01, 0.25, 'log-uniform'),
        'colsample_bytree': (0.1, 0.6, 'uniform'),
        'cat_smooth' : (10, 100),
        'cat_l2': (1, 20),
        'min_data_per_group': (1, 200),
        'n_estimators': (300, 1000),
        'metric': ['binary_logloss']
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

label = 'Potability'
detector = Learning('water_potability.csv', [f for f in df.columns if f != label], label_col=label)
detector.type_learner = 'Classification'
detector.set_methods_process(preprocessings)
#detector.set_mapping(mapping)
detector.processing()

for clf, params in zip(classifiers, parameters):
    detector.set_methods(clf)
    detector.set_parameters(params)
    detector.run(method='BayesSearch')