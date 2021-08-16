import os, sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path + "/common")

from eda import *
from learning import *

df = pd.read_csv('housing.csv')

#df = pd.read_csv('housing.csv', header=None, delimiter=r"\s+", names=column_names)
#df.to_csv('housing.csv', index_label=False)


preprocessings = {
    #'imputer': SimpleImputer(strategy='most_frequent'),
    #'mapping': mapping,
    #'dim_reduce': False,
    #'remove_high_corr': False,
    'encoder': None,
    'cat_encoder': None,
    'preprocess': MinMaxScaler(),
    'feature_selection': True
}

classifiers = [
    LinearRegression(),
    #xgb.XGBRegressor(),  # Take insanely long time to run. Why???
    RandomForestRegressor()
]

parameters = [
    # Linear Regression
    {
        'normalize': [False, True]
    },
    # XGB Regressor
    #{
    #    'n_estimators': (100, 200, 500, 1000), 
    #    'max_depth': (10, 100), 
    #    'eta': (0.01, 10., 'log-uniform'), 
    #    'subsample': (0.1, 0.9, 'log-uniform'),
    #},
    # Random Forest Regressor
    {
        'bootstrap': [True, False],
        'criterion': ['mse', 'mae'],
        'max_depth': (10, 100),
        'max_features': ['auto', 'sqrt', 'log2'],
        'min_samples_leaf': (1, 10),
        'min_samples_split': (2, 10),
        'n_estimators':(100, 200, 500, 1000)
    }
]
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 
                'RM', 'AGE', 'DIS', 'RAD', 'TAX',
                'PTRATIO', 'B', 'LSTAT', 'MEDV']

column_sels = ['LSTAT', 'INDUS', 'NOX', 'PTRATIO', 'RM', 'TAX', 'DIS', 'AGE']

detector = Learning('housing.csv', 
                    [f for f in df.columns if f != 'MEDV'], 
                    label_col='MEDV')
print(detector.df.isna().sum())
detector.set_methods_process(preprocessings)
#detector.set_mapping(mapping)
detector.processing()
for clf, params in zip(classifiers, parameters):
    detector.set_methods(clf)
    detector.set_parameters(params)
    detector.run(method='BayesSearch')
