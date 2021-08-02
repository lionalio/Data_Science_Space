import os, sys

from sklearn.base import ClassifierMixin

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path + "/common")

from eda import *

eda_ = EDA('creditcard.csv', label='Class')

eda_.dump()