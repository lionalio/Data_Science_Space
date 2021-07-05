# Common tools
import pandas as pd
import numpy as np
import time
import itertools
import copy
import os, glob, pickle
import sys

# Data Preparation
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# For NLP vectorization
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Resampling
from imblearn.over_sampling import SMOTE

# Machine Learning / AI
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, \
    GradientBoostingRegressor, RandomForestRegressor
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, Lasso, LassoCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import Isomap

# Evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score,  roc_curve, auc, \
    silhouette_score

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns