# Common tools
import pandas as pd
import numpy as np
import time
import itertools

# Data Preparation
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# For NLP vectorization
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Machine Learning / AI
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.tree import DecisionTreeClassifier

# Evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score,  roc_curve, auc

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns