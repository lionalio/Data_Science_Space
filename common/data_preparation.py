from os import path
from libs import *


def load_single_file(filename, delimiter):
    df = pd.read_csv(filename, delimiter=delimiter)
    return df


def load_path(root_path, delimiter):
    group_data =[]
    for root, d, files in os.walk(root_path):
        for f in files:
            filepath = "{}{}".format(root, f)
            group_data.append(pd.read_csv(filepath, delimiter=delimiter))

    return pd.concat(group_data)  # Watch out!


class DataPreparation():
    def __init__(self, filename, features, label_col, delimiter=',', 
                single_file=True, data_type='tabular', test_size=0.2):
        self.df = None
        if single_file:
            self.df = load_single_file(filename, delimiter=delimiter)
        self.data_type = data_type
        self.features = features
        self.label = label_col
        self.methods = None
        self.mapping = None
        self.test_size = test_size
        self.X, self.y = None, None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.X_train_engineer = None
        self.X_test_engineer = None
        self.features = features
        self.cat_features = [f for f in self.features if self.df[f].dtypes == 'object' and f != label_col]
        self.num_features = [f for f in self.features if self.df[f].dtypes != 'object' and f != label_col]
        self.guide()

    def guide(self):
        print('Data Preparation: consider doing the following steps:')
        print('0/ Any EDA beforehand is highly recommended!')
        print('1/ set_method_process(). It is recommended after EDA for any imputing/mapping/dim reduction/other processes')
        print('2/ processing() will run the entire data preparation process.')

    def set_methods_process(self, methods):
        '''
        format: {
            'imputer': imputer/None,
            'mapping': mapping,
            'pca': True/False,
            'preprocess': scaler,
        }
        '''
        self.methods = methods

    def set_mapping(self, mapping):
        self.mapping = mapping

    def missing_imputer(self, imputer):
        if imputer is not None:
            for f in self.cat_features:
                self.df[f] = imputer.fit_transform(self.df[f])
        
    def mapping_data(self):
        for k, v in self.mapping.items():
            if self.df[k].dtypes == 'object':
                self.df[k] = self.df[k].map(v)

    def encoder(self, enc):
        self.df = enc.fit_transform(self.df)

    def create_train_test(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, 
                                                                                test_size=self.test_size, random_state=7)

    def remove_high_correlation(self, thresh=0.8):
        print('Remove high correlation features with threshold {}'.format(thresh))
        corr = self.X.corr()
        up_matrix = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
        drop_cols = [col for col in up_matrix.columns if any(up_matrix[col] > thresh)]
        self.X = self.X[[col for col in self.features if col not in drop_cols]]

    def select_Lasso(self):
        scaler = StandardScaler()
        print(self.X_train)
        X_train_std = scaler.fit_transform(self.X_train)
        lasso = LassoCV()
        lasso.fit(X_train_std, self.y_train)
        X_test_std = lasso.transform(self.X_test)
        r2 = lasso.score(X_test_std, self.y_test)
        print('features selected from Lasso: ', self.X.columns[lasso.coef_ != 0])

        return lasso.coef_ != 0

    def select_GBR(self):
        n_feature_select = int(len(self.X.columns)*3/4)
        rfe = RFE(estimator = GradientBoostingRegressor(),
            n_feature_to_select = n_feature_select,
            step=2, verbose=0
        )
        rfe.fit(self.X_train, self.y_train)
        print('features selected from Gradient Boosting: ', self.X.columns[rfe.support_])

        return rfe.support_

    def select_RFC(self):
        n_feature_select = int(len(self.X.columns)*3/4)
        rfe = RFE(estimator = RandomForestClassifier(),
            n_feature_to_select = n_feature_select,
            step=2, verbose=0
        )
        rfe.fit(self.X_train, self.y_train)
        print('features selected from Random Forest: ', self.X.columns[rfe.support_])

        return rfe.support_

    def feature_selection(self):
        mask_lasso = self.select_Lasso()
        mask_gbr = self.select_GBR()
        mask_rfc = self.select_RFC()
        votes = np.sum([mask_lasso, mask_gbr, mask_rfc], axis=0)

        selected_cols = self.X_columns[votes >= 2]
        print('selected features: ', selected_cols)
        self.X_train = self.X_train[selected_cols]
        self.X_test = self.X_test[selected_cols]

    def dim_reduction(self, thresh=0.95):
        X_pca = StandardScaler().fit_transform(self.X)
        pca = PCA(n_components=X_pca.shape[1])
        pca.fit(X_pca)
        variances = np.cumsum(pca.explained_variance_ratio_)
        position = np.argmax(variances > thresh)
        print('dimemsion reduction: position to choose: ', position)
        pca_final = PCA(n_components=position)
        self.X = pca_final.fit_transform(self.X)

    def processing(self):
        if 'imputer' in self.methods and self.methods['imputer'] is not None:
            self.missing_imputer(self.methods['imputer'])
        if self.mapping is not None:
            self.mapping_data()
        if 'encoder' in self.methods and self.methods['encoder'] is not None:
            self.encoder(self, self.methods['encoder'])

        self.X = self.df[self.features] 
        if self.label is not None: 
            self.y = self.df[self.label]#.values
            if 'resampling' in self.methods and self.methods['resampling'] is not None:
                self.X, self.y = self.methods['resampling'].fit_resample(self.X, self.y)
        if 'remove_high_corr' in self.methods and self.methods['remove_high_corr'] is True:
            self.remove_high_correlation()
        if 'dim_reduce' in self.methods and self.methods['dim_reduce'] is True:
            self.dim_reduction()
        if self.label is not None:
            self.process_supervised()

    def process_supervised(self):
        self.create_train_test()
        if 'feature_selection' in self.methods and self.methods['feature_selection'] is True:
            self.feature_selection()
            
        if 'preprocess' in self.methods and self.methods['preprocess'] is not None:
            if self.data_type == 'tabular':
                if 'dim_reduce' in self.methods and self.methods['dim_reduce'] is False:
                    self.X_train_engineer = copy.copy(self.X_train)
                    self.X_test_engineer = copy.copy(self.X_test)
                    self.X_train_engineer[self.num_features] = self.methods['preprocess'].fit_transform(self.X_train[self.num_features])
                    self.X_test_engineer[self.num_features] = self.methods['preprocess'].transform(self.X_test[self.num_features])
                else:
                    # Great. features changed entirely. Let's convert them
                    self.X_train_engineer = self.methods['preprocess'].fit_transform(self.X_train)
                    self.X_test_engineer = self.methods['preprocess'].transform(self.X_test)
            elif self.data_type == 'text':
                # Let's hope in NLP, we only have ONE feature of document to take care of...
                self.X_train_engineer = self.methods['preprocess'].fit_transform(self.X_train[self.cat_features[0]])
                self.X_test_engineer = self.methods['preprocess'].transform(self.X_test[self.cat_features[0]])
        else:
            self.X_train_engineer = self.X_train
            self.X_test_engineer = self.X_test