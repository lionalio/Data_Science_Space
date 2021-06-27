from libs import *

class Classification():
    def __init__(self, filename, features, label_col, 
                mapping=None, test_size=0.2):
        self.df = pd.read_csv(filename)
        self.mapping = mapping
        self.data_preparation()
        self.X, self.y = self.df[features], self.df[label_col]#.values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size)
        self.X_train_engineer = None
        self.X_test_engineer = None
        self.method_preprocessing = None
        self.method_classifier = None
        self.params_classifier = None
        self.method_set = False
        self.parameter_set = False
    
    def timing(function):
        def wrapper(self):
            print('Running ', function.__name__)
            start = time.time()
            function(self)
            stop = time.time()
            print('Time elapsed: ', stop - start)
            
        return wrapper

    def mapping_data(self):
        #print('You must convert all categorical data into numbers')
        for k, v in self.mapping.items():
            self.df[k] = self.df[k].map(v)
    
    def set_methods(self, preprocess, clf):
        self.method_preprocessing = preprocess
        self.method_classifier = clf
        self.method_set = True
        
    def set_parameters(self, params):
        self.params_classifier = params
        self.parameter_set = True
    
    @timing
    def data_preparation(self):
        #print('----- Running data preprocessing. Currently nothing to do...')
        if self.mapping is not None:
            self.mapping_data()
    
    @timing
    def preprocessing(self):
        self.X_train_engineer = self.method_preprocessing.fit_transform(self.X_train)
        self.X_test_engineer = self.method_preprocessing.transform(self.X_test)
        
    @timing
    def classifier(self):
        if self.parameter_set is False:
            raise Exception('Error: All parameters are not yet set!')
        grid = GridSearchCV(estimator=self.method_classifier, 
                            param_grid=self.params_classifier)
        grid.fit(self.X_train_engineer, self.y_train)        
        self.method_classifier = grid.best_estimator_
        
    @timing
    def evaluate(self):
        preds = self.method_classifier.predict(self.X_test_engineer)
        if self.method_classifier.__class__.__name__ == 'XGBClassifier':
            xgb.plot_importance(self.method_classifier)
            plt.show()
        elif self.method_classifier.__class__.__name__ == 'RandomForestClassifier':
            importances = self.method_classifier.feature_importances_
            indices = np.argsort(importances)

            plt.figure(figsize=(15, 8))
            plt.barh(range(len(indices)), importances[indices])
            plt.show()
        print('accuracy_score: ', accuracy_score(self.y_test, preds))
        print('confusion matrix for ', self.method_classifier.__class__.__name__ , ": ", confusion_matrix(self.y_test, preds))
        if len(np.unique(self.y)) == 2:
            if type(self.y[0]) != str:
                self.plot_roc()

    def plot_roc(self):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        if "predict_proba" not in dir(self.method_classifier):
            print("This function doesnt have probability calculation")
            return
        probs = self.method_classifier.predict_proba(self.X_test_engineer)
        fpr, tpr, _ = roc_curve(self.y_test, probs[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

    @timing
    def run(self):
        if self.method_set is False:
            raise Exception('Methods are not yet set. Aborting!')
        if self.parameter_set is False:
            print('Warning: All parameters are taking default values. Consider tuning!')
        self.data_preparation()
        self.preprocessing()
        self.classifier()
        self.evaluate()