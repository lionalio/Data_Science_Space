from libs import *
from data_preparation import *

class Classification(DataPreparation):
    def __init__(self, filename, features, label_col, delimiter=',', 
                single_file=True, data_cat='tabular', test_size=0.2):
        super().__init__(filename, features, label_col, delimiter, single_file, data_cat, test_size)
        self.load_data = False
        self.method_classifier = None
        self.params_classifier = None
        self.method_set = False
        self.parameter_set = False

    def load_processed_data(self, processed_data):
        self.load_data = True
        self.data_type = processed_data.data_type
        self.features = processed_data.features
        self.label = processed_data.label_col
        self.methods = processed_data.methods
        self.mapping = processed_data.mapping
        self.test_size = processed_data.test_size
        self.X, self.y = processed_data.X, processed_data.y
        self.X_train, self.X_test, self.y_train, self.y_test = processed_data.X_train, processed_data.X_test, \
            processed_data.y_train, processed_data.y_test
        self.X_train_engineer = None
        self.X_test_engineer = None
        self.features = processed_data.features
        self.cat_features = processed_data.cat_features
        self.num_features = processed_data.num_features
    
    def timing(function):
        def wrapper(self):
            print('Running ', function.__name__)
            start = time.time()
            function(self)
            stop = time.time()
            print('Time elapsed: ', stop - start)
            
        return wrapper

    def set_methods(self, clf):
        self.method_classifier = clf
        self.method_set = True
        
    def set_parameters(self, params):
        self.params_classifier = params
        self.parameter_set = True
        
    @timing
    def classifier(self):
        if self.parameter_set is False:
            raise Exception('Error: All parameters are not yet set!')
        grid = GridSearchCV(estimator=self.method_classifier, 
                            param_grid=self.params_classifier)
        grid.fit(self.X_train_engineer, self.y_train)        
        self.method_classifier = grid.best_estimator_
        print(grid.best_params_)
        
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
        if self.load_data is False:
            super().processing()
        self.classifier()
        self.evaluate()