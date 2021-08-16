from libs import *
from data_preparation import *


# For regression, value column is the same meaning as 'label' column in classification
class Learning(DataPreparation):
    def __init__(self, filename, features, label_col, delimiter=',', 
                single_file=True, data_cat='tabular', test_size=0.2):
        super().__init__(filename, features, label_col, delimiter, single_file, data_cat, test_size)
        self.method_learner = None
        self.params_learner = None
        self.method_set = False
        self.parameter_set = False

    @classmethod
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
        def wrapper(self, *args, **kwargs):
            print('Running ', function.__name__)
            start = time.time()
            function(self, *args, **kwargs)
            stop = time.time()
            print('Time elapsed: ', stop - start)
            
        return wrapper

    def set_methods(self, rgr):
        self.method_learner = rgr
        self.method_set = True
        
    def set_parameters(self, params):
        self.params_learner = params
        self.parameter_set = True

    @timing
    def learner(self, method='GridSearch'):
        opt = None
        if self.parameter_set is False:
            raise Exception('Error: All parameters are not yet set!')
        if method == 'GridSearch':
            opt = GridSearchCV(
                estimator=self.method_learner, 
                param_grid=self.params_learner
            )
        elif method == 'BayesSearch':     
            opt = BayesSearchCV(
                estimator=self.method_learner,
                search_spaces=self.params_learner,
                n_iter=50,
                random_state=7
            )
            print('to Bayes')
        opt.fit(self.X_train_engineer, self.y_train)   
        self.method_learner = opt.best_estimator_
        print('Best parameters for learner {}'.format(self.method_learner.__class__.__name__))
        print(opt.best_params_)

    def evaluate_regression(self):
        preds = self.method_learner.predict(self.X_test_engineer)
        print('Mean squared error: ', mean_squared_error(self.y_test, preds))
        #plt.figure(figsize=(15, 8))
        #plt.plot(self.y_test, color='b', label='True')
        #plt.plot(preds, color='r', label='Prediction')
        #plt.show()

    def evaluate_classification(self):
        preds = self.method_learner.predict(self.X_test_engineer)
        if self.method_learner.__class__.__name__ == 'XGBClassifier':
            xgb.plot_importance(self.method_learner)
            plt.show()
        elif self.method_learner.__class__.__name__ == 'RandomForestClassifier':
            importances = self.method_learner.feature_importances_
            indices = np.argsort(importances)

            plt.figure(figsize=(15, 8))
            plt.barh(range(len(indices)), importances[indices])
            plt.show()
        print('accuracy_score: ', accuracy_score(self.y_test, preds))
        print('confusion matrix for ', self.method_learner.__class__.__name__ , ": ", confusion_matrix(self.y_test, preds))
        if len(np.unique(self.y)) == 2:
            if type(self.y[0]) != str:
                self.plot_roc()

    def plot_roc(self):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        if "predict_proba" not in dir(self.method_learner):
            print("This function doesnt have probability calculation")
            return
        probs = self.method_learner.predict_proba(self.X_test_engineer)
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
    def run(self, method='GridSearch'):
        if self.method_set is False:
            raise Exception('Methods are not yet set. Aborting!')
        if self.parameter_set is False:
            print('Warning: All parameters are taking default values. Consider tuning!')
        #if self.load_data is False:
        #    super().processing()
        self.learner(method=method)
        if self.type_learner == 'Classification':
            self.evaluate_classification()
        else:
            self.evaluate_regression()