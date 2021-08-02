from maths import JS_divergence, bhattacharyya_distance, hellinger_distance
from libs import *

class EDA():
    def __init__(self, filename, label=None, delimiter=','):
        if filename.split('.')[-1] == 'csv':
            self.df = pd.read_csv(filename, delimiter=delimiter)
        else:
            self.df = pd.read_excel(filename)
        self.label = label
        self.cat_features = [f for f in self.df.columns if self.df[f].dtypes == 'object' and f != self.label]
        self.num_features = [f for f in self.df.columns if self.df[f].dtypes != 'object' and f != self.label]
        self.sum_na = self.df.isna().sum()
        self.ratio_na = self.df.isna().sum() / len(self.df)

    def suggest_drop_features(self, thresh=0.3):
        drops = []
        for f in self.ratio_na.index:
            if self.ratio_na[f] > thresh:
                drops.append(f)

        return drops

    def features_to_be_imputed(self, thresh=0.3):
        for f in self.ratio_na.index:
            if self.ratio_na[f] < thresh and self.ratio_na[f] > 0.:
                if f in self.cat_features:
                    sns.countplot(data=self.df, x=f)
                else:
                    sns.histplot(data=self.df, x=f)
                plt.show()

    def all_pairplots(self):
        if len(self.num_features) == 0:
            print('No point to draw pairplots without a feature with real values')
            return
        sns.pairplot(data=self.df, vars=self.num_features, hue=self.label)
        plt.show()

    def label_distribution(self):
        if self.label is not None:
            sns.countplot(data=self.df, x=self.label)
            plt.show()

    def plot_compare_labels(self):
        if self.label is None:
            print('This kind of comparison is not necessary for data without labels')
            return
        for f in self.num_features:
            plt.figure(figsize=(10, 5))
            for val in self.df[self.label].unique():
                sns.distplot(self.df[f][self.df[self.label] == val], bins=50, label=str(val))
            plt.legend()
            plt.title('feature {}'.format(f))
            plt.show()

    # Make sure to put some more stuffs for EDA:
    # 1/ Divergence measurements between classes 
    # (for now, take between two classes only, skip if data is multiclasses)
    # 2/ Type of divergences: Kullback-Liebler, Hellinger, Kolmogorov-Smirnov test, Jensen-Shannon ...
    # 3/ Might take normality tests for numerical features (Shapiro-Wilks, Q-Q plot, kurtosis, ..., )
    # 4/ Might suggest some log transform for some highly skewed distributions
    def divergence_measurements(self):
        if self.label is None:
            print('Currently the divergence measurement is not considered if data has no label or class')
            return
        if len(self.df[self.label].unique()) > 2:
            print('This implementation of divergence measurement currently does not support multiclasses data')
            return
        for f in self.num_features:
            hist =[]
            xmin, xmax = 9999, -9999
            for val in self.df[self.label].unique():
                xmin = min(xmin, min(self.df[f][self.df[self.label] == val]))
                xmax = max(xmax, max(self.df[f][self.df[self.label] == val]))
            plt.figure(figsize=(10, 5))
            bins = None
            for val in self.df[self.label].unique():
                h, bins = np.histogram(self.df[f][self.df[self.label] == val],
                                        range=[xmin, xmax], 
                                        bins=50, density=True)
                plt.hist(self.df[f][self.df[self.label] == val], bins=bins, range=[xmin, xmax], density=True, alpha=0.3)
                hist.append(h)
            deltas = np.diff(bins)[0]
            dist1, dist2 = hist[0]*deltas, hist[1]*deltas
            print('KS test: ', ks_2samp(dist1, dist2))
            print('Hellinger distance: ', hellinger_distance(dist1, dist2))
            print('Bhattacharyya distance: ', bhattacharyya_distance(dist1, dist2))
            print('Jensen-Shannon divergence: ', JS_divergence(dist1, dist2))
            plt.show()

    def plot_correlation(self):
        if len(self.num_features) == 0:
            print('No point to draw pairplots without a feature with real values')
            return
        sns.heatmap(self.df[self.num_features].corr())
        plt.show()

    def plot_count(self):
        for f in self.cat_features:
            sns.countplot(data=self.df, x=f)
            plt.show()

    def plot_box(self):
        if len(self.num_features) == 0:
            print('No point to draw a box plot without a feature with real values')
            return
        for fc in self.cat_features:
            for fn in self.num_features:
                sns.boxplot(x=fc, y=fn, data=self.df)
                plt.show()

    def dump(self):
        print('Numerical features: ', self.num_features)
        print('Categorical features: ', self.cat_features)
        print('Counting missing values: ', self.sum_na)
        print('Missing value ratio: ', self.ratio_na)
        print('Consider drop following features: ')
        print(self.suggest_drop_features())
        if len(self.num_features) > 0:
            print('Description of numerical features: ', self.df[self.num_features].describe())
        if len(self.cat_features) > 0:
            print('Description of numerical features: ', self.df[self.cat_features].value_counts())
        self.divergence_measurements()
        self.label_distribution()
        self.features_to_be_imputed()
        #self.plot_compare_labels()
        #self.all_pairplots()from sklearn.base import ClassifierMixin

        

    