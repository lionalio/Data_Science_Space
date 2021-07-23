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
        self.label_distribution()
        self.features_to_be_imputed()
        #self.all_pairplots()
        #self.plot_correlation()
        #self.plot_count()
        #self.plot_box()
        

    