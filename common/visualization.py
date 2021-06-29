from libs import *

class Visualization():
    def __init__(self, df):
        self.df = df

    def plot_bar(self, var):
        #plt.figure(figsize=(15, 8))
        #plt.bar(x, self.df[var])
        #plt.set_xticks()
        #plt.show()
        pass

    def plot_hist(self, var, bins=50):
        plt.figure(figsize=(20, 20))
        plt.hist(self.df[var], bins=bins)
        plt.show()

    def plot_corr(self, features):
        plt.figure(figsize=(15, 8))
        sns.heatmap(self.df[features].corr(), annot=True)
        plt.show()

    def plot_pairplot(self, features, hue=None):
        plt.figure(figsize=(15, 8))
        sns.pairplot(data=self.df, vars=features, hue=hue)
        plt.show()

    def plot_timeseries(self):
        pass
