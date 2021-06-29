from data_preparation import DataPreparation
from libs import *

def plot_bar(data, var):
    x = data[var].unique()
    y = data[var].value_counts()
    plt.figure(figsize=(15, 8))
    plt.title(var)
    plt.bar(x, y)
    #plt.set_xticklabels()
    plt.show()
    
def plot_hist(data, var):
    x = data[var].values
    xrange=[min(x), max(x)]
    plt.figure(figsize=(15, 8))
    plt.title(var)
    plt.hist(x, bins=50, range=xrange)
    plt.show()
    
def plot_val(values_x, values_y, name):
    plt.figure(figsize=(15, 8))
    plt.title(name)
    plt.plot(values_x, values_y)
    plt.show()
    

class Clustering(DataPreparation):
    def __init__(self, filename, features, label_col, delimiter=',', data_cat='conventional', test_size=0.2):
        super().__init__(filename, features, label_col, delimiter, data_cat, test_size)
        self.n_best_cluster = 0

    def run(self):
        super().processing()
        n_test = 20
        scores, inertia = [], []
        for i in range(2, n_test):
            kmeans = KMeans(n_clusters=i, random_state=7).fit(self.X)
            pred = kmeans.predict(self.X)
            silhouette = silhouette_score(self.X, pred)
            scores.append(silhouette)
            inertia.append(kmeans.inertia_)
        n_cluster_chosen = scores.index(max(scores))+2
        self.n_best_cluster = n_cluster_chosen
        print('best n cluster: ', n_cluster_chosen)
        plot_val(range(2, n_test), scores, 'Sihouette')
        plot_val(range(2, n_test), inertia, 'Sum of squares')

    def plot_clusters(self, var1, var2):
        label = KMeans(n_clusters=self.n_best_cluster, random_state=7).fit_predict(self.X)
        colors = ['red', 'green', 'blue', 'cyan', 'yellow', 'gray']
        plt.figure(figsize=(15, 8))
        for i in range(self.n_best_cluster):
            data = self.df[label == i]
            print(i, data)
            plt.scatter(data[var1], data[var2], color=colors[i])
        plt.show()