from data_preparation import DataPreparation
from libs import *
from modules_DL import *

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

    def find_clusters(self, data):
        scores = []
        n_test = 20
        for i in range(2, n_test):
            cluster = KMeans(n_clusters=i, random_state=7).fit(data)
            pred = cluster.predict(data)
            silhouette = silhouette_score(data, pred)
            scores.append(silhouette)
            #inertia.append(cluster.inertia_)
        n_cluster_chosen = scores.index(max(scores))+2
        self.n_best_cluster = n_cluster_chosen
        plot_val(range(2, n_test), scores, 'Sihouette')

    def auto_encode(self):
        #print(self.X)
        X_scaled = StandardScaler().fit_transform(self.X)
        #print(X_scaled)
        autoencoder, encoder = model_auto_encoder(X_scaled.shape[1])
        autoencoder.fit(X_scaled, X_scaled, batch_size=32, epochs = 25, verbose = 1)
        preds = encoder.predict(X_scaled)
        return preds

    def run(self, autoencode=False):
        super().processing()
        if autoencode:
            data = self.auto_encode().astype('float')
        else:
            data = self.X
        print(data.shape)
        self.find_clusters(data)
        print('best n cluster: ', self.n_best_cluster)

    def plot_clusters(self, var1, var2):
        label = KMeans(n_clusters=self.n_best_cluster, random_state=7).fit_predict(self.X)
        #colors = mcolors.BASE_COLORS #['red', 'green', 'blue', 'cyan', 'yellow', 'gray']
        print(mcolors.BASE_COLORS)
        colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

        # Sort colors by hue, saturation, value and name.
        by_hsv = [(tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name) for name, color in colors.items()]
        colors = [name for hsv, name in by_hsv]
        plt.figure(figsize=(15, 8))
        for i in range(self.n_best_cluster):
            data = self.df[label == i]
            plt.scatter(data[var1], data[var2], color=colors[i])
        plt.show()