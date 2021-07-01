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


def audio_extract(audio_file, chroma, mfcc, mel):
    X = audio_file.read(dtype="float32")
    sample_rate = audio_file.samplerate
    if chroma:
        # Fourier transform
        stft = np.abs(librosa.stft(X))
        result = np.array([])
    if mfcc:   
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        result = np.hstack((result, chroma))
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
        result = np.hstack((result, mel))
    return result

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
            self.df[self.cat_features] = imputer.fit_transform(self.df[self.cat_features])
        
    def mapping_data(self):
        for k, v in self.mapping.items():
            if self.df[k].dtypes == 'object':
                self.df[k] = self.df[k].map(v)

    def encoder(self, enc):
        self.df = enc.fit_transform(self.df)

    def dim_reduction(self, thresh=0.95):
        pca = PCA(n_components=self.X.shape[1])
        pca.fit(self.X)
        variances = np.cumsum(pca.explained_variance_ratio_)
        position = np.argmax(variances > thresh)
        print('position to choose: ', position)
        pca_final = PCA(n_components=position)
        self.X = pca_final.fit_transform(self.X)

    def processing(self):
        if 'imputer' in self.methods and self.methods['imputer'] is not None:
            self.missing_imputer(self.methods['imputer'])
        if self.mapping is not None:
            self.mapping_data()

        self.X = self.df[self.features] 
        if self.label is not None: 
            self.y = self.df[self.label]#.values
            if 'resampling' in self.methods and self.methods['resampling'] is not None:
                self.X, self.y = self.methods['resampling'].fit_resample(self.X, self.y)
        if 'dim_reduce' in self.methods and self.methods['dim_reduce'] is True:
            self.dim_reduction()
        if self.label is not None:
            self.process_supervised()

    def process_supervised(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, 
                                                                                test_size=self.test_size, random_state=7)
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