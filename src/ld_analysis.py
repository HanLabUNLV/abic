import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA, NMF
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd

pipe = Pipeline(
    [
        # the reduce_dim stage is populated by the param_grid
        ("reduce_dim", "passthrough"),
        ("classify", LinearSVC(dual=False, max_iter=10000)),
    ]
)

N_FEATURES_OPTIONS = [1, 2, 4, 8, 10, 15]
C_OPTIONS = [1, 10, 100, 1000]
param_grid = [
    {
        "reduce_dim": [PCA(iterated_power=7), NMF()],
        "reduce_dim__n_components": N_FEATURES_OPTIONS,
        "classify__C": C_OPTIONS,
    },
    {
        "reduce_dim": [SelectKBest(chi2)],
        "reduce_dim__k": N_FEATURES_OPTIONS,
        "classify__C": C_OPTIONS,
    },
]
reducer_labels = ["PCA", "NMF", "KBest(chi2)"]

clf = LinearDiscriminantAnalysis(solver='lsqr')

#X, y = load_digits(return_X_y=True)
#import our data
data = pd.read_csv('full_feature_matrix.tsv',sep='\t', header=0)

#normalize data
normalizer = preprocessing.MinMaxScaler()
data['activity'] = normalizer.fit_transform(data[["activity"]].values)
data['contact'] = normalizer.fit_transform(data[["contact"]].values)
data['abc_score'] = normalizer.fit_transform(data[["abc_score"]].values)

#code roles as binary
data['e1'] = 0
data['e2'] = 0
data['e3'] = 0

data.loc[data['role']=='E1','e1'] = 1
data.loc[data['role']=='E2','e2'] = 1
data.loc[data['role']=='E3','e3'] = 1

sample_labels = [['chr','start','stop','tss','gene','classification','role'],data.columns.difference(['abc_score', 'sig']).tolist(),data.columns.difference(['activity','sig']).tolist(),['abc_score','chr','start','stop','tss','gene','classification','role','activity','e1','e2','e3']]
tval = []
for i in [0,1,2,3]:
    #1: all vbles
    #2: only abc
    #3: only activity
    #4: only chipseq
    sample = data.drop(labels=sample_labels[i], axis=1, inplace=False)
    sample.dropna(inplace=True)

    pos_data = sample.loc[sample['sig']==1,]
    neg_data = sample.loc[sample['sig']==0,].sample(200)

    sample = pd.concat([pos_data, neg_data])

    X = sample.loc[:,sample.columns != 'sig'].to_numpy()
    y = sample.loc[:,sample.columns == 'sig'].to_numpy().T[0]


    clf.fit(X, y)
    tval.append(clf.score(X, y))

reducer_labels = ['All','Only ABC','Only Activity','Only CHiPSeq']
mean_scores = tval

plt.figure()
COLORS = "bgrcmyk"
for i in [0,1,2,3]:
    plt.bar(i, mean_scores[i], color=COLORS[i])

plt.title("LDA")
plt.xlabel("Input Variable Set")
plt.xticks([0,1,2,3], reducer_labels)
plt.ylabel("Enhancer-promoter classification accuracy")
plt.ylim((0, 1))
#plt.legend(loc="upper left")
plt.savefig('LDA_accuracy.png')
