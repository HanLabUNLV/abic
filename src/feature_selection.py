import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA, NMF
from sklearn.feature_selection import SelectKBest, chi2
import pandas as pd

pipe = Pipeline(
    [
        # the reduce_dim stage is populated by the param_grid
        ("reduce_dim", "passthrough"),
        ("classify", LinearSVC(dual=False, max_iter=10000)),
    ]
)

N_FEATURES_OPTIONS = [1]
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

grid = GridSearchCV(pipe, n_jobs=1, param_grid=param_grid, scoring='f1_micro')
#X, y = load_digits(return_X_y=True)
#import our data
data = pd.read_csv('full_feature_matrix.tsv',sep='\t', header=0)

#code roles as binary
data['e1'] = 0
data['e2'] = 0
data['e3'] = 0

data.loc[data['role']=='E1','e1'] = 1
data.loc[data['role']=='E2','e2'] = 1
data.loc[data['role']=='E3','e3'] = 1

data.drop(labels=['chr','start','stop','tss','gene','classification','role'], axis=1, inplace=True)
data.dropna(inplace=True)

pos_data = data.loc[data['sig']==1,]
neg_data = data.loc[data['sig']==0,].sample(200)

data = pd.concat([pos_data, neg_data])

X = data.loc[:,data.columns != 'sig'].to_numpy()
y = data.loc[:,data.columns == 'sig'].to_numpy().T[0]


grid.fit(X, y)
print(grid)
exit()
mean_scores = np.array(grid.cv_results_["mean_test_score"])
# scores are in the order of param_grid iteration, which is alphabetical
mean_scores = mean_scores.reshape(len(C_OPTIONS), -1, len(N_FEATURES_OPTIONS))
# select score for best C
mean_scores = mean_scores.max(axis=0)
bar_offsets = np.arange(len(N_FEATURES_OPTIONS)) * (len(reducer_labels) + 1) + 0.5

plt.figure()
COLORS = "bgrcmyk"
for i, (label, reducer_scores) in enumerate(zip(reducer_labels, mean_scores)):
    plt.bar(bar_offsets + i, reducer_scores, label=label, color=COLORS[i])

plt.title("Comparing feature reduction techniques")
plt.xlabel("Reduced number of features")
plt.xticks(bar_offsets + len(reducer_labels) / 2, N_FEATURES_OPTIONS)
plt.ylabel("Enhancer-promoter classification F1 score")
plt.ylim((0, 1))
plt.legend(loc="upper left")
plt.savefig('feature_select_compare.png')
