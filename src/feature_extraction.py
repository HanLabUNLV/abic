import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA, NMF
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd

#X, y = load_digits(return_X_y=True)
#import our data
data = pd.read_csv('full_feature_matrix.tsv',sep='\t', header=0)

#normalize all non binary variables
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

data.drop(labels=['chr','start','stop','tss','classification','gene','role','abc_score'], axis=1, inplace=True)
data.dropna(inplace=True)

pos_data = data.loc[data['sig']==1,]
neg_data = data.loc[data['sig']==0,].sample(200)

data = pd.concat([pos_data, neg_data])
#print(data['classification'].value_counts())
#exit()

X = data.loc[:,data.columns != 'sig'].to_numpy()
y = data.loc[:,data.columns == 'sig'].to_numpy().T[0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

select = SelectKBest(chi2, k=9)
X_train_selected = select.fit_transform(X_train,y_train)
#print(data.columns[model.get_support(indices=True)])
X_test_selected = select.transform(X_test)
x = data.drop(labels=['sig'], axis=1, inplace=False)
print('Features: ' + ', '.join(x.columns[select.get_support()].tolist()))
clf = LinearDiscriminantAnalysis(solver='lsqr')
clf.fit(X_train_selected, y_train)
print('Training accuracy: ' + str(clf.score(X_train_selected, y_train)))
print('Test accuracy: ' + str(clf.score(X_test_selected, y_test)))
#print('Feature names: ' + str(data.iloc[:,select.get_support()].columns))
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
