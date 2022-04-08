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
from sklearn.metrics import precision_recall_curve
from sklearn.svm import LinearSVC
import pandas as pd

out = open('logs/feature_extraction.log','w')

def plot_coefficients(classifier, feature_names, top_features=20):
    coef = classifier.coef_.ravel()
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # create plot
    plt.figure(figsize=(15, 5))
    colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(1, 1 +2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
    plt.tight_layout()
    plt.savefig('data/top_20_features.png')

#X, y = load_digits(return_X_y=True)
#import our data
data1 = pd.read_csv('data/full_feature_matrix.coboundp.dataset1.tsv',sep='\t', header=0)
data2 = pd.read_csv('data/full_feature_matrix.coboundp.merged.tsv',sep='\t', header=0)

#for some reason MCM5 didnt run on the new dataset, so remove that col
data1 = data1.loc[:,data1.columns != 'MCM5']
data1 = data1.loc[:,data1.columns != 'MCM5_p_cobound']
#data = pd.concat([data1,data2],ignore_index=True)
#normalize all non binary variables
normalizer = preprocessing.MinMaxScaler()
data1['activity'] = normalizer.fit_transform(data1[["activity"]].values)
data1['contact'] = normalizer.fit_transform(data1[["contact"]].values)
data1['abc_score'] = normalizer.fit_transform(data1[["abc_score"]].values)


data2['activity'] = normalizer.fit_transform(data2[["activity"]].values)
data2['contact'] = normalizer.fit_transform(data2[["contact"]].values)
data2['abc_score'] = normalizer.fit_transform(data2[["abc_score"]].values)

plt.hist(data1['activity'], density=True, bins=50)
plt.xlabel('activity normalized')
plt.savefig('data/ds1_activity_histogram.png')

plt.hist(data2['activity'], density=True, bins=50)
plt.xlabel('activity normalized')
plt.savefig('data/ds2_activity_histogram.png')

#code roles as binary
data1['e1'] = 0
data1['e2'] = 0
data1['e3'] = 0

data2['e1'] = 0
data2['e2'] = 0
data2['e3'] = 0

data1.loc[data1['role']=='E1','e1'] = 1
data1.loc[data1['role']=='E2','e2'] = 1
data1.loc[data1['role']=='E3','e3'] = 1

data2.loc[data2['role']=='E1','e1'] = 1
data2.loc[data2['role']=='E2','e2'] = 1
data2.loc[data2['role']=='E3','e3'] = 1

data1.drop(labels=['chr','start','stop','tss','classification','gene','role','abc_score'], axis=1, inplace=True)
data1.dropna(inplace=True)

data2.drop(labels=['chr','start','stop','tss','classification','gene','role','abc_score'], axis=1, inplace=True)
data2.dropna(inplace=True)

pos_data1 = data1.loc[data1['sig']==1,]
neg_data1 = data1.loc[data1['sig']==0,].sample(200)

pos_data2 = data2.loc[data2['sig']==1,]
neg_data2 = data2.loc[data2['sig']==0,].sample(200)

data1 = pd.concat([pos_data1, neg_data1])

data2 = pd.concat([pos_data2, neg_data2])

X1 = data1.loc[:,data1.columns != 'sig'].to_numpy()
y1 = data1.loc[:,data1.columns == 'sig'].to_numpy().T[0]

X2 = data2.loc[:,data2.columns != 'sig'].to_numpy()
y2 = data2.loc[:,data2.columns == 'sig'].to_numpy().T[0]

test_sz = 0.2
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=test_sz)

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=test_sz)

X_train = np.concatenate((X1_train, X2_train), axis=0)
X_test = np.concatenate((X1_test, X2_test), axis=0)
y_train = np.concatenate((y1_train, y2_train), axis=0)
y_test = np.concatenate((y1_test, y2_test), axis=0)

#nfeat = 9 #num feat selected
for nfeat in [100]:#[5,10,15,30,50,100,200]:
    select = SelectKBest(chi2, k=nfeat)
    X_train_selected = select.fit_transform(X_train,y_train)
    #print(data.columns[model.get_support(indices=True)])
    X_test_selected = select.transform(X_test)
    x = data1.drop(labels=['sig'], axis=1, inplace=False)
    out.write('k='+str(nfeat)+'\n')
    out.write('Features: ' + ', '.join(x.columns[select.get_support()].tolist())+'\n')
    clf = LinearSVC()
    clf.fit(X_train_selected, y_train)
    out.write('Training accuracy: ' + str(clf.score(X_train_selected, y_train))+'\n')
    out.write('Test accuracy: ' + str(clf.score(X_test_selected, y_test))+'\n')
    out.write('\n\n')

    #generate coef plot
    plot_coefficients(clf, data1.loc[:,data1.columns != 'sig'].columns[select.get_support()], 40)
    #next generate pr curve

    y_score = clf.decision_function(X_test_selected)#predict_proba(X_test_selected)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_score)
    fig, ax = plt.subplots()
    ax.plot(recall, precision, color='blue')
    ax.set_title('Precision-Recall Curve k='+str(nfeat))
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    plt.savefig('data/figs/pr_curve.'+str(nfeat)+'.png')
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
