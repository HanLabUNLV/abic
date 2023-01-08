from os import listdir as ls
import numpy as np
import time
import joblib
from  scipy.stats import rankdata as rank
import matplotlib.pyplot as plt
from statistics import mean, stdev
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import preprocessing
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold, cross_val_score
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer 
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA, NMF
from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import precision_recall_curve, balanced_accuracy_score, make_scorer
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
import pandas as pd

tstart = time.time()


##################################
#import our data, then format it #
##################################

data2 = pd.read_csv('data/full_feature_matrix.promPOL.subset2.tsv',sep='\t', header=0)
normalizer = preprocessing.MinMaxScaler()
data2['activity'] = normalizer.fit_transform(data2[["activity"]].values)
data2['contact'] = normalizer.fit_transform(data2[["contact"]].values)
data2['abc_score'] = normalizer.fit_transform(data2[["abc_score"]].values)
data2['degree'] = normalizer.fit_transform(data2[["degree"]].values)
data2['effect_size'] = normalizer.fit_transform(data2[["effect_size"]].values)
data2['sig'] = data2.sig.astype(int)

#normalize promoter rnapol signal
for i in ['POLR2A_promoter', 'POLR2AphosphoS2_promoter', 'POLR2AphosphoS5_promoter', 'POLR2B_promoter', 'POLR2G_promoter', 'POLR3A_promoter', 'POLR3G_promoter']:
    data2[i] = normalizer.fit_transform(data2[[i]].values)

#add genomic distance param
data2['midpoint'] = (data2['stop']-data2['start'])/2
data2['distance'] = abs(data2['midpoint'] - data2['tss'])
data2.drop(columns=['midpoint'],inplace=True)
data2['distance'] = normalizer.fit_transform(data2[["distance"]].values)

#remove contact cols
data2.drop(columns=['contact','degree'],inplace=True)
 
#code roles as binary
####data2['e1'] = 0
####data2['e2'] = 0
####data2['e3'] = 0

####data2.loc[data2['role']=='E1','e1'] = 1
####data2.loc[data2['role']=='E2','e2'] = 1
####data2.loc[data2['role']=='E3','e3'] = 1

data2.dropna(inplace=True)
abc_score = data2['abc_score'].values #normalizer.fit_transform(data2[['abc_score']].values)
data2.drop(labels=['chr','start','stop','tss','gene','role','abc_score','effect_size'], axis=1, inplace=True)

pos_data2 = data2.loc[data2['sig']==1,]
neg_data2 = data2.loc[data2['sig']==0,]

data2 = pd.concat([pos_data2, neg_data2]).copy()
data2.replace([np.inf, -np.inf], np.nan)
data2.dropna(inplace=True)

X2 = data2.loc[:,data2.columns != 'sig'].to_numpy()
y2 = data2.loc[:,data2.columns == 'sig'].to_numpy().T[0]

#X=np.concatenate((X1,X2), axis=0)
#y=np.concatenate((y1, y2), axis=0)
#until I add effect_size and degree to data1, train on data2
X=X2
y=y2
#####################################
# load pretrained models for evals  #
#####################################
colors = ['purple','red','blue','orange','cyan','green','pink','yellow','blueviolet']
fig, ax = plt.subplots()

model_dir = 'data/trained_models/no_contact_models_nmf/'
for i in ls(model_dir):
    pipe = joblib.load(model_dir+i)
    if 'RandomForest' in i:
        if 'SelectKBest' in i:
            reduced = pipe.named_steps.SelectKBest.transform(X)
            color = colors[0]
            label = 'SelKBest + RF'
        if 'PCA' in i:
            reduced = pipe.named_steps.PCA.transform(X)
            color = colors[1]
            label = 'PCA + RF'
        if 'NMF' in i:
            reduced = pipe.named_steps.NMF.transform(X)
            color = colors[6]
            label = 'NMF + RF'
        ypred = pipe.named_steps.RandomForestClassifier.predict_proba(reduced)
        precision, recall, thresholds = precision_recall_curve(y, ypred[:,1])
        ax.plot(recall, precision, color=color, label=label)
 

    if 'SVC' in i:
        if 'SelectKBest' in i:
            reduced = pipe.named_steps.SelectKBest.transform(X)
            color = colors[2]
            label = 'SelKBest + SVC'
        if 'PCA' in i:
            reduced = pipe.named_steps.PCA.transform(X)
            color = colors[3]
            label = 'PCA + SVC'
        if 'NMF' in i:
            reduced = pipe.named_steps.NMF.transform(X)
            color = colors[7]
            label = 'NMF + SVC'
        ypred = pipe.named_steps.SVC.decision_function(reduced)
        precision, recall, thresholds = precision_recall_curve(y, ypred)
        ax.plot(recall, precision, color=color, label=label)

    if 'LogisticRegression' in i:
        if 'SelectKBest' in i:
            reduced = pipe.named_steps.SelectKBest.transform(X)
            color = colors[4]
            label = 'SelKBest + LR'
        if 'PCA' in i:
            reduced = pipe.named_steps.PCA.transform(X)
            color = colors[5]
            label = 'PCA + LR'
        if 'NMF' in i:
            reduced = pipe.named_steps.NMF.transform(X)
            color = colors[8]
            label = 'NMF + LR'
        ypred = pipe.named_steps.LogisticRegression.predict_proba(reduced)
        precision, recall, thresholds = precision_recall_curve(y, ypred[:,1])
        ax.plot(recall, precision, color=color, label=label)

#add abc and distance
precision, recall, thresholds = precision_recall_curve(y, data2['distance'].values)
ax.plot(recall, precision, color='black', label='Lin Distance')

precision, recall, thresholds = precision_recall_curve(y, abc_score)
ax.plot(recall, precision, color='brown', label='ABC')

#add axis labels to plot
ax.set_title('Precision-Recall Curve')
ax.set_ylabel('Precision')
ax.set_xlabel('Recall')
ax.legend(loc='upper right')

plt.savefig('data/no_contact_pr_nmf.png')

print('Total runtime: ' + str(time.time() - tstart))    
