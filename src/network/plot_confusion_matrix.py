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
from sklearn.metrics import precision_recall_curve, balanced_accuracy_score, make_scorer, plot_confusion_matrix
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
#data2.drop(columns=['contact','degree'],inplace=True)
 
#code roles as binary
data2['e1'] = 0
data2['e2'] = 0
data2['e3'] = 0

data2.loc[data2['role']=='E1','e1'] = 1
data2.loc[data2['role']=='E2','e2'] = 1
data2.loc[data2['role']=='E3','e3'] = 1

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
model_dir = 'data/trained_models/contact_models_nmf/'
for i in ls(model_dir):
    pipe = joblib.load(model_dir+i)
    fig, ax = plt.subplots()
    plot_confusion_matrix(pipe, X, y,normalize='true', ax=ax)
    ax.set_title(i.split('.')[0]) 
    plt.savefig('data/conf_matrix/'+i.split('.')[0]+'.png')



print('Total runtime: ' + str(time.time() - tstart))    
