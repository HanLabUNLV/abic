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
import seaborn as sns

tstart = time.time()


##################################
#import our data, then format it #
##################################

data2 = pd.read_csv('/data8/han_lab/mhan/abcd/data/Gasperini2019.at_scale.ABC.TF.txt',sep='\t', header=0)

#rename enh tfbs cols
data2.columns = data2.columns.str.replace(r'_e','')
data2.columns = data2.columns.str.replace(r'_TSS','_p_cobound')

#delete some cols
#data2.drop(labels=['chr','start','stop','tss','gene','role','abc_score','effect_size'], axis=1, inplace=True)
data2.drop(labels=['TargetGeneTSS','Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'name_x', 'chrEnhancer', 'startEnhancer', 'endEnhancer', 'chrTSS', 'startTSS','endTSS', 'EffectSize', 'strandGene', 'CellType_x', 'pValue', 'EnsemblD', 'GeneSymbol', 'Reference', 'PerturbationMethod', 'ReadoutMethod', 'EnhancerID', 'GenomeBuild', 'gRNAs', 'Notes', 'pValueAdjusted', 'enhancerID', 'G.id', 'ABC.id', 'chr_x', 'start_x', 'end_x', 'name_y', 'class_x', 'normalized_h3K27ac', 'normalized_dhs',  'chr:start-end_TargetGene', 'chr_y', 'start_y', 'end_y', 'name', 'class_y', 'activity_base_y', 'TargetGene', 'TargetGeneExpression', 'TargetGenePromoterActivityQuantile', 'TargetGeneIsExpressed','isSelfPromoter', 'powerlaw_contact', 'powerlaw_contact_reference', 'hic_contact', 'hic_contact_pl_scaled', 'hic_pseudocount', 'ABC.Score.Numerator', 'ABC.Score', 'powerlaw.Score.Numerator', 'powerlaw.Score', 'CellType_y'], axis=1, inplace=True)



#data2['source'] = 'gasperini'
#data1['source'] = 'fulco'
#for some reason MCM5 didnt run on the new dataset, so remove that col
#data = pd.concat([data1,data2],ignore_index=True)
#normalize all non binary variables
normalizer = preprocessing.MinMaxScaler()

data2['activity'] = normalizer.fit_transform(data2[["activity_base_x"]].values)
data2['contact'] = normalizer.fit_transform(data2[["hic_contact_pl_scaled_adj"]].values)
data2['sig'] = data2.Significant.astype(int)

data2.drop(labels=['Significant','activity_base_x','hic_contact_pl_scaled_adj'], axis=1, inplace=True)

pos_data2 = data2.loc[data2['sig']==1,]
neg_data2 = data2.loc[data2['sig']==0,]

data2 = pd.concat([pos_data2, neg_data2]).copy()
data2.replace([np.inf, -np.inf], np.nan)
data2.dropna(inplace=True)

X2 = data2.loc[:,data2.columns != 'sig'].to_numpy()
y2 = data2.loc[:,data2.columns == 'sig'].to_numpy().T[0]
fnames = [x for x in data2.columns]
fnames.remove('sig')
#X=np.concatenate((X1,X2), axis=0)
#y=np.concatenate((y1, y2), axis=0)
#until I add effect_size and degree to data1, train on data2
X=X2
y=y2

#####################################
# load pretrained models for evals  #
#####################################
model_dir = 'data/trained_models/mira_data/'
i = 'SelectKBest_LogisticRegression.pkl'
pipe = joblib.load(model_dir+i)
coef = pipe.named_steps.LogisticRegression.coef_[0]
fnames = [i for i in data2.columns]

def f_importances(coef, names):
    imp = coef
    imp,names = zip(*sorted(zip(imp,names)))
    #plt.show()

    fig, ax = plt.subplots()
    plt.figure(figsize=(15,15))
    #plt.bar(range(len(importances)), importances, tick_label = fnames)
    ax.set_title(i.split('.')[0]+' Component Importances')
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.savefig('data/'+i.split('.')[0]+'_importances.m.png')

f_importances(coef, fnames)

print('Total runtime: ' + str(time.time() - tstart))    
