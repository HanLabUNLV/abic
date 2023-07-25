import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import joblib


#import data, normalize

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
abc_score = normalizer.fit_transform(data2[['abc_score']].values)
data2.drop(labels=['chr','start','stop','tss','gene','role','abc_score','effect_size'], axis=1, inplace=True)

pos_data2 = data2.loc[data2['sig']==1,]
neg_data2 = data2.loc[data2['sig']==0,]

data2 = pd.concat([pos_data2, neg_data2]).copy()
data2.replace([np.inf, -np.inf], np.nan)
data2.dropna(inplace=True)

X2 = data2.loc[:,data2.columns != 'sig'].to_numpy()
y2 = data2.loc[:,data2.columns == 'sig'].to_numpy().T[0]

X=X2
y=y2

#load trained model
pipe = joblib.load('data/trained_models/contact_models/PCA_RandomForestClassifier.pkl')

#rank features within each PC
n_pcs = pipe.named_steps.PCA.components_.shape[0]
most_important = [np.abs(pipe.named_steps.PCA.components_[i]).argsort()[-3:][::-1].tolist() for i in range(n_pcs)] #returns index of top 3 feats for each pc
#print(most_important)
feat_names = [i for i in data2.columns if i != 'sig']
most_important_names = [[feat_names[j] for j in most_important[i]] for i in range(n_pcs)]
#print(most_important_names)
dic = {'PC{}'.format(i): most_important_names[i] for i in range(n_pcs)}
df = pd.DataFrame(dic.items())
df['explained_var'] = pipe.named_steps.PCA.explained_variance_ratio_
print(df)


#check which pcs are important in rf
feat_import = pipe.named_steps.RandomForestClassifier.feature_importances_
top3_idx = feat_import.argsort()[-3:][::-1]
df['rf_pc_importance'] = feat_import
print(df.iloc[top3_idx])



