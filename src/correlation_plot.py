import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data1 = pd.read_csv('data/full_feature_matrix.coboundp.dataset1.tsv',sep='\t', header=0)
data2 = pd.read_csv('data/full_feature_matrix.coboundp.validated.tsv',sep='\t', header=0)
#for some reason MCM5 didnt run on the new dataset, so remove that col
data1 = data1.loc[:,data1.columns != 'MCM5']
data1 = data1.loc[:,data1.columns != 'MCM5_p_cobound']

data = pd.concat([data1,data2])
data.drop(labels=['chr','start','stop','tss','classification','gene','role','abc_score'], axis=1, inplace=True)

data_trim = data.iloc[:,3:]
cor_mat = data_trim.corr()
csum = cor_mat.sum(axis=1)
print(cor_mat.isna().sum())
print(csum.sort_values(ascending=True)[1:10])
exit()
plt.figure(figsize=(10,10))
sns.heatmap(cor_mat,xticklabels=False, yticklabels=False)
plt.savefig('data/figs/correlation_plot.png')
