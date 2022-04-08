import pandas as pd
 
enhancers = pd.read_csv('raw_data/EnhancerPredictionsAllPutative_Gasperini.txt', sep='\t')
enhancers  = enhancers.rename(columns={'end':'stop'})
#s2a = pd.read_csv('raw_data/gasperini.s2a.csv', sep='\t', header=None)
#s2a = s2a.rename(columns={0: 'spacer', 1:'id', 2:'chr', 3:'start', 4:'stop', 5:'description'})
s2b = pd.read_csv('data/enhancers_significant.bed', sep='\t',header=None)
s2b = s2b.rename(columns={0:'chr',1:'start',2:'stop',3:'s2b.chr', 4:'s2b.start',5:'s2b.stop',6:'TargetGene',7:'overlap'})

df_all = enhancers.merge(s2b, on=['chr','start','stop', 'TargetGene'], how='left', indicator=True)

df_all['significant']=0 
df_all.loc[df_all['_merge']=='both', 'significant']=1
df_all.drop(columns=['s2b.chr','s2b.start','s2b.stop','overlap','_merge'], inplace=True)
df_all.rename(columns={'stop':'end'})
df_all.to_csv('data/enhancers.gas.class.subset1.tsv',sep='\t')






