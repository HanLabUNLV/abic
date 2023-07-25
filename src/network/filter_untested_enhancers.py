import pandas as pd
import pyranges as pr 
enhancers = pd.read_csv('data/full_feature_matrix.coboundp.tsv', sep='\t')
enhancers['start'] = enhancers['start'].astype(int)
enhancers['stop'] = enhancers['stop'].astype(int)
enhancers['tss'] = enhancers['tss'].astype(int)
enhancers.rename(columns = {'chr':'Chromosome', 'start':'Start', 'stop':'End'}, inplace=True)
validation = pd.read_csv('data/s2a.bed', sep='\t', header=None)
validation.rename(columns={0:'Chromosome',1:'Start',2:'End'}, inplace=True)

enh_pr, val_pr = pr.PyRanges(enhancers), pr.PyRanges(validation)
result = enh_pr.intersect(val_pr)
outdf = result.as_df()
outdf.rename(columns = {'Chromosome':'chr', 'Start':'start', 'End':'stop'}, inplace=True)
print(outdf.columns)
outdf.to_csv('data/full_feature_matrix.coboundp.validated.tsv', sep='\t', index=False)
exit()

#df_all = enhancers.merge(validation, on=['chr','start','stop'], how='left', indicator=True)

#df_all['significant']=0
#df_all.loc[df_all['_merge']=='both', 'significant']=1
#df_all.drop(columns=['_merge'], inplace=True)
#df_all.rename(columns={'stop':'end'})
df_all.to_csv('data/full_feature_matrix.coboundp.validated.tsv',sep='\t')

