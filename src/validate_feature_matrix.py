import pandas as pd
import pyranges as pr
 
enhancers = pd.read_csv('data/full_feature_matrix.validated2.tsv', sep='\t')
#enhancers  = enhancers.rename(columns={'end':'stop'})
s2a = pd.read_csv('data/s2a.bed', sep='\t', header=None)
s2a = s2a.rename(columns={0: 'Chromosome', 1:'Start', 2:'End'})
s2b = pd.read_csv('data/s2b.bed', sep='\t',header=None)
s2b = s2b.rename(columns={0: 'Chromosome', 1:'Start', 2:'End',3:'gene'})


#use pyranges to get overlap, only enhancers that were tested

edf = enhancers.loc[:,['chr','start','stop','gene']]
edf.dropna(axis=0, how='any', inplace=True)
epr = pr.PyRanges(edf.rename(columns={'chr':'Chromosome','start':'Start','stop':'End'}))

s2apr = pr.PyRanges(s2a)

#s2bpr = pr.PyRanges(s2b)

es2a = epr.overlap(s2apr).as_df().rename(columns={'Chromosome':'chr','Start':'start','End':'stop'})
#es2b = epr.overlap(s2bpr).as_df().rename(columns={'Chromosome':'chr','Start':'start','End':'stop'})


#df_all = enhancers.merge(es2a, on=['chr','start','stop'], how='inner', left_index=True)
eidx = enhancers.set_index(['chr','start','stop']).index
s2aidx = es2a.set_index(['chr','start','stop']).index 

#print(enhancers['sig'].value_counts())
#enhancers.loc[eidx.isin(s2bidx), 'sig']=1
#print(enhancers['sig'].value_counts())
df_all = enhancers.loc[(eidx.isin(s2aidx))].copy()
#now all enhancers tested, still need to verify which are significant

edf = df_all.loc[:,['chr','start','stop','gene']].copy()
edf.dropna(axis=0, how='all', inplace=True)
epr = pr.PyRanges(edf.rename(columns={'chr':'Chromosome','start':'Start','stop':'End'}))
#to do this we iterate through table s2b
log = open('logs/positives_missing_from_feature_matrix.2.log','w')
total_changed =0
for idx, row in s2b.iterrows():
    chrm = row['Chromosome']
    start = row['Start']
    end = row['End']
    gene = row['gene']
    #print([chrm, start, end, gene])
    tpr = pr.from_dict({'Chromosome':[chrm], 'Start':[start], 'End':[end]})
    ov = epr.overlap(tpr)
    if len(ov) > 0:
        glist = ov.as_df().gene.tolist()
        if gene not in glist:
            log.write('\t'.join(['_'.join([chrm, str(start), str(end)]), gene, ','.join(glist)])+'\n')
        else:
            chrm1 = ov.as_df().Chromosome.tolist()[0]
            start1 = ov.as_df().Start.tolist()[0]
            stop1 = ov.as_df().End.tolist()[0]
            
            df_all.loc[((df_all['chr']==chrm1)&(df_all['start']==start1)&(df_all['stop']==stop1)&(df_all['gene']==gene)),'sig'] = 1.0
            #print(row)
            total_changed +=1

#print(len(df_all))
#print(len(enhancers))

#print(df_all['sig'].value_counts())

#df_all.rename(columns={'stop':'end'})
df_all.to_csv('data/full_feature_matrix.revalidated2.tsv',sep='\t', index=False)






