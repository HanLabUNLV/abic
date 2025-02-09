import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt








# ABC
data_dir = "data/Gasperini/"
Gasperini_enhancer = pd.read_csv(data_dir+"Gasperini2019.enhancer.ABC.overlap.bed", sep='\t')
Gasperini_TSS = pd.read_csv(data_dir+"Gasperini2019.TSS.ABC.overlap.bed", sep='\t')
#Gasperini_atscale = pd.read_csv(data_dir+"Gasperini2019.at_scale_screen.cand_enhancer_x_exprsd_genes.200503.csv")
#Gasperini_atscale = pd.read_csv(data_dir+"GSE120861_all_deg_results.at_scale.dropNA.mapID.txt", sep='\t')
Gasperini_atscale = pd.read_csv(data_dir+"GSE120861_all_deg_results.at_scale.dropNA.dropTSS.mapID.txt", sep='\t')
#Gasperini_atscale['pValueAdjusted'] = Gasperini_atscale['pValueAdjusted'].fillna(1)
#Gasperini_atscale.dropna(subset=['pValueAdjusted'], inplace=True)

ABC = pd.read_csv(data_dir+"ABC.EnhancerPredictionsAllPutative.txt", sep='\t')
ABC_by_gene = ABC.groupby('TargetGene')
ABC['Enhancer.count.near.TSS'] = ABC_by_gene[['TargetGene']].transform('count')
ABC['mean.contact.to.TSS'] = ABC_by_gene[['hic_contact']].transform('mean')
ABC['std.contact.to.TSS'] = ABC_by_gene[['hic_contact']].transform('std')
ABC['std.contact.to.TSS'] = ABC['std.contact.to.TSS'].fillna(0)
ABC['zscore.contact.to.TSS'] = (ABC['hic_contact'] - ABC['mean.contact.to.TSS']) / ABC['std.contact.to.TSS']
ABC['zscore.contact.to.TSS'] = ABC['zscore.contact.to.TSS'].fillna(0)
ABC['max.contact.to.TSS'] = ABC_by_gene[['hic_contact']].transform('max')
ABC['diff.from.max.contact.to.TSS'] = ABC['hic_contact']-ABC['max.contact.to.TSS']
ABC['total.contact.to.TSS'] = ABC_by_gene[['hic_contact']].transform('sum')
ABC['remaining.enhancers.contact.to.TSS'] = ABC['total.contact.to.TSS'] - ABC['hic_contact']
ABC_by_enhancer = ABC.groupby('name')
ABC['TSS.count.near.enhancer'] = ABC_by_enhancer[['name']].transform('count')
ABC['mean.contact.from.enhancer'] = ABC_by_enhancer[['hic_contact']].transform('mean')
ABC['std.contact.from.enhancer'] = ABC_by_enhancer[['hic_contact']].transform('std')
ABC['std.contact.from.enhancer'] = ABC['std.contact.from.enhancer'].fillna(0)
ABC['zscore.contact.from.enhancer'] = (ABC['hic_contact'] - ABC['mean.contact.from.enhancer']) / ABC['std.contact.from.enhancer']
ABC['zscore.contact.from.enhancer'] = ABC['zscore.contact.from.enhancer'].fillna(0)
ABC['max.contact.from.enhancer'] = ABC_by_enhancer[['hic_contact']].transform('max')
ABC['diff.from.max.contact.from.enhancer'] = ABC['hic_contact']-ABC['max.contact.from.enhancer']
ABC['total.contact.from.enhancer'] = ABC_by_enhancer[['hic_contact']].transform('sum')
ABC['remaining.TSS.contact.from.enhancer'] = ABC['total.contact.from.enhancer'] - ABC['hic_contact']
# combined
ABC['nearby.counts'] = ABC['Enhancer.count.near.TSS']+ABC['TSS.count.near.enhancer']
ABC['mean.contact'] = ((ABC['Enhancer.count.near.TSS']*ABC['mean.contact.to.TSS'])+(ABC['TSS.count.near.enhancer']*ABC['mean.contact.from.enhancer'])) / (ABC['Enhancer.count.near.TSS']+ABC['TSS.count.near.enhancer'])
q1 = (ABC['Enhancer.count.near.TSS']-1)*(ABC['std.contact.to.TSS'].pow(2)) + (ABC['Enhancer.count.near.TSS'])*(ABC['mean.contact.to.TSS'].pow(2))
q2 = (ABC['Enhancer.count.near.TSS']-1)*(ABC['std.contact.from.enhancer'].pow(2)) + (ABC['Enhancer.count.near.TSS'])*(ABC['mean.contact.from.enhancer'].pow(2))
qc = (q1 + q2) 
meansq = (ABC['Enhancer.count.near.TSS']+ABC['TSS.count.near.enhancer'])*(ABC['mean.contact'].pow(2))
denom = (ABC['Enhancer.count.near.TSS']+ABC['TSS.count.near.enhancer']-1)
ABC['std.contact'] =  (( qc - meansq ) / denom).pow(1./2)
ABC['zscore.contact'] = (ABC['hic_contact'] - ABC['mean.contact']) / ABC['std.contact']
ABC['zscore.contact'] = ABC['zscore.contact'].fillna(0)
ABC['max.contact'] = ABC[['max.contact.to.TSS', 'max.contact.from.enhancer']].max(axis=1)
ABC['diff.from.max.contact'] = ABC['hic_contact']-ABC['max.contact']
ABC['total.contact'] = ABC[['total.contact.to.TSS', 'total.contact.from.enhancer']].sum(axis=1)
ABC['remaining.contact'] = ABC['total.contact'] - ABC['hic_contact']



new = Gasperini_atscale["name"].str.split(":", n=1, expand=True)
Gasperini_atscale["enhancerID"]= new[0]
Gasperini_atscale["Significant"] = Gasperini_atscale["pValueAdjusted"] < 0.1
Gasperini_atscale.to_csv(data_dir+"GSE120861_all_deg_results.at_scale.dropNA.mapID.sig.txt", sep='\t')
Gasperini_enhancer = Gasperini_enhancer[["G.id", "ABC.id"]]

ABC_enhancer = pd.read_csv(data_dir+"EnhancerList.txt", sep="\t")
ABC_enhancer = ABC_enhancer.loc[:,['chr','start','end','name','class', 'normalized_h3K27ac', 'normalized_dhs', 'activity_base']]
ABC_enhancer["ABC.id"] = ABC_enhancer['chr']+":"+ABC_enhancer['start'].astype(str)+"-"+ABC_enhancer['end'].astype(str)
ABC_enhancer_H3K4me3 = pd.read_csv(data_dir+"EnhancerList.H3K4me3.txt", sep="\t")
ABC_enhancer_H3K4me3.rename({'normalized_h3K27ac': 'normalized_h3K4me3'}, axis=1, inplace=True)
ABC_enhancer_H3K4me3 = ABC_enhancer_H3K4me3.loc[:,['chr','start','end','normalized_h3K4me3']]
ABC_enhancer_H3K4me3["ABC.id"] = ABC_enhancer_H3K4me3['chr']+":"+ABC_enhancer_H3K4me3['start'].astype(str)+"-"+ABC_enhancer_H3K4me3['end'].astype(str)
ABC_enhancer_H3K4me3.drop(columns=['chr','start','end'], axis=1, inplace=True)
ABC_enhancer_H3K27me3 = pd.read_csv(data_dir+"EnhancerList.H3K27me3.txt", sep="\t")
ABC_enhancer_H3K27me3.rename({'normalized_h3K27ac': 'normalized_h3K27me3'}, axis=1, inplace=True)
ABC_enhancer_H3K27me3 = ABC_enhancer_H3K27me3.loc[:,['chr','start','end','normalized_h3K27me3']]
ABC_enhancer_H3K27me3["ABC.id"] = ABC_enhancer_H3K27me3['chr']+":"+ABC_enhancer_H3K27me3['start'].astype(str)+"-"+ABC_enhancer_H3K27me3['end'].astype(str)
ABC_enhancer_H3K27me3.drop(columns=['chr','start','end'], axis=1, inplace=True)
#ABC_enhancer_H3K4me1 = pd.read_csv(data_dir+"EnhancerList.H3K4me1.txt", sep="\t")
#ABC_enhancer_H3K4me1.rename({'normalized_h3K27ac': 'normalized_h3K4me1'}, axis=1, inplace=True)
#ABC_enhancer_H3K4me1 = ABC_enhancer_H3K4me1.loc[:,['chr','start','end','normalized_h3K4me1']]
#ABC_enhancer_H3K4me1["ABC.id"] = ABC_enhancer_H3K4me1['chr']+":"+ABC_enhancer_H3K4me1['start'].astype(str)+"-"+ABC_enhancer_H3K4me1['end'].astype(str)
#ABC_enhancer_H3K4me1.drop(columns=['chr','start','end'], axis=1, inplace=True)

Gasperini_enhancer = pd.merge(Gasperini_enhancer, ABC_enhancer, left_on=["ABC.id"], right_on=["ABC.id"], suffixes=('', '_y'))
Gasperini_enhancer.drop(Gasperini_enhancer.filter(regex='_y$').columns, axis=1, inplace=True)
Gasperini_enhancer = pd.merge(Gasperini_enhancer, ABC_enhancer_H3K4me3, left_on=["ABC.id"], right_on=["ABC.id"], suffixes=('', '_y'))
Gasperini_enhancer.drop(Gasperini_enhancer.filter(regex='_y$').columns, axis=1, inplace=True)
Gasperini_enhancer = pd.merge(Gasperini_enhancer, ABC_enhancer_H3K27me3, left_on=["ABC.id"], right_on=["ABC.id"], suffixes=('', '_y'))
Gasperini_enhancer.drop(Gasperini_enhancer.filter(regex='_y$').columns, axis=1, inplace=True)
#Gasperini_enhancer = pd.merge(Gasperini_enhancer, ABC_enhancer_H3K4me1, left_on=["ABC.id"], right_on=["ABC.id"], suffixes=('', '_y'))
#Gasperini_enhancer.drop(Gasperini_enhancer.filter(regex='_y$').columns, axis=1, inplace=True)

Gasperini_atscale2 = pd.merge(Gasperini_atscale, Gasperini_enhancer, left_on=["enhancerID"], right_on=["G.id"], suffixes=('', '_y'))
Gasperini_atscale2.drop(Gasperini_atscale2.filter(regex='_y$').columns, axis=1, inplace=True)
Gasperini_atscale2["ABC.id"] = Gasperini_atscale2["ABC.id"] + "_" + Gasperini_atscale2["GeneSymbol"]
Gasperini_atscale2.to_csv(data_dir+"Gasperini2019.at_scale2.txt", sep='\t')



ABC_gene = pd.read_csv(data_dir+"GeneList.txt", sep="\t")
ABC_gene = ABC_gene.loc[:,['name','H3K27ac.RPKM.quantile.TSS1Kb']]
ABC_gene_H3K4me3 = pd.read_csv(data_dir+"GeneList.H3K4me3.txt", sep="\t")
ABC_gene_H3K4me3.rename({'H3K27ac.RPKM.quantile.TSS1Kb': 'H3K4me3.RPKM.quantile.TSS1Kb'}, axis=1, inplace=True)
ABC_gene_H3K4me3 = ABC_gene_H3K4me3.loc[:,['name','H3K4me3.RPKM.quantile.TSS1Kb']]
ABC_gene_H3K27me3 = pd.read_csv(data_dir+"GeneList.H3K27me3.txt", sep="\t")
ABC_gene_H3K27me3.rename({'H3K27ac.RPKM.quantile.TSS1Kb': 'H3K27me3.RPKM.quantile.TSS1Kb'}, axis=1, inplace=True)
ABC_gene_H3K27me3 = ABC_gene_H3K27me3.loc[:,['name','H3K27me3.RPKM.quantile.TSS1Kb']]
#ABC_gene_H3K4me1 = pd.read_csv(data_dir+"GeneList.H3K4me1.txt", sep="\t")
#ABC_gene_H3K4me1.rename({'H3K27ac.RPKM.quantile.TSS1Kb': 'H3K4me1.RPKM.quantile.TSS1Kb'}, axis=1, inplace=True)
#ABC_gene_H3K4me1 = ABC_gene_H3K4me1.loc[:,['name','H3K4me1.RPKM.quantile.TSS1Kb']]
ABC = pd.merge(ABC, ABC_gene, left_on=["TargetGene"], right_on=["name"], suffixes=('', '_y'))
ABC.drop(ABC.filter(regex='_y$').columns, axis=1, inplace=True)
ABC = pd.merge(ABC, ABC_gene_H3K4me3, left_on=["TargetGene"], right_on=["name"], suffixes=('', '_y'))
ABC.drop(ABC.filter(regex='_y$').columns, axis=1, inplace=True)
ABC = pd.merge(ABC, ABC_gene_H3K27me3, left_on=["TargetGene"], right_on=["name"], suffixes=('', '_y'))
ABC.drop(ABC.filter(regex='_y$').columns, axis=1, inplace=True)
#ABC = pd.merge(ABC, ABC_gene_H3K4me1, left_on=["TargetGene"], right_on=["name"], suffixes=('', '_y'))
#ABC.drop(ABC.filter(regex='_y$').columns, axis=1, inplace=True)

# gene prediction training data 
Gasperini_atscale_ABC = pd.merge(Gasperini_atscale2, ABC, left_on=["ABC.id"], right_on=["chr:start-end_TargetGene"], suffixes=('', '_y'))
Gasperini_atscale_ABC.drop(Gasperini_atscale_ABC.filter(regex='_y$').columns, axis=1, inplace=True)
# by_gene
Gasperini_ABC_by_gene = Gasperini_atscale_ABC.groupby('TargetGene')
Gasperini_ABC_by_gene_symbol = Gasperini_ABC_by_gene[['GeneSymbol', 'chr', 'chrTSS',  'startTSS',  'endTSS', 'Enhancer.count.near.TSS', 'mean.contact.to.TSS', 'max.contact.to.TSS', 'total.contact.to.TSS']].first()
Gasperini_ABC_by_gene_sig = Gasperini_ABC_by_gene[['Significant', 'TargetGeneIsExpressed']].any()
#Gasperini_ABC_by_gene_means = Gasperini_ABC_by_gene[['TargetGeneTSS', 'TargetGeneExpression','TargetGenePromoterActivityQuantile','H3K27ac.RPKM.quantile.TSS1Kb','H3K4me3.RPKM.quantile.TSS1Kb', 'H3K27me3.RPKM.quantile.TSS1Kb', 'H3K4me1.RPKM.quantile.TSS1Kb']].mean()
Gasperini_ABC_by_gene_means = Gasperini_ABC_by_gene[['TargetGeneTSS', 'TargetGeneExpression','TargetGenePromoterActivityQuantile','H3K27ac.RPKM.quantile.TSS1Kb','H3K4me3.RPKM.quantile.TSS1Kb', 'H3K27me3.RPKM.quantile.TSS1Kb']].mean()
Gasperini_ABC_by_gene_sums = Gasperini_ABC_by_gene[['ABC.Score']].sum()
Gasperini_ABC_by_gene_sums = Gasperini_ABC_by_gene_sums.rename(columns={'ABC.Score': 'joined.ABC.sum'})
Gasperini_ABC_by_gene_sums['joined.ABC.sum'] = Gasperini_ABC_by_gene_sums['joined.ABC.sum'].fillna(0)
Gasperini_ABC_by_gene_maxs = Gasperini_ABC_by_gene[['ABC.Score']].max()
Gasperini_ABC_by_gene_maxs = Gasperini_ABC_by_gene_maxs.rename(columns={'ABC.Score': 'joined.ABC.max'})
Gasperini_ABC_by_gene_maxs['joined.ABC.max'] = Gasperini_ABC_by_gene_maxs['joined.ABC.max'].fillna(0)
Gasperini_ABC_by_genes = pd.concat([Gasperini_ABC_by_gene_symbol, Gasperini_ABC_by_gene_sig, Gasperini_ABC_by_gene_means, Gasperini_ABC_by_gene_sums, Gasperini_ABC_by_gene_maxs], axis=1)
# by_enhancer
Gasperini_ABC_by_enhancer = Gasperini_atscale_ABC.groupby('enhancerID')
Gasperini_ABC_by_enhancer_symbol = Gasperini_ABC_by_enhancer[['enhancerID', 'chr', 'chrEnhancer',  'startEnhancer',  'endEnhancer', 'TSS.count.near.enhancer', 'mean.contact.from.enhancer', 'max.contact.from.enhancer', 'total.contact.from.enhancer']].first()
Gasperini_ABC_by_enhancer_sig = Gasperini_ABC_by_enhancer[['Significant']].any()
#Gasperini_ABC_by_enhancer_means = Gasperini_ABC_by_enhancer[['normalized_dhs', 'normalized_h3K27ac','normalized_h3K4me3', 'normalized_h3K27me3', 'normalized_h3K4me1']].mean()
Gasperini_ABC_by_enhancer_means = Gasperini_ABC_by_enhancer[['normalized_dhs', 'normalized_h3K27ac','normalized_h3K4me3', 'normalized_h3K27me3']].mean()
Gasperini_ABC_by_enhancer_sums = Gasperini_ABC_by_enhancer[['ABC.Score']].sum()
Gasperini_ABC_by_enhancer_sums = Gasperini_ABC_by_enhancer_sums.rename(columns={'ABC.Score': 'joined.ABC.sum'})
Gasperini_ABC_by_enhancer_sums['joined.ABC.sum'] = Gasperini_ABC_by_enhancer_sums['joined.ABC.sum'].fillna(0)
Gasperini_ABC_by_enhancer_maxs = Gasperini_ABC_by_enhancer[['ABC.Score']].max()
Gasperini_ABC_by_enhancer_maxs = Gasperini_ABC_by_enhancer_maxs.rename(columns={'ABC.Score': 'joined.ABC.max'})
Gasperini_ABC_by_enhancer_maxs['joined.ABC.max'] = Gasperini_ABC_by_enhancer_maxs['joined.ABC.max'].fillna(0)
Gasperini_ABC_by_enhancers = pd.concat([Gasperini_ABC_by_enhancer_symbol, Gasperini_ABC_by_enhancer_sig, Gasperini_ABC_by_enhancer_means, Gasperini_ABC_by_enhancer_sums, Gasperini_ABC_by_enhancer_maxs], axis=1)


# left join
#Gasperini_atscale_ABC = pd.merge(Gasperini_atscale2, ABC, how="left", left_on=["ABC.id"], right_on=["chr:start-end_TargetGene"], suffixes=('', '_y'))
#Gasperini_atscale_ABC.drop(Gasperini_atscale_ABC.filter(regex='_y$').columns, axis=1, inplace=True)
#Gasperini_atscale_ABC.to_csv(data_dir+"Gasperini2019.at_scale.ABC.leftjoin.withdups.txt", sep='\t')
#remove_dups_isNA_not_sig = Gasperini_atscale_ABC.duplicated('ABC.id', keep='first') & Gasperini_atscale_ABC['pValueAdjusted'].isna() & ~Gasperini_atscale_ABC['Significant']  # remove dups that are isna and not_significant first
#Gasperini_atscale_ABC = Gasperini_atscale_ABC[~remove_dups_isNA_not_sig] # remove duplicates leaving first item except when significant
#remove_dups_not_sig = Gasperini_atscale_ABC.duplicated('ABC.id', keep='first') & ~Gasperini_atscale_ABC['Significant']  # remove dups that are not_significant first
#Gasperini_atscale_ABC = Gasperini_atscale_ABC[~remove_dups_not_sig] # remove duplicates leaving first item except when significant
#Gasperini_atscale_ABC = Gasperini_atscale_ABC[~Gasperini_atscale_ABC.duplicated('ABC.id', keep='last')]  # now remove duplicates leaving last (significant)
#dups = Gasperini_atscale_ABC[['chr', 'start', 'end', 'TargetGene']].duplicated()
#if (dups.sum() > 0): 
#  gasperini_atscale_ABC_orig_cols = Gasperini_atscale_ABC.columns.tolist() 
#  grouped = Gasperini_atscale_ABC.groupby(['chr', 'start', 'end', 'TargetGene'], as_index=False)
#  Gasperini_atscale_ABC_sum = grouped.agg(lambda x: x.sum() if x.name=='ABC.Score' else (x.any() if x.name=='Significant' else (x.max() if np.issubdtype(x.dtype,     np.number) else x.iloc[0])))
#  Gasperini_atscale_ABC_sum = Gasperini_atscale_ABC_sum[gasperini_atscale_ABC_orig_cols] 
#else:
#  Gasperini_atscale_ABC_sum = Gasperini_atscale_ABC
#Gasperini_atscale_ABC_sum['Significant'] = Gasperini_atscale_ABC_sum['Significant'].astype('bool')
#Gasperini_ABC_by_gene = Gasperini_atscale_ABC_sum.groupby('GeneSymbol', as_index=False)
#Gasperini_ABC_by_gene_sig = Gasperini_ABC_by_gene['Significant'].any()
#Gasperini_ABC_by_gene_sig= Gasperini_ABC_by_gene_sig.rename(columns={'Significant': 'atleast1Sig'})
#Gasperini_atscale_ABC_sum = pd.merge(Gasperini_atscale_ABC_sum, Gasperini_ABC_by_gene_sig, on=['GeneSymbol'], how='inner')
#Gasperini_atscale_ABC_sum.to_csv(data_dir+"Gasperini2019.at_scale.ABC.leftjoin.txt", sep='\t')

# inner join
Gasperini_atscale_ABC = pd.merge(Gasperini_atscale2, ABC, left_on=["ABC.id"], right_on=["chr:start-end_TargetGene"], suffixes=('', '_y'))
Gasperini_atscale_ABC.drop(Gasperini_atscale_ABC.filter(regex='_y$').columns, axis=1, inplace=True)
Gasperini_atscale_ABC.to_csv(data_dir+"Gasperini2019.at_scale.ABC.innerjoin.withdups.txt", sep='\t')
remove_dups_isNA_not_sig = Gasperini_atscale_ABC.duplicated('ABC.id', keep='first') & Gasperini_atscale_ABC['pValueAdjusted'].isna()   # remove dups that are isna first 
Gasperini_atscale_ABC = Gasperini_atscale_ABC[~remove_dups_isNA_not_sig] # remove duplicates that are isna first
remove_dups_isNA_not_sig = Gasperini_atscale_ABC.duplicated('ABC.id', keep='last') & Gasperini_atscale_ABC['pValueAdjusted'].isna()   # remove dups that are isna first 
Gasperini_atscale_ABC = Gasperini_atscale_ABC[~remove_dups_isNA_not_sig] # remove duplicates that are isna first
remove_dups_not_sig = Gasperini_atscale_ABC.duplicated('ABC.id', keep='first') & ~Gasperini_atscale_ABC['Significant']  # then remove dups that are not_significant
Gasperini_atscale_ABC = Gasperini_atscale_ABC[~remove_dups_not_sig] # remove duplicates leaving first item except when significant
Gasperini_atscale_ABC = Gasperini_atscale_ABC[~Gasperini_atscale_ABC.duplicated('ABC.id', keep='last')]  # now remove duplicates leaving last (significant)
dups = Gasperini_atscale_ABC[['chr', 'start', 'end', 'TargetGene']].duplicated()
if (dups.sum() > 0): 
  gasperini_atscale_ABC_orig_cols = Gasperini_atscale_ABC.columns.tolist() 
  grouped = Gasperini_atscale_ABC.groupby(['chr', 'start', 'end', 'TargetGene'], as_index=False)
  Gasperini_atscale_ABC_sum = grouped.agg(lambda x: x.sum() if x.name=='ABC.Score' else (x.any() if x.name=='Significant' else (x.max() if np.issubdtype(x.dtype,     np.number) else x.iloc[0])))
  Gasperini_atscale_ABC_sum = Gasperini_atscale_ABC_sum[gasperini_atscale_ABC_orig_cols] 
else:
  Gasperini_atscale_ABC_sum = Gasperini_atscale_ABC
Gasperini_atscale_ABC_sum['Significant'] = Gasperini_atscale_ABC_sum['Significant'].astype('bool')
Gasperini_ABC_by_gene = Gasperini_atscale_ABC_sum.groupby('GeneSymbol', as_index=False)
Gasperini_ABC_by_gene_sig = Gasperini_ABC_by_gene['Significant'].any()
Gasperini_ABC_by_gene_sig= Gasperini_ABC_by_gene_sig.rename(columns={'Significant': 'atleast1Sig'})
Gasperini_atscale_ABC_sum = pd.merge(Gasperini_atscale_ABC_sum, Gasperini_ABC_by_gene_sig, on=['GeneSymbol'], how='inner')
Gasperini_atscale_ABC_sum.to_csv(data_dir+"Gasperini2019.at_scale.ABC.innerjoin.txt", sep='\t')



# K562 TF
Gasperini_atscale_ABC = pd.read_csv(data_dir+"Gasperini2019.at_scale.ABC.innerjoin.txt", sep='\t')
enhancer_TF = pd.read_csv(data_dir+"Gasperini2019.enhancer.TF.overlap.bed", sep="\t", names=['chr', 'start', 'end', 'ID', 'chr.TF', 'start.TF', 'end.TF', 'TF', 'score', 'celltype', 'score2'], header=None)  
enhancer_TF['count'] = 1
enhancer_TF_pivot = enhancer_TF.pivot_table(index='ID', columns = 'TF', values='count', aggfunc=np.sum, fill_value=0)  

TSS_TF = pd.read_csv(data_dir+"Gasperini2019.TSS.TF.overlap.bed", sep="\t", names=['chr', 'start', 'end', 'gene', 'dummy', 'strand', 'chr.TF', 'start.TF', 'end.TF', 'TF', 'score', 'celltype', 'score2'], header=None)  
TSS_TF['count'] = 1
TSS_TF_pivot = TSS_TF.pivot_table(index='gene', columns = 'TF', values='count', aggfunc=np.sum, fill_value=0)  

TFcolumns = set().union(list(enhancer_TF_pivot.columns), list(TSS_TF_pivot.columns))  
columns_diff = list(set(TSS_TF_pivot.columns) - set(enhancer_TF_pivot.columns))   
for e in columns_diff:
  enhancer_TF_pivot[e] = 0
if columns_diff:
  enhancer_TF_pivot = enhancer_TF_pivot.reindex(sorted(enhancer_TF_pivot.columns), axis=1)

columns_diff = list(set(enhancer_TF_pivot.columns) - set(TSS_TF_pivot.columns))   
for e in columns_diff:
  TSS_TF_pivot[e] = 0
if columns_diff:
  TSS_TF_pivot = TSS_TF_pivot.reindex(sorted(TSS_TF_pivot.columns), axis=1)

enhancer_TF_pivot = enhancer_TF_pivot.add_suffix('_e')
TSS_TF_pivot = TSS_TF_pivot.add_suffix('_TSS')

enhancer_TF_pivot.to_csv(data_dir+"Gasperini2019.enhancer.TF.txt", sep='\t')
TSS_TF_pivot.to_csv(data_dir+"Gasperini2019.TSS.TF.txt", sep='\t')
TFcolumns = set().union(list(enhancer_TF_pivot.columns), list(TSS_TF_pivot.columns))  
TFcolumnsdf = pd.DataFrame(TFcolumns, columns=['features'])

# join TF to by_gene
Gasperini_ABC_by_genes = pd.merge(Gasperini_ABC_by_genes, TSS_TF_pivot, how="left", left_on=["GeneSymbol"], right_on=["gene"], suffixes=('', '_y'))
Gasperini_ABC_by_genes.drop(Gasperini_ABC_by_genes.filter(regex='_y$').columns, axis=1, inplace=True)
Gasperini_ABC_by_genes.loc[:,list(TSS_TF_pivot.columns)] = Gasperini_ABC_by_genes.loc[:,list(TSS_TF_pivot.columns)].fillna(0)
Gasperini_ABC_by_genes.to_csv(data_dir+"Gasperini2019.bygene.ABC.TF.txt", sep='\t')

# join TF to by_enhancer
Gasperini_ABC_by_enhancers.reset_index(drop=True, inplace=True)
Gasperini_ABC_by_enhancers = pd.merge(Gasperini_ABC_by_enhancers, enhancer_TF_pivot, how="left", left_on=["enhancerID"], right_on=["ID"], suffixes=('', '_y'))
Gasperini_ABC_by_enhancers.drop(Gasperini_ABC_by_enhancers.filter(regex='_y$').columns, axis=1, inplace=True)
Gasperini_ABC_by_enhancers.loc[:,list(enhancer_TF_pivot.columns)] = Gasperini_ABC_by_enhancers.loc[:,list(enhancer_TF_pivot.columns)].fillna(0)
Gasperini_ABC_by_enhancers.to_csv(data_dir+"Gasperini2019.byenhancer.ABC.TF.txt", sep='\t')


# join TF to ABC
Gasperini_atscale_ABC = pd.merge(Gasperini_atscale_ABC, enhancer_TF_pivot, how="left", left_on=["enhancerID"], right_on=["ID"], suffixes=('', '_y'))
Gasperini_atscale_ABC.drop(Gasperini_atscale_ABC.filter(regex='_y$').columns, axis=1, inplace=True)
Gasperini_atscale_ABC = pd.merge(Gasperini_atscale_ABC, TSS_TF_pivot, how="left", left_on=["GeneSymbol"], right_on=["gene"], suffixes=('', '_y'))
Gasperini_atscale_ABC.drop(Gasperini_atscale_ABC.filter(regex='_y$').columns, axis=1, inplace=True)
Gasperini_atscale_ABC.loc[:,list(TFcolumns)] = Gasperini_atscale_ABC.loc[:,list(TFcolumns)].fillna(0)
Gasperini_atscale_ABC.to_csv(data_dir+"Gasperini2019.at_scale.ABC.TF.txt", sep='\t')


# erole
erole = pd.read_csv(data_dir+"Gasperini2019.at_scale.eroles.txt", sep="\t")
Gasperini_atscale_ABC = pd.merge(Gasperini_atscale_ABC, erole, left_on=["name"], right_on=["name_x"])
Gasperini_atscale_ABC.drop(Gasperini_atscale_ABC.filter(regex='_x$').columns, axis=1, inplace=True)
Gasperini_atscale_ABC = Gasperini_atscale_ABC.loc[:,~Gasperini_atscale_ABC.columns.str.match("Unnamed")]
Gasperini_atscale_ABC.to_csv(data_dir+"Gasperini2019.at_scale.ABC.TF.erole.txt", sep='\t')



