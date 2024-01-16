import pandas as pd
import numpy as np

# ABC
data_dir = "data/Shraivogel/"
Exp_enhancer = pd.read_csv(data_dir+"enhancer.ABC.overlap.bed", sep='\t')
Exp_TSS = pd.read_csv(data_dir+"TSS.ABC.overlap.bed", sep='\t')
Exp_crispr = pd.read_csv(data_dir+"Shraivogel.input.txt", sep="\t")


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
ABC['std.contact.from.enhancer'] = ABC_by_gene[['hic_contact']].transform('std')
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


#new = Exp_enhancer["G.id"].str.split("|", n=1, expand=True)
#Exp_enhancer["G.class"]=new[0]
#Exp_enhancer["G.id"]=new[1]
new = Exp_enhancer["ABC.id"].str.split("|", n=1, expand=True)
Exp_enhancer["ABC.class"]=new[0]
Exp_enhancer["ABC.id"]=new[1]
Exp_enhancer = Exp_enhancer[["G.id", "ABC.id"]]
#new = Exp_crispr["Element name"].str.split("|", n=1, expand=True)
#Exp_crispr["enhancerID"]= new[1]
Exp_crispr["enhancerID"]= Exp_crispr['perturbation']

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


Exp_enhancer = pd.merge(Exp_enhancer, ABC_enhancer, left_on=["ABC.id"], right_on=["ABC.id"])
Exp_enhancer = pd.merge(Exp_enhancer, ABC_enhancer, left_on=["ABC.id"], right_on=["ABC.id"], suffixes=('', '_y'))
Exp_enhancer.drop(Exp_enhancer.filter(regex='_y$').columns, axis=1, inplace=True)
Exp_enhancer = pd.merge(Exp_enhancer, ABC_enhancer_H3K4me3, left_on=["ABC.id"], right_on=["ABC.id"], suffixes=('', '_y'))
Exp_enhancer.drop(Exp_enhancer.filter(regex='_y$').columns, axis=1, inplace=True)
Exp_enhancer = pd.merge(Exp_enhancer, ABC_enhancer_H3K27me3, left_on=["ABC.id"], right_on=["ABC.id"], suffixes=('', '_y'))
Exp_enhancer.drop(Exp_enhancer.filter(regex='_y$').columns, axis=1, inplace=True)
Exp_enhancer = Exp_enhancer.drop_duplicates()


Exp_crispr.to_csv(data_dir+"Exp_crispr.txt", sep='\t')
Exp_enhancer.to_csv(data_dir+"Exp_enhancer.txt", sep='\t')
Exp_crispr2 = pd.merge(Exp_crispr, Exp_enhancer, left_on=["enhancerID"], right_on=["G.id"], suffixes=('', '_y'))
Exp_crispr2.drop(Exp_crispr2.filter(regex='_y$').columns, axis=1, inplace=True)
Exp_crispr2["ABC.id"] = Exp_crispr2["ABC.id"] + "_" + Exp_crispr2["Gene"]
Exp_crispr2.to_csv(data_dir+"CRISPR2.txt", sep='\t')
# significant.negative
#Exp_crispr2.rename(columns={'Significant':'Sig.pos.neg'}, inplace=True)
#Exp_crispr2['Significant'] = Exp_crispr2['Sig.pos.neg'] & (Exp_crispr2['Fraction change in gene expr'] < 0)
#Exp_crispr2 = Exp_crispr2.loc[:,~Exp_crispr2.columns.str.match("Unnamed")]
#Exp_crispr2.reset_index(drop=True)
#Exp_crispr2.to_csv(data_dir+"CRISPR2.txt", sep='\t')



ABC_gene = pd.read_csv(data_dir+"GeneList.txt", sep="\t")
ABC_gene = ABC_gene.loc[:,['name','H3K27ac.RPKM.quantile.TSS1Kb']]
ABC_gene_H3K4me3 = pd.read_csv(data_dir+"GeneList.H3K4me3.txt", sep="\t")
ABC_gene_H3K4me3.rename({'H3K27ac.RPKM.quantile.TSS1Kb': 'H3K4me3.RPKM.quantile.TSS1Kb'}, axis=1, inplace=True)
ABC_gene_H3K4me3 = ABC_gene_H3K4me3.loc[:,['name','H3K4me3.RPKM.quantile.TSS1Kb']]
ABC_gene_H3K27me3 = pd.read_csv(data_dir+"GeneList.H3K27me3.txt", sep="\t")
ABC_gene_H3K27me3.rename({'H3K27ac.RPKM.quantile.TSS1Kb': 'H3K27me3.RPKM.quantile.TSS1Kb'}, axis=1, inplace=True)
ABC_gene_H3K27me3 = ABC_gene_H3K27me3.loc[:,['name','H3K27me3.RPKM.quantile.TSS1Kb']]
ABC = pd.merge(ABC, ABC_gene, left_on=["TargetGene"], right_on=["name"], suffixes=('', '_y'))
ABC.drop(ABC.filter(regex='_y$').columns, axis=1, inplace=True)
ABC = pd.merge(ABC, ABC_gene_H3K4me3, left_on=["TargetGene"], right_on=["name"], suffixes=('', '_y'))
ABC.drop(ABC.filter(regex='_y$').columns, axis=1, inplace=True)
ABC = pd.merge(ABC, ABC_gene_H3K27me3, left_on=["TargetGene"], right_on=["name"], suffixes=('', '_y'))
ABC.drop(ABC.filter(regex='_y$').columns, axis=1, inplace=True)



Exp_crispr2.to_csv(data_dir+"Exp_crispr2.txt", sep='\t')
ABC.to_csv(data_dir+"ABC.txt", sep='\t')

#raise SystemExit()
Exp_crispr_ABC = pd.merge(Exp_crispr2, ABC, how="left", left_on=["ABC.id"], right_on=["chr:start-end_TargetGene"], suffixes=('', '_y'))
Exp_crispr_ABC.drop(Exp_crispr_ABC.filter(regex='_y$').columns, axis=1, inplace=True)
Exp_crispr_ABC.to_csv(data_dir+"CRISPR.ABC.leftjoin.withdups.txt", sep='\t')
#grouped = Exp_crispr_ABC.groupby(['G.id','ABC.id'], as_index=False)
grouped = Exp_crispr_ABC.groupby(['chr', 'start', 'end', 'Gene'], as_index=False)
Exp_crispr_ABC_sum = grouped.agg(lambda x: x.sum() if x.name=='ABC.Score' else (x.any() if x.name=='Significant' else (x.max() if np.issubdtype(x.dtype, np.number) else x.iloc[0]))) 
Exp_crispr_ABC_sum['Significant'] = Exp_crispr_ABC_sum['Significant'].astype('bool')
Exp_crispr_ABC_sum.to_csv(data_dir+"CRISPR.ABC.sum.leftjoin.txt", sep='\t')


Exp_crispr_ABC = pd.merge(Exp_crispr2, ABC, left_on=["ABC.id"], right_on=["chr:start-end_TargetGene"], suffixes=('', '_y'))
Exp_crispr_ABC.drop(Exp_crispr_ABC.filter(regex='_y$').columns, axis=1, inplace=True)
Exp_crispr_ABC.to_csv(data_dir+"CRISPR.ABC.innerjoin.withdups.txt", sep='\t')
#grouped = Exp_crispr_ABC.groupby(['G.id','ABC.id'], as_index=False)
grouped = Exp_crispr_ABC.groupby(['chr', 'start', 'end', 'Gene'], as_index=False)
Exp_crispr_ABC_sum = grouped.agg(lambda x: x.sum() if x.name=='ABC.Score' else (x.any() if x.name=='Significant' else (x.max() if np.issubdtype(x.dtype, np.number) else x.iloc[0]))) 
Exp_crispr_ABC_sum['Significant'] = Exp_crispr_ABC_sum['Significant'].astype('bool')
Exp_crispr_ABC_sum.to_csv(data_dir+"CRISPR.ABC.sum.innerjoin.txt", sep='\t')



#at least one significant
Exp_crispr_ABC = pd.read_csv(data_dir+"CRISPR.ABC.sum.innerjoin.txt", sep='\t')
Exp_ABC_by_gene = Exp_crispr_ABC.groupby('Gene', as_index=False)
Exp_ABC_by_gene_sig = Exp_ABC_by_gene['Significant'].any()
Exp_ABC_by_gene_sig= Exp_ABC_by_gene_sig.rename(columns={'Significant': 'atleast1Sig'})
Exp_crispr_ABC = pd.merge(Exp_crispr_ABC, Exp_ABC_by_gene_sig, on=['Gene'], how='inner')
Exp_crispr_ABC.to_csv(data_dir+"CRISPR.ABC.sum.innerjoin.txt", sep='\t')


# K562 TF
Exp_crispr_ABC = pd.read_csv(data_dir+"CRISPR.ABC.sum.innerjoin.txt", sep='\t')
enhancer_TF = pd.read_csv(data_dir+"enhancer.TF.overlap.bed", sep="\t", names=['chr', 'start', 'end', 'ID', 'chr.TF', 'start.TF', 'end.TF', 'TF', 'score', 'celltype', 'score2'], header=None)  
enhancer_TF['count'] = 1
enhancer_TF_pivot = enhancer_TF.pivot_table(index='ID', columns = 'TF', values='count', aggfunc=np.sum, fill_value=0)  

TSS_TF = pd.read_csv(data_dir+"TSS.TF.overlap.bed", sep="\t", names=['chr', 'start', 'end', 'gene', 'dummy', 'strand', 'chr.TF', 'start.TF', 'end.TF', 'TF', 'score', 'celltype', 'score2'], header=None)  
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

enhancer_TF_pivot.to_csv(data_dir+"enhancer.TF.txt", sep='\t')
TSS_TF_pivot.to_csv(data_dir+"TSS.TF.txt", sep='\t')


# join TF to ABC
start = Exp_crispr_ABC.shape[1]
Exp_crispr_ABC = pd.merge(Exp_crispr_ABC, enhancer_TF_pivot, how="left", left_on=["enhancerID"], right_on=["ID"])
Exp_crispr_ABC.drop(Exp_crispr_ABC.filter(regex='_y$').columns, axis=1, inplace=True)
Exp_crispr_ABC = pd.merge(Exp_crispr_ABC, TSS_TF_pivot, how="left", left_on=["Gene"], right_on=["gene"])
Exp_crispr_ABC.drop(Exp_crispr_ABC.filter(regex='_y$').columns, axis=1, inplace=True)
Exp_crispr_ABC.to_csv(data_dir+"CRISPR.ABC.TF.txt", sep='\t')


#e_TF = Exp_crispr_ABC.iloc[:,start:(start+len(TFcolumns))]
#TSS_TF = Exp_crispr_ABC.iloc[:,(start+len(TFcolumns)):(start+len(TFcolumns)+len(TFcolumns))]
#cobinding = pd.DataFrame(np.logical_and(e_TF, np.asarray(TSS_TF))).astype(int)  
#cobinding.columns = TFcolumns
#cobinding = cobinding.add_suffix('_co')
#Exp_crispr_ABC = pd.concat([Exp_crispr_ABC, cobinding], axis=1)


# erole
#erole = pd.read_csv(data_dir+"CRISPR.eroles.txt", sep="\t")
#Exp_crispr_ABC = pd.merge(Exp_crispr_ABC, erole, left_on=["name"], right_on=["name_x"])
#Exp_crispr_ABC.drop(Exp_crispr_ABC.filter(regex='_x$').columns, axis=1, inplace=True)
#Exp_crispr_ABC.to_csv(data_dir+"CRISPR.ABC.TF.cobinding.erole.txt", sep='\t')


