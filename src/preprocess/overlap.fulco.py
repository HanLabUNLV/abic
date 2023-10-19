import pandas as pd
import numpy as np

# ABC
data_dir = "data/Fulco/"
Fulco_enhancer = pd.read_csv(data_dir+"Fulco2019.enhancer.ABC.overlap.bed", sep='\t')
Fulco_TSS = pd.read_csv(data_dir+"Fulco2019.TSS.ABC.overlap.bed", sep='\t')
Fulco_crispr = pd.read_csv(data_dir+"Fulco2019.STable6a.tab", sep="\t")
ABC = pd.read_csv(data_dir+"ABC.EnhancerPredictionsAllPutative.txt", sep='\t')

new = Fulco_enhancer["G.id"].str.split("|", n=1, expand=True)
Fulco_enhancer["G.class"]=new[0]
Fulco_enhancer["G.id"]=new[1]
new = Fulco_enhancer["ABC.id"].str.split("|", n=1, expand=True)
Fulco_enhancer["ABC.class"]=new[0]
Fulco_enhancer["ABC.id"]=new[1]
Fulco_enhancer = Fulco_enhancer[["G.id", "ABC.id"]]
new = Fulco_crispr["Element name"].str.split("|", n=1, expand=True)
Fulco_crispr["enhancerID"]= new[1]

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


Fulco_enhancer = pd.merge(Fulco_enhancer, ABC_enhancer, left_on=["ABC.id"], right_on=["ABC.id"])
Fulco_enhancer = pd.merge(Fulco_enhancer, ABC_enhancer, left_on=["ABC.id"], right_on=["ABC.id"], suffixes=('', '_y'))
Fulco_enhancer.drop(Fulco_enhancer.filter(regex='_y$').columns, axis=1, inplace=True)
Fulco_enhancer = pd.merge(Fulco_enhancer, ABC_enhancer_H3K4me3, left_on=["ABC.id"], right_on=["ABC.id"], suffixes=('', '_y'))
Fulco_enhancer.drop(Fulco_enhancer.filter(regex='_y$').columns, axis=1, inplace=True)
Fulco_enhancer = pd.merge(Fulco_enhancer, ABC_enhancer_H3K27me3, left_on=["ABC.id"], right_on=["ABC.id"], suffixes=('', '_y'))
Fulco_enhancer.drop(Fulco_enhancer.filter(regex='_y$').columns, axis=1, inplace=True)
Fulco_enhancer = Fulco_enhancer.drop_duplicates()


Fulco_crispr.to_csv(data_dir+"Fulco_crispr.txt", sep='\t')
Fulco_enhancer.to_csv(data_dir+"Fulco_enhancer.txt", sep='\t')
Fulco_crispr2 = pd.merge(Fulco_crispr, Fulco_enhancer, left_on=["enhancerID"], right_on=["G.id"], suffixes=('', '_y'))
Fulco_crispr2.drop(Fulco_crispr2.filter(regex='_y$').columns, axis=1, inplace=True)
Fulco_crispr2["ABC.id"] = Fulco_crispr2["ABC.id"] + "_" + Fulco_crispr2["Gene"]
Fulco_crispr2.to_csv(data_dir+"Fulco2019.CRISPR2.txt", sep='\t')

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



Fulco_crispr2.to_csv(data_dir+"Fulco_crispr2.txt", sep='\t')
ABC.to_csv(data_dir+"ABC.txt", sep='\t')

#raise SystemExit()
Fulco_crispr_ABC = pd.merge(Fulco_crispr2, ABC, how="left", left_on=["ABC.id"], right_on=["chr:start-end_TargetGene"], suffixes=('', '_y'))
Fulco_crispr_ABC.drop(Fulco_crispr_ABC.filter(regex='_y$').columns, axis=1, inplace=True)
Fulco_crispr_ABC.to_csv(data_dir+"Fulco2019.CRISPR.ABC.leftjoin.withdups.txt", sep='\t')
grouped = Fulco_crispr_ABC.groupby(['G.id','ABC.id'], as_index=False)
Fulco_crispr_ABC_sum = grouped.agg(lambda x: x.sum() if np.issubdtype(x.dtype, np.number) else x.iloc[0])
Fulco_crispr_ABC_sum['Significant'] = Fulco_crispr_ABC_sum['Significant'].astype('bool')
Fulco_crispr_ABC_sum.to_csv(data_dir+"Fulco2019.CRISPR.ABC.sum.leftjoin.txt", sep='\t')
Fulco_crispr_ABC_max = grouped.agg(lambda x: x.max() if np.issubdtype(x.dtype, np.number) else x.iloc[0])
Fulco_crispr_ABC_max['Significant'] = Fulco_crispr_ABC_max['Significant'].astype('bool')
Fulco_crispr_ABC_max.to_csv(data_dir+"Fulco2019.CRISPR.ABC.max.leftjoin.txt", sep='\t')


Fulco_crispr_ABC = pd.merge(Fulco_crispr2, ABC, left_on=["ABC.id"], right_on=["chr:start-end_TargetGene"], suffixes=('', '_y'))
Fulco_crispr_ABC.drop(Fulco_crispr_ABC.filter(regex='_y$').columns, axis=1, inplace=True)
Fulco_crispr_ABC.to_csv(data_dir+"Fulco2019.CRISPR.ABC.innerjoin.withdups.txt", sep='\t')
grouped = Fulco_crispr_ABC.groupby(['G.id','ABC.id'], as_index=False)
Fulco_crispr_ABC_sum = grouped.agg(lambda x: x.sum() if np.issubdtype(x.dtype, np.number) else x.iloc[0])
Fulco_crispr_ABC_sum['Significant'] = Fulco_crispr_ABC_sum['Significant'].astype('bool')
Fulco_crispr_ABC_sum.to_csv(data_dir+"Fulco2019.CRISPR.ABC.sum.innerjoin.txt", sep='\t')
Fulco_crispr_ABC_max = grouped.agg(lambda x: x.max() if np.issubdtype(x.dtype, np.number) else x.iloc[0])
Fulco_crispr_ABC_max['Significant'] = Fulco_crispr_ABC_max['Significant'].astype('bool')
Fulco_crispr_ABC_max.to_csv(data_dir+"Fulco2019.CRISPR.ABC.max.innerjoin.txt", sep='\t')



# K562 TF
Fulco_crispr_ABC = pd.read_csv(data_dir+"Fulco2019.CRISPR.ABC.sum.innerjoin.txt", sep='\t')
enhancer_TF = pd.read_csv(data_dir+"Fulco2019.enhancer.TF.overlap.bed", sep="\t", names=['chr', 'start', 'end', 'ID', 'chr.TF', 'start.TF', 'end.TF', 'TF', 'score', 'celltype', 'score2'], header=None)  
enhancer_TF['count'] = 1
enhancer_TF_pivot = enhancer_TF.pivot_table(index='ID', columns = 'TF', values='count', aggfunc=np.sum, fill_value=0)  

TSS_TF = pd.read_csv(data_dir+"Fulco2019.TSS.TF.overlap.bed", sep="\t", names=['chr', 'start', 'end', 'gene', 'dummy', 'strand', 'chr.TF', 'start.TF', 'end.TF', 'TF', 'score', 'celltype', 'score2'], header=None)  
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

enhancer_TF_pivot.to_csv(data_dir+"Fulco2019.enhancer.TF.txt", sep='\t')
TSS_TF_pivot.to_csv(data_dir+"Fulco2019.TSS.TF.txt", sep='\t')


# join TF to ABC
start = Fulco_crispr_ABC.shape[1]
Fulco_crispr_ABC = pd.merge(Fulco_crispr_ABC, enhancer_TF_pivot, how="left", left_on=["enhancerID"], right_on=["ID"])
Fulco_crispr_ABC.drop(Fulco_crispr_ABC.filter(regex='_y$').columns, axis=1, inplace=True)
Fulco_crispr_ABC = pd.merge(Fulco_crispr_ABC, TSS_TF_pivot, how="left", left_on=["Gene"], right_on=["gene"])
Fulco_crispr_ABC.drop(Fulco_crispr_ABC.filter(regex='_y$').columns, axis=1, inplace=True)
Fulco_crispr_ABC.to_csv(data_dir+"Fulco2019.CRISPR.ABC.TF.txt", sep='\t')
e_TF = Fulco_crispr_ABC.iloc[:,start:(start+len(TFcolumns))]
TSS_TF = Fulco_crispr_ABC.iloc[:,(start+len(TFcolumns)):(start+len(TFcolumns)+len(TFcolumns))]
cobinding = pd.DataFrame(np.logical_and(e_TF, np.asarray(TSS_TF))).astype(int)  
cobinding.columns = TFcolumns
cobinding = cobinding.add_suffix('_co')
Fulco_crispr_ABC = pd.concat([Fulco_crispr_ABC, cobinding], axis=1)
Fulco_crispr_ABC.to_csv(data_dir+"Fulco2019.CRISPR.ABC.TF.cobinding.allsig.txt", sep='\t')


# significant.negative
Fulco_crispr_ABC.rename(columns={'Significant':'Sig.pos.neg'}, inplace=True)
Fulco_crispr_ABC['Significant'] = Fulco_crispr_ABC['Sig.pos.neg'] & (Fulco_crispr_ABC['Fraction change in gene expr'] < 0)
Fulco_crispr_ABC.to_csv(data_dir+"Fulco2019.CRISPR.ABC.TF.cobinding.txt", sep='\t')

# erole
#erole = pd.read_csv(data_dir+"Fulco2019.CRISPR.eroles.txt", sep="\t")
#Fulco_crispr_ABC = pd.merge(Fulco_crispr_ABC, erole, left_on=["name"], right_on=["name_x"])
#Fulco_crispr_ABC.drop(Fulco_crispr_ABC.filter(regex='_x$').columns, axis=1, inplace=True)
#Fulco_crispr_ABC.to_csv(data_dir+"Fulco2019.CRISPR.ABC.TF.cobinding.erole.txt", sep='\t')


