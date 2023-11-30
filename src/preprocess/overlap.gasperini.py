import pandas as pd
import numpy as np

# ABC
data_dir = "data/Gasperini/"
Gasperini_enhancer = pd.read_csv(data_dir+"Gasperini2019.enhancer.ABC.overlap.bed", sep='\t')
Gasperini_TSS = pd.read_csv(data_dir+"Gasperini2019.TSS.ABC.overlap.bed", sep='\t')
Gasperini_atscale = pd.read_csv(data_dir+"Gasperini2019.at_scale_screen.cand_enhancer_x_exprsd_genes.200503.csv")
ABC = pd.read_csv(data_dir+"ABC.EnhancerPredictionsAllPutative.txt", sep='\t')
ABC_by_gene = ABC.groupby('TargetGene')
ABC['Enhancer.count'] = ABC_by_gene[['ABC.Score']].transform('count')
ABC['ABC.Score.mean'] = ABC_by_gene[['ABC.Score']].transform('mean')
ABC['ABC.Score.Numerator.sum'] = ABC_by_gene[['ABC.Score.Numerator']].transform('sum')
ABC['ABC.Score.rest'] = ABC['ABC.Score.Numerator.sum'] - ABC['ABC.Score.Numerator']


new = Gasperini_atscale["name"].str.split(":", n=1, expand=True)
Gasperini_atscale["enhancerID"]= new[0]
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

Gasperini_enhancer = pd.merge(Gasperini_enhancer, ABC_enhancer, left_on=["ABC.id"], right_on=["ABC.id"], suffixes=('', '_y'))
Gasperini_enhancer.drop(Gasperini_enhancer.filter(regex='_y$').columns, axis=1, inplace=True)
Gasperini_enhancer = pd.merge(Gasperini_enhancer, ABC_enhancer_H3K4me3, left_on=["ABC.id"], right_on=["ABC.id"], suffixes=('', '_y'))
Gasperini_enhancer.drop(Gasperini_enhancer.filter(regex='_y$').columns, axis=1, inplace=True)
Gasperini_enhancer = pd.merge(Gasperini_enhancer, ABC_enhancer_H3K27me3, left_on=["ABC.id"], right_on=["ABC.id"], suffixes=('', '_y'))
Gasperini_enhancer.drop(Gasperini_enhancer.filter(regex='_y$').columns, axis=1, inplace=True)

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
ABC = pd.merge(ABC, ABC_gene, left_on=["TargetGene"], right_on=["name"], suffixes=('', '_y'))
ABC.drop(ABC.filter(regex='_y$').columns, axis=1, inplace=True)
ABC = pd.merge(ABC, ABC_gene_H3K4me3, left_on=["TargetGene"], right_on=["name"], suffixes=('', '_y'))
ABC.drop(ABC.filter(regex='_y$').columns, axis=1, inplace=True)
ABC = pd.merge(ABC, ABC_gene_H3K27me3, left_on=["TargetGene"], right_on=["name"], suffixes=('', '_y'))
ABC.drop(ABC.filter(regex='_y$').columns, axis=1, inplace=True)

Gasperini_atscale_ABC = pd.merge(Gasperini_atscale2, ABC, how="left", left_on=["ABC.id"], right_on=["chr:start-end_TargetGene"], suffixes=('', '_y'))
Gasperini_atscale_ABC.drop(Gasperini_atscale_ABC.filter(regex='_y$').columns, axis=1, inplace=True)
remove_dups_not_sig = Gasperini_atscale_ABC.duplicated('ABC.id', keep='first') & ~Gasperini_atscale_ABC['Significant']
Gasperini_atscale_ABC = Gasperini_atscale_ABC[~remove_dups_not_sig] # remove duplicates leaving first item except when significant
Gasperini_atscale_ABC = Gasperini_atscale_ABC[~Gasperini_atscale_ABC.duplicated('ABC.id', keep='last')]  # now remove duplicates leaving last (significant)
Gasperini_atscale_ABC.to_csv(data_dir+"Gasperini2019.at_scale.ABC.leftjoin.txt", sep='\t')
Gasperini_atscale_ABC = pd.merge(Gasperini_atscale2, ABC, left_on=["ABC.id"], right_on=["chr:start-end_TargetGene"], suffixes=('', '_y'))
Gasperini_atscale_ABC.drop(Gasperini_atscale_ABC.filter(regex='_y$').columns, axis=1, inplace=True)
remove_dups_not_sig = Gasperini_atscale_ABC.duplicated('ABC.id', keep='first') & ~Gasperini_atscale_ABC['Significant']
Gasperini_atscale_ABC = Gasperini_atscale_ABC[~remove_dups_not_sig] # remove duplicates leaving first item except when significant
Gasperini_atscale_ABC = Gasperini_atscale_ABC[~Gasperini_atscale_ABC.duplicated('ABC.id', keep='last')]  # now remove duplicates leaving last (significant)
Gasperini_atscale_ABC.to_csv(data_dir+"Gasperini2019.at_scale.ABC.txt", sep='\t')


# K562 TF
Gasperini_atscale_ABC = pd.read_csv(data_dir+"Gasperini2019.at_scale.ABC.txt", sep='\t')
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


# join TF to ABC
start = Gasperini_atscale_ABC.shape[1]
Gasperini_atscale_ABC = pd.merge(Gasperini_atscale_ABC, enhancer_TF_pivot, how="left", left_on=["enhancerID"], right_on=["ID"], suffixes=('', '_y'))
Gasperini_atscale_ABC.drop(Gasperini_atscale_ABC.filter(regex='_y$').columns, axis=1, inplace=True)
Gasperini_atscale_ABC = pd.merge(Gasperini_atscale_ABC, TSS_TF_pivot, how="left", left_on=["GeneSymbol"], right_on=["gene"], suffixes=('', '_y'))
Gasperini_atscale_ABC.drop(Gasperini_atscale_ABC.filter(regex='_y$').columns, axis=1, inplace=True)
Gasperini_atscale_ABC.to_csv(data_dir+"Gasperini2019.at_scale.ABC.TF.txt", sep='\t')
e_TF = Gasperini_atscale_ABC.iloc[:,start:(start+len(TFcolumns))]
TSS_TF = Gasperini_atscale_ABC.iloc[:,(start+len(TFcolumns)):(start+len(TFcolumns)+len(TFcolumns))]
cobinding = pd.DataFrame(np.logical_and(e_TF, np.asarray(TSS_TF))).astype(int)  # ufunc(df1, np.asarray(df2)
cobinding.columns = TFcolumns
cobinding = cobinding.add_suffix('_co')
Gasperini_atscale_ABC = pd.concat([Gasperini_atscale_ABC, cobinding], axis=1)
Gasperini_atscale_ABC.to_csv(data_dir+"Gasperini2019.at_scale.ABC.TF.cobinding.txt", sep='\t')

# erole
erole = pd.read_csv(data_dir+"Gasperini2019.at_scale.eroles.txt", sep="\t")
Gasperini_atscale_ABC = pd.merge(Gasperini_atscale_ABC, erole, left_on=["name"], right_on=["name_x"])
Gasperini_atscale_ABC.drop(Gasperini_atscale_ABC.filter(regex='_x$').columns, axis=1, inplace=True)
Gasperini_atscale_ABC.to_csv(data_dir+"Gasperini2019.at_scale.ABC.TF.cobinding.erole.txt", sep='\t')


