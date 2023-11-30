import pandas as pd
import numpy as np
import joblib
from sklearn.decomposition import NMF
import seaborn as sns
import matplotlib.pyplot as plt


def DR_NMF_features(TFmatrix,outdir, study_name_prefix):
    n_components = 12 
    init = "nndsvd"
    nmf_model = NMF(
        solver='cd',
        n_components=n_components,
        random_state=1,
        init=init,
        beta_loss="frobenius",
        alpha_W=0.005,
        alpha_H=0.00005,
        l1_ratio=0.7,
        max_iter=500
    )
    nmf_model.fit(TFmatrix)
    joblib.dump(nmf_model, outdir+'/'+study_name_prefix+'.gz')
    W = nmf_model.transform(TFmatrix)
    Wdf = pd.DataFrame(W, index=TFmatrix.index, columns =  ["TF_NMF_" + str(i+1) for i in range(n_components)])
    Wdf.to_csv(outdir+'/'+study_name_prefix+'.TF.W.txt', index=False, sep='\t')
    H = nmf_model.components_
    Hdf = pd.DataFrame(H, columns=TFmatrix.columns)
    Hdf.to_csv(outdir+'/'+study_name_prefix+'.TF.H.txt', index=False, sep='\t')
    # heatmap for NMF features.
    sns.set(font_scale=2)
    fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(400, 100))
    hm = sns.heatmap(data = Hdf)
    plt.title("Heatmap NMF H")
    plt.savefig(outdir+'/'+study_name_prefix+'.heatmap.NMF.H.pdf')
    plt.close(fig)
    plt.show()
    return (Wdf)









# ABC
data_dir = "data/Gasperini/"
Gasperini_enhancer = pd.read_csv(data_dir+"Gasperini2019.enhancer.ABC.overlap.bed", sep='\t')
Gasperini_TSS = pd.read_csv(data_dir+"Gasperini2019.TSS.ABC.overlap.bed", sep='\t')
Gasperini_atscale = pd.read_csv(data_dir+"Gasperini2019.at_scale_screen.cand_enhancer_x_exprsd_genes.200503.csv")
Gasperini_atscale['pValueAdjusted'] = Gasperini_atscale['pValueAdjusted'].fillna(1)
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

# gene prediction training data 
Gasperini_atscale_ABC = pd.merge(Gasperini_atscale2, ABC, left_on=["ABC.id"], right_on=["chr:start-end_TargetGene"], suffixes=('', '_y'))
Gasperini_atscale_ABC.drop(Gasperini_atscale_ABC.filter(regex='_y$').columns, axis=1, inplace=True)
Gasperini_atscale_ABC.to_csv(data_dir+"Gasperini2019.at_scale.ABC.innerjoin.txt", sep='\t')
Gasperini_ABC_by_gene = Gasperini_atscale_ABC.groupby('TargetGene')
Gasperini_ABC_by_gene_symbol = Gasperini_ABC_by_gene[['GeneSymbol', 'chr', 'Enhancer.count', 'ABC.Score.mean', 'ABC.Score.Numerator.sum']].first()
Gasperini_ABC_by_gene_sig = Gasperini_ABC_by_gene[['Significant', 'TargetGeneIsExpressed']].any()
Gasperini_ABC_by_gene_means = Gasperini_ABC_by_gene[['TargetGeneTSS', 'TargetGeneExpression','TargetGenePromoterActivityQuantile','H3K27ac.RPKM.quantile.TSS1Kb','H3K4me3.RPKM.quantile.TSS1Kb', 'H3K27me3.RPKM.quantile.TSS1Kb']].mean()
Gasperini_ABC_by_gene_sums = Gasperini_ABC_by_gene[['ABC.Score']].sum()
Gasperini_ABC_by_gene_sums = Gasperini_ABC_by_gene_sums.rename(columns={'ABC.Score': 'joined.ABC.sum'})
Gasperini_ABC_by_gene_maxs = Gasperini_ABC_by_gene[['ABC.Score']].max()
Gasperini_ABC_by_gene_maxs = Gasperini_ABC_by_gene_maxs.rename(columns={'ABC.Score': 'joined.ABC.max'})
Gasperini_ABC_by_genes = pd.concat([Gasperini_ABC_by_gene_symbol, Gasperini_ABC_by_gene_sig, Gasperini_ABC_by_gene_means, Gasperini_ABC_by_gene_sums, Gasperini_ABC_by_gene_maxs], axis=1)



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
Gasperini_ABC_by_gene = Gasperini_atscale_ABC.groupby('TargetGene', as_index=False)
Gasperini_ABC_by_gene_sig = Gasperini_ABC_by_gene['Significant'].any()
Gasperini_ABC_by_gene_sig= Gasperini_ABC_by_gene_sig.rename(columns={'Significant': 'atleast1Sig'})
Gasperini_atscale_ABC = pd.merge(Gasperini_atscale_ABC, Gasperini_ABC_by_gene_sig, on=['TargetGene'], how='inner')
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
TFcolumns = set().union(list(enhancer_TF_pivot.columns), list(TSS_TF_pivot.columns))  
TFcolumnsdf = pd.DataFrame(TFcolumns, columns=['features'])

# join TF to bygene
Gasperini_ABC_by_genes = pd.merge(Gasperini_ABC_by_genes, TSS_TF_pivot, how="left", left_on=["GeneSymbol"], right_on=["gene"], suffixes=('', '_y'))
Gasperini_ABC_by_genes.drop(Gasperini_ABC_by_genes.filter(regex='_y$').columns, axis=1, inplace=True)
Gasperini_ABC_by_genes.loc[:,list(TSS_TF_pivot.columns)] = Gasperini_ABC_by_genes.loc[:,list(TSS_TF_pivot.columns)].fillna(0)
Gasperini_ABC_by_genes.to_csv(data_dir+"Gasperini2019.bygene.ABC.TF.txt", sep='\t')

# join TF to ABC
Gasperini_atscale_ABC = pd.merge(Gasperini_atscale_ABC, enhancer_TF_pivot, how="left", left_on=["enhancerID"], right_on=["ID"], suffixes=('', '_y'))
Gasperini_atscale_ABC.drop(Gasperini_atscale_ABC.filter(regex='_y$').columns, axis=1, inplace=True)
Gasperini_atscale_ABC = pd.merge(Gasperini_atscale_ABC, TSS_TF_pivot, how="left", left_on=["GeneSymbol"], right_on=["gene"], suffixes=('', '_y'))
Gasperini_atscale_ABC.drop(Gasperini_atscale_ABC.filter(regex='_y$').columns, axis=1, inplace=True)
Gasperini_atscale_ABC.to_csv(data_dir+"Gasperini2019.at_scale.ABC.TF.txt", sep='\t')
Gasperini_atscale_ABC.loc[Gasperini_atscale_ABC['atleast1Sig'] == True,].to_csv(data_dir+"Gasperini2019.at_scale.ABC.TF.atleast1sig.txt", sep='\t')
Gasperini_atscale_ABC.loc[:,list(TFcolumns)] = Gasperini_atscale_ABC.loc[:,list(TFcolumns)].fillna(0)

#e_TF = Gasperini_atscale_ABC.iloc[:,start:(start+len(TFcolumns))]
#TSS_TF = Gasperini_atscale_ABC.iloc[:,(start+len(TFcolumns)):(start+len(TFcolumns)+len(TFcolumns))]
#cobinding = pd.DataFrame(np.logical_and(e_TF, np.asarray(TSS_TF))).astype(int)  # ufunc(df1, np.asarray(df2)
#cobinding.columns = TFcolumns
#cobinding = cobinding.add_suffix('_co')
#Gasperini_atscale_ABC = pd.concat([Gasperini_atscale_ABC, cobinding], axis=1)
#Gasperini_atscale_ABC.to_csv(data_dir+"Gasperini2019.at_scale.ABC.TF.cobinding.txt", sep='\t')
#Gasperini_atscale_ABC.loc[Gasperini_atscale_ABC['atleast1Sig'] == True,].to_csv(data_dir+"Gasperini2019.at_scale.ABC.TF.cobinding.atleast1sig.txt", sep='\t')
e_TFfeatures =  Gasperini_atscale_ABC.loc[:,list(enhancer_TF_pivot.columns)]
NMFprefix='Gasperini2019.eTF.NMF'
pd.DataFrame(enhancer_TF_pivot.columns).to_csv(data_dir+NMFprefix+'.featureinput.txt', sep='\t')
eTF_nmf_reduced_features = DR_NMF_features(e_TFfeatures, data_dir, NMFprefix)
eTF_nmf_reduced_features = eTF_nmf_reduced_features.add_prefix('e')
NMFprefix='Gasperini2019.TSSTF.NMF'
pd.DataFrame(TSS_TF_pivot.columns).to_csv(data_dir+NMFprefix+'.featureinput.txt', sep='\t')
TSS_TFfeatures =  Gasperini_atscale_ABC.loc[:,list(TSS_TF_pivot.columns)]
TSSTF_nmf_reduced_features = DR_NMF_features(TSS_TFfeatures, data_dir, 'Gasperini2019.TSSTF.NMF')
TSSTF_nmf_reduced_features = TSSTF_nmf_reduced_features.add_prefix('TSS')
Gasperini_atscale_ABC = pd.concat([Gasperini_atscale_ABC, eTF_nmf_reduced_features, TSSTF_nmf_reduced_features], axis=1)
Gasperini_atscale_ABC.to_csv(data_dir+"Gasperini2019.at_scale.ABC.TF.NMF.txt", sep='\t')
Gasperini_atscale_ABC.loc[Gasperini_atscale_ABC['atleast1Sig'] == True,].to_csv(data_dir+"Gasperini2019.at_scale.ABC.TF.NMF.atleast1sig.txt", sep='\t')

# erole
erole = pd.read_csv(data_dir+"Gasperini2019.at_scale.eroles.txt", sep="\t")
Gasperini_atscale_ABC = pd.merge(Gasperini_atscale_ABC, erole, left_on=["name"], right_on=["name_x"])
Gasperini_atscale_ABC.drop(Gasperini_atscale_ABC.filter(regex='_x$').columns, axis=1, inplace=True)
Gasperini_atscale_ABC.to_csv(data_dir+"Gasperini2019.at_scale.ABC.TF.NMF.erole.txt", sep='\t')
Gasperini_atscale_ABC.loc[Gasperini_atscale_ABC['atleast1Sig'] == True,].to_csv(data_dir+"Gasperini2019.at_scale.ABC.TF.NMF.erole.atleast1sig.txt", sep='\t')


