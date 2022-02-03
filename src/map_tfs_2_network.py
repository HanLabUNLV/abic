import igraph as ig
import pandas as pd
import pickle as pkl
from pathlib import Path

gene_tss_file = 'gene_tss.uniq.tsv'
with open(gene_tss_file, 'r') as f:
    lines = f.readlines()[1:]
gene_tss = {}
for line in lines:
    chrm, gene, tss = line.strip().split('\t')
    if Path('gene_networks_optimized/'+gene+'_network.pkl').is_file():
        tss = int(tss)
        gene_tss[gene] = tss


tf_matrix = pd.read_csv('enhancer_chipseq_featurematrix.tsv',sep='\t')
#gene = 'JUNB'
for gene in gene_tss:
    with open('gene_networks_optimized/'+gene+'_network.pkl','rb') as f:
        network = pkl.load(f)

    for n in network.vs:
        n['tf'] = {}
        for tf in tf_matrix.columns.tolist()[2:]:
            n['tf'][tf] = []

        for enh in n['enhancers']['local_enhancers']:
            if enh[1]==enh[2]: #if promoter node, the window had to be stretched to work with bedtools, so lookup is different
                pid = '_'.join([enh[0], str(enh[1]-500), str(enh[1]+500)])
                row = tf_matrix.loc[tf_matrix['enhancer']==pid,]
            else:
                enhid = '_'.join([str(x) for x in enh])
                row = tf_matrix.loc[tf_matrix['enhancer']==enhid,]

            for tf in row.columns.tolist()[2:]:
                if len(row[tf])>0:
                    n['tf'][tf].append(row[tf].values[0])
                else:
                    n['tf'][tf].append('NA')
                
    with open('gene_networks_chipped/'+gene+'_network.pkl','wb') as f:
        pkl.dump(network, f)
