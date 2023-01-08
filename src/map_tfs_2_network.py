import igraph as ig
import pandas as pd
import pickle as pkl
from pathlib import Path
#adds tf binding data to nodes in place
gene_tss_file = 'data/gene_tss.gas.long.tsv'
gene_networks_dir = 'data/gene_networks_validated_2/'
with open(gene_tss_file, 'r') as f:
    lines = f.readlines()[1:]
gene_tss = {}
for line in lines:
    chrm, gene, tss = line.strip().split('\t')
    if Path(gene_networks_dir+gene+'_network.pkl').is_file():
        tss = int(tss)
        gene_tss[gene] = tss


tf_matrix = pd.read_csv('data/enhancer_chipseq_featurematrix.validated2.tsv',sep='\t')
ptf_matrix = pd.read_csv('data/promoter_chipseq_featurematrix.tsv',sep='\t')
#gene = 'JUNB'
for gene in gene_tss:
    with open(gene_networks_dir+gene+'_network.pkl','rb') as f:
        network = pkl.load(f)

    for n in network.vs:
        n['tf'] = {}
        for tf in tf_matrix.columns.tolist()[1:]:
            n['tf'][tf] = []
        if n['enhancers'] is not None:
            for enh in n['enhancers']['local_enhancers']:
                if enh[1]==enh[2]: #if promoter node, the window had to be stretched to work with bedtools, so lookup is different
                    pid = '_'.join([enh[0], str(enh[1]-500), str(enh[1]+500)])
                    row = ptf_matrix.loc[ptf_matrix['promoter']==pid,]
                else:
                    enhid = '_'.join([str(x) for x in enh])
                    row = tf_matrix.loc[tf_matrix['enhancer']==enhid,]

                for tf in row.columns.tolist()[1:]:
                    if len(row[tf])>0:
                        n['tf'][tf].append(row[tf].values[0])
                    else:
                        n['tf'][tf].append('NA')
                
    with open(gene_networks_dir+gene+'_network.pkl','wb') as f:
        pkl.dump(network, f)
