from predictor import *
from igraph import *
import pandas as pd
import numpy as np


gene_tss = {}
with open('data/gene_tss.gas.long.tsv','r') as f:
    for line in f:
        chrm, gene, tss = line.strip().split('\t')
        gene_tss[gene] = [chrm, int(tss)]


data = []
columns = ['Gene','tss','PromoterDegree','PromoterCloseness','nNodes','nEdges']
for gene in gene_tss:
    try:
        with open('data/gene_networks_validated_2/'+gene+'_network.pkl','rb') as f:
            network = pkl.load(f)
    except:
        print('Network not found for gene: '+ gene)
        continue
    else:
        chromosome, tss = gene_tss[gene]
        hic_resolution = 5000
        promoter_node = '_'.join([chromosome,str(int(np.floor(tss/hic_resolution)*hic_resolution)), str(int(np.floor(tss/hic_resolution)*hic_resolution + hic_resolution))])        
        try:
            pnode_degree = network.vs.find(promoter_node).degree()
            pnode_closeness = network.vs.find(promoter_node).closeness()
        except:
            print('Promoter not in gene: '+gene)
            pnode_degree = 0
            pnode_closeness = 0
        else:
            nnodes = len([n for n in network.vs])
            nedges = len([e for e in network.es])
        data.append([gene, promoter_node, pnode_degree, pnode_closeness, nnodes, nedges])
        

out = pd.DataFrame(data, columns=columns)
out.to_csv('data/network_stats.tsv',sep='\t', index=None)
