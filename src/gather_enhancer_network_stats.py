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
columns = ['Enhancer','Gene','Role','Closeness','Betweenness']
for gene in gene_tss:
    print(gene)
    try:
        with open('data/gene_networks_validated_2/'+gene+'_network.pkl','rb') as f:
            network = pkl.load(f)
            network.simplify()
    except:
        print('Network not found for gene: '+ gene)
        continue
    
    else:
        chromosome = gene_tss[gene][0]
        tss = gene_tss[gene][1]
        hic_resolution = 5000
        promoter_node = '_'.join([chromosome,str(int(np.floor(tss/hic_resolution)*hic_resolution)), str(int(np.floor(tss/hic_resolution)*hic_resolution + hic_resolution))])        
        try:
            pnode_degree = network.vs.find(promoter_node).degree()
            pnode_closeness = network.vs.find(promoter_node).betweenness()
        except:
            print('Promoter not in gene: '+gene)
            pnode_degree = 0
            pnode_closeness = 0
        else:
            #this is where we check each enhancer and record the role and the betweenness centrality, degree, etc
            for v in network.vs:
                print(v['name'])
                if v['name']==promoter_node:
                    ename = promoter_node
                    role = 'P'
                    closeness = network.vs.find(promoter_node).closeness()
                    betweenness  = network.vs.find(promoter_node).betweenness()
                else:
                    if v['role'] is not None:
                        ename = v['name']
                        role = v['role']
                        closeness = network.vs.find(ename).closeness()
                        betweenness  = network.vs.find(ename).betweenness()
        data.append([ename, gene, role, closeness, betweenness])
        


out = pd.DataFrame(data, columns=columns)
out.to_csv('data/network_enhancer_stats.tsv',sep='\t', index=None)
