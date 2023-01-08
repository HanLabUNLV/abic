import pickle as pkl
import os
import numpy as np

gene_tss = {}
with open('data/gene_tss.gas.long.tsv','r') as f:
    for line in f:
        chromosome, gene, tss = line.strip().split('\t')
        gene_tss[gene] = [chromosome, int(tss)]

remove = []
for gene in gene_tss:
    if os.path.isfile('data/gene_networks_validated_2/'+gene+'_network.pkl'):
        pass
    else:
        remove.append(gene)

for i in remove:
    del gene_tss[i]

rows = []
for gene in gene_tss:
    with open('data/gene_networks_validated_2/'+gene+'_network.pkl','rb') as f:
        network = pkl.load(f)

    chromosome, tss = gene_tss[gene]    
    resolution = 5000
    pnode = '_'.join([chromosome, str(int(np.floor(tss/resolution)*resolution)), str(int(np.floor(tss/resolution)*resolution+resolution))])    

    if pnode in [v['name'] for v in network.vs]:    
        for e1 in network.vs.find(pnode).neighbors():
            if e1['role']=='E1':
                for e2 in e1.neighbors():
                    if e2['role']=='E2':
                        rows.append([e1['name'], e2['name'], gene])
    else:
        print('promoter node pruned out of gene: ' + gene)
                    

with open('data/e1_e2_validated_pairs.tsv','w') as f:
    f.write('\t'.join(['e1','e2','gene'])+'\n')
    for row in rows:
        f.write('\t'.join(row) + '\n')

