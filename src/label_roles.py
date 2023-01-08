import pickle as pkl
import os
import numpy as np
gene_tss = {}
tss_file = 'data/gene_tss.gas.long.tsv'
gene_networks_dir = 'data/gene_networks_validated_2/'
with open(tss_file,'r') as f:
    for line in f:
        chromosome, gene, tss = line.strip().split('\t')
        gene_tss[gene] = [chromosome, int(tss)]

remove = []
for gene in gene_tss:
    if os.path.isfile(gene_networks_dir+gene+'_network.pkl'):
        pass
    else:
        remove.append(gene)

for i in remove:
    del gene_tss[i]


genes = []
for gene in gene_tss:
    genes.append(gene)
    chromosome, tss = gene_tss[gene]
    #load network
    with open(gene_networks_dir+gene+'_network.pkl', 'rb') as f:
        network = pkl.load(f)

    #promoter node
    resolution = 5000
    pnode = '_'.join([chromosome, str(int(np.floor(tss/resolution)*resolution)), str(int(np.floor(tss/resolution)*resolution+resolution))])
    try:
        pindx = network.vs.find(pnode).index
    except:
        print('gene: ' + gene + ' has no promoter')
        continue 
   #Total nodes
    node_indx = [v.index for v in network.vs]

    #ID E1s
    E1 = [v.index for v in network.vs.find(pnode).neighbors()]
    for n in E1:
        network.vs.find(n)['role'] = 'E1'

    #ID E2s
    E2 = []
    for e in E1:
        neighbors = [v.index for v in network.vs.find(e).neighbors()]
        for n in neighbors:
            if n not in E1+E2: #not already a primary or secondary enhancer
                if n != network.vs.find(pnode).index: #not the promoter
                    E2.append(n)
                    network.vs.find(n)['role']='E2'
    #ID E3s
    E3 = []
    for e in E2:
        neighbors = [v.index for v in network.vs.find(e).neighbors()]
        for n in neighbors:
            if n not in E1+E2+E3:
                if n != network.vs.find(pnode).index:
                    E3.append(n)
                    network.vs.find(n)['role'] = 'E3'
    with open(gene_networks_dir+gene+'_network.pkl', 'wb') as f:
        network = pkl.dump(network,f)

