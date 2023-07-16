import joblib as jl
import os
import numpy as np
import argparse
import time

parser = argparse.ArgumentParser(description = "Description for my parser")
parser.add_argument("-g", "--gene", help = "Gene Symbol of single gene", required = False, default = "")
argument = parser.parse_args()
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

if argument.gene !='':
    genes = [argument.gene]
else:
    genes = list(gene_tss.keys())
    print('Single gene not found: running for all ' + str(len(genes)) + ' genes')
    time.sleep(2)

for gene in genes:
    print(gene)
    chromosome, tss = gene_tss[gene]
    #load network
    network = jl.load(gene_networks_dir+gene+'_network.pkl')

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

    print('For gene: '+gene+' E1 labels complete')
    #ID E2s
    E2 = []
    for e in E1:
        neighbors = [v.index for v in network.vs.find(e).neighbors()]
        for n in neighbors:
            if n not in E1+E2: #not already a primary or secondary enhancer
                if n != network.vs.find(pnode).index: #not the promoter
                    E2.append(n)
                    network.vs.find(n)['role']='E2'
    print('For gene: '+gene+' E2 labels complete')
    #ID E3s
    E3 = []
    for e in E2:
        neighbors = [v.index for v in network.vs.find(e).neighbors()]
        for n in neighbors:
            if n not in E1+E2+E3:
                if n != network.vs.find(pnode).index:
                    E3.append(n)
                    network.vs.find(n)['role'] = 'E3'
    print('For gene: '+gene+' E3 labels complete')
    jl.dump(network, gene_networks_dir+gene+'_network.pkl')
    

