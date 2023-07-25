import pickle as pkl
import os
import pandas as pd 
#iterate through genes
#check if enh in dict, if not, add, else, add gene to list
#then create binary membership matrix where rows are enh bins and cols are genes
netdir = 'data/gene_networks_validated_2/'
gene_tss = {}
with open('data/gene_tss.gas.long.tsv','r') as f:
    for line in f:
        chromosome, gene, tss = line.strip().split('\t')
        gene_tss[gene] = [chromosome, int(tss)]

remove = []
for gene in gene_tss:
    if os.path.isfile(netdir+gene+'_network.pkl'):
        pass
    else:
        remove.append(gene)

for i in remove:
    del gene_tss[i]

enh_mem = {}

for gene in gene_tss:
    with open(netdir+gene+'_network.pkl','rb') as f:
        network = pkl.load(f)

    for enh_id in network.vs['name']:
        if enh_id not in enh_mem:
            enh_mem[enh_id] = [gene]
        else:
            enh_mem[enh_id].append(gene)

enh_ids = [x for x in enh_mem.keys()]
genes = [g for g in gene_tss.keys()]

data = pd.DataFrame(columns=['enh_id']+genes)

for enh in enh_ids:
    membership = ['None']*len(genes)
    for i in range(len(genes)):
        gene = genes[i]
        if gene in enh_mem[enh]:
            with open(netdir+gene+'_network.pkl','rb') as f:
                network = pkl.load(f)
            membership[i] = str(network.vs.find(enh)['role'])
    
    entry = [enh] + membership
    #data.loc[data['enh_id']==enh] = entry
    data.loc[len(data)] = entry
data.to_csv('data/enh_network_membership.csv', index=False)

