import pickle as pkl
import os
import numpy as np

gene_tss = {}
with open('data/gene_tss.uniq.tsv','r') as f:
    for line in f:
        chromosome, gene, tss = line.strip().split('\t')
        gene_tss[gene] = [chromosome, int(tss)]

remove = []
for gene in gene_tss:
    if os.path.isfile('data/gene_networks_wd/'+gene+'_network.pkl'):
        pass
    else:
        remove.append(gene)

for i in remove:
    del gene_tss[i]

#set up empty data structure
data = {}
gene = [f for f in gene_tss.keys()][0]
with open('data/gene_networks_wd/'+gene+'_network.pkl','rb') as f:
    network = pkl.load(f)
chromosome, tss = gene_tss[gene]
resolution = 5000
pnode = '_'.join([chromosome, str(int(np.floor(tss/resolution)*resolution)), str(int(np.floor(tss/resolution)*resolution+resolution))])
for tf in network.vs.find(pnode)['tf']:
    data[tf] = {'P-E1':[0,0,0,0], 'P-E2':[0,0,0,0], 'E1-E2':[0,0,0,0]}
    #0th entry is peak found in x1 and x2
    #1st entry is peak found in x1 not x2
    #2nd entry is peak found in x2 not x1
    #3rd entry is peak found in neither x2 nor n2

#populate data structure
for gene in gene_tss:
    with open('data/gene_networks_wd/'+gene+'_network.pkl','rb') as f:
        network = pkl.load(f)

    chromosome, tss = gene_tss[gene]    
    resolution = 5000
    pnode = '_'.join([chromosome, str(int(np.floor(tss/resolution)*resolution)), str(int(np.floor(tss/resolution)*resolution+resolution))])    


    #identify which e in node is promoter 
    j = 0
    for i in network.vs.find(pnode)['enhancers']['local_enhancers']:
        if i[1]==i[2]:
            break
        else:
            j+=1
    
    #identify peaks in promoter 
    p_peaks = {}
    pmissing=False
    for tf in network.vs.find(pnode)['tf']:
        try:
            p_peaks[tf] = network.vs.find(pnode)['tf'][tf][j]
        except:
            print(gene+' no promoter found in network/peaks missing from promoter')
            pmissing = True
            break
    if pmissing:
        continue
    #iter through nodes, get cobinding with promoter
    for n in network.vs:
        if n['role']=='E1':
            for tf in n['tf']:
                if p_peaks[tf]==1:
                    data[tf]['P-E1'][0] += n['tf'][tf].count(1) #add number of 1s
                    data[tf]['P-E1'][1] += n['tf'][tf].count(0) #add number of 0s
                else:
                    data[tf]['P-E1'][2] += n['tf'][tf].count(1) #add number of 1s
                    data[tf]['P-E1'][3] += n['tf'][tf].count(0) #add number of 0s

                for e2 in n.neighbors():
                    if e2['role']=='E2':
                        data[tf]['E1-E2'][0] += n['tf'][tf].count(1)*e2['tf'][tf].count(1) #pairwise combinations between peaks
                        data[tf]['E1-E2'][1] += n['tf'][tf].count(1)*e2['tf'][tf].count(0) #pairwise combinations between peaks
                        data[tf]['E1-E2'][2] += n['tf'][tf].count(0)*e2['tf'][tf].count(1) #pairwise combinations between peaks
                        data[tf]['E1-E2'][3] += n['tf'][tf].count(0)*e2['tf'][tf].count(0) #pairwise combinations between peaks
                        

        elif n['role']=='E2':
            for tf in n['tf']:
                if p_peaks[tf]==1:
                    data[tf]['P-E2'][0] += n['tf'][tf].count(1) #add number of 1s
                    data[tf]['P-E2'][1] += n['tf'][tf].count(0) #add number of 0s
                else:
                    data[tf]['P-E2'][2] += n['tf'][tf].count(1) #add number of 1s
                    data[tf]['P-E2'][3] += n['tf'][tf].count(0) #add number of 0s

                    

with open('data/cobinding_chi.tsv','w') as f:
    f.write('\t'.join(['tf','comparison','peak.peak','peak.nopeak','nopeak.peak','nopeak.nopeak'])+'\n')
    for tf in data:
        for comparison in data[tf]:
            entry = [tf, comparison] + [str(x) for x in data[tf][comparison]]
            f.write('\t'.join(entry)+'\n')

