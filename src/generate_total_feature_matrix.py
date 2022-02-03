import pickle as pkl
import os
import numpy as np

gene_tss = {}
with open('gene_tss.uniq.tsv','r') as f:
    for line in f:
        chromosome, gene, tss = line.strip().split('\t')
        gene_tss[gene] = [chromosome, int(tss)]

remove = []
for gene in gene_tss:
    if os.path.isfile('gene_networks_chipped_labeled/'+gene+'_network_labeled.pkl'):
        pass
    else:
        remove.append(gene)

for i in remove:
    del gene_tss[i]

#set up empty data structure
rows = []
gene = [f for f in gene_tss.keys()][0]
with open('gene_networks_chipped_labeled/'+gene+'_network_labeled.pkl','rb') as f:
    network = pkl.load(f)
chromosome, tss = gene_tss[gene]
resolution = 5000
pnode = '_'.join([chromosome, str(int(np.floor(tss/resolution)*resolution)), str(int(np.floor(tss/resolution)*resolution+resolution))])
tfs = []
for tf in network.vs.find(pnode)['tf']:
    tfs.append(tf)

#populate data structure
#each row will be an enhancer-promoter pair
#contents will be enhancer, promoter, gene, all tf peaks, and other attributes of network structure like activity and contact
for gene in gene_tss:
    with open('gene_networks_chipped_labeled/'+gene+'_network_labeled.pkl','rb') as f:
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
    for tf in network.vs.find(pnode)['tf']:
        p_peaks[tf] = network.vs.find(pnode)['tf'][tf][j]

    #iter through nodes, get cobinding with promoter
    for n in network.vs:
        #order of columns: echr estart estop tss gene activity abc_score significant classification role [every tf peak]
        for i in range(len(n['enhancers']['local_enhancers'])):
            row = [i for i in n['enhancers']['local_enhancers'][i]]
            row.append(tss)
            row.append(gene)
            row.append(n['enhancers']['activity'][i])
            row.append(n['Ceg'])
            
            #some enhancers are missing some values, fill with NA
            try:
                row.append(n['enhancers']['abc_score'][i])
            except:
                row.append('NA')
                
            try:
                row.append(n['enhancers']['sig'][i])
            except:
                row.append('NA')

            try:
                row.append(n['enhancers']['classification'][i])
            except:    
                row.append('NA')

            try:
                row.append(n['role'])
            except:    
                row.append('NA')

            for tf in tfs:
                row.append(n['tf'][tf][i])        

            rows.append(row)
colnames = ['chr','start','stop','tss','gene','activity','contact','abc_score','sig','classification','role']
colnames.extend(tfs)
with open('full_feature_matrix.tsv','w') as f:
    f.write('\t'.join(colnames)+'\n')
    for row in rows:
        f.write('\t'.join([str(x) for x in row])+'\n')

