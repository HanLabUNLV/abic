import pickle as pkl
import os
import numpy as np

tss_file = 'data/gene_tss.gas.long.tsv'
gene_networks_dir = 'data/gene_networks_validated_2/'
outfile = 'data/full_feature_matrix.validated2.tsv'

gene_tss = {}
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

#set up empty data structure
rows = []
gene = [f for f in gene_tss.keys()][1]
with open(gene_networks_dir+gene+'_network.pkl','rb') as f:
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
    with open(gene_networks_dir+gene+'_network.pkl','rb') as f:
        network = pkl.load(f)

    chromosome, tss = gene_tss[gene]    
    resolution = 5000
    pnode = '_'.join([chromosome, str(int(np.floor(tss/resolution)*resolution)), str(int(np.floor(tss/resolution)*resolution+resolution))])    


    #identify which e in node is promoter 
    j = 0
    if pnode in network.vs['name']:
        for i in network.vs.find(pnode)['enhancers']['local_enhancers']:
            if i[1]==i[2]:
                break
            else:
                j+=1
    else:
        print('gene: ' + gene + ' has no promoter node in network')
        
    
    #identify peaks in promoter 
    #p_peaks = {}
    #for tf in network.vs.find(pnode)['tf']:
    #    p_peaks[tf] = network.vs.find(pnode)['tf'][tf][j]

    #iter through nodes, get cobinding with promoter
    for n in network.vs:
        #order of columns: echr estart estop tss gene activity abc_score significant classification role [every tf peak]
        if n['enhancers'] is not None:
            for i in range(len(n['enhancers']['local_enhancers'])):
                row = [i for i in n['enhancers']['local_enhancers'][i]]
                row.append(tss)
                row.append(gene)
                row.append(n['enhancers']['activity'][i])
                #to get contact, find edge between node and promoter
                try:
                    eid = network.get_eid(network.vs.find(pnode), n.index)
                    row.append(network.es[eid]['contact'])
                    #print(network.es[eid]['contact'])
                except:
                    row.append(0)
                #some enhancers are missing some values, fill with NA
                try:
                    row.append(n['enhancers']['abc_score'][i])
                except:
                    row.append('NA')
       
                try:
                    row.append(n['enhancers']['effect_size'][i])
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

                try:
                    row.append(network.degree(n))
                except:    
                    row.append('NA')

                for tf in tfs:
                    try:
                        row.append(n['tf'][tf][i])        
                    except:
                        row.append('NA')
                rows.append(row)
        colnames = ['chr','start','stop','tss','gene','activity','contact','abc_score','effect_size','sig','classification','role','degree']
colnames.extend(tfs)
with open(outfile,'w') as f:
    f.write('\t'.join(colnames)+'\n')
    for row in rows:
        f.write('\t'.join([str(x) for x in row])+'\n')

