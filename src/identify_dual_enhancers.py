import pickle as pkl
import igraph as ig
import numpy as np
import pandas as pd
import os

#import gene tss, chromosome info
gene_tss = {}
with open('gene_tss.uniq.tsv','r') as f:
    for line in f:
        chromosome, gene, tss = line.strip().split('\t')
        gene_tss[gene] = [chromosome, int(tss)]

remove = []
for gene in gene_tss:
    if os.path.isfile('gene_networks_chipped/'+gene+'_network.pkl'):
        pass
    else:
        remove.append(gene)

for i in remove:
    del gene_tss[i]        

#choose gene
#gene = 'BAX'

E1_class = {'TP':0,'FP':0,'FN':0,'TN':0,'NA':0}
E2_class = {'TP':0,'FP':0,'FN':0,'TN':0,'NA':0}
E3_class = {'TP':0,'FP':0,'FN':0,'TN':0,'NA':0}

E1_dist = []
E2_dist = []
E3_dist = []

C_E1_G = []
C_E2_G = []
C_E2_E1 = []
C_E3_G = []
C_E3_E2 = []

E1_act = []
E2_act = []
E3_act = []

contact_difference = []
difference_classes = []

p_e1_e2 = 0
p_e2_e1 = 0
e2_p_e1 = 0

enh_class = {}

genes = []
for gene in gene_tss:
    genes.append(gene)
    chromosome, tss = gene_tss[gene]
    #load network
    with open('gene_networks_chipped/'+gene+'_network.pkl', 'rb') as f:
        network = pkl.load(f)
    
    #promoter node
    resolution = 5000
    pnode = '_'.join([chromosome, str(int(np.floor(tss/resolution)*resolution)), str(int(np.floor(tss/resolution)*resolution+resolution))])
    pindx = network.vs.find(pnode).index
    #Total nodes
    node_indx = [v.index for v in network.vs]

    #ID E1s
    E1 = [v.index for v in network.vs.find(pnode).neighbors()]

    #ID E2s
    E2 = []
    for e in E1:
        neighbors = [v.index for v in network.vs.find(e).neighbors()]
        for n in neighbors:
            if n not in E1+E2: #not already a primary or secondary enhancer
                if n != network.vs.find(pnode).index: #not the promoter
                    E2.append(n)

    #ID E3s
    E3 = []
    for e in E2:
        neighbors = [v.index for v in network.vs.find(e).neighbors()]
        for n in neighbors:
            if n not in E1+E2+E3:
                if n != network.vs.find(pnode).index:
                    E3.append(n)

    #loop dynamics
    for e in E1:
        ch, start, end = network.vs.find(e)['name'].split('_')
        midpoint = int(int(end)+int(start))/2
        e1_dist = midpoint - tss
        neighbors = [v.index for v in network.vs.find(e).neighbors()]
        
        if e1_dist > 0:
            for n in neighbors:
                if n in E2:
                    ch, start, end = network.vs.find(n)['name'].split('_')
                    nidpoint = int(int(end)+int(start))/2
                    e2_dist = nidpoint - tss
                    if e2_dist < 0:
                        e2_p_e1 += 1
                    elif e2_dist < e1_dist:
                        p_e2_e1 += 1
                    elif e2_dist > e1_dist:
                        p_e1_e2 +=1
                    
        else:
            for n in neighbors:
                if n in E2:
                    ch, start, end = network.vs.find(n)['name'].split('_')
                    nidpoint = int(int(end)+int(start))/2
                    e2_dist = nidpoint - tss
                    if e2_dist > 0:
                        e2_p_e1 += 1
                    elif e2_dist > e1_dist:
                        p_e2_e1 += 1
                    elif e2_dist < e1_dist:
                        p_e1_e2 +=1
    
    #gather class info
    for e in E1:
        ename = network.vs.find(e)['name']
        if ename not in enh_class:
            enh_class[ename] = [1,0,0]
        else:
            enh_class[ename][0] += 1
        #classification data
        if 'classification' in network.vs.find(e)['enhancers']:
            for c in network.vs.find(e)['enhancers']['classification']:
                E1_class[c] += 1
        else:
            E1_class['NA'] += len(network.vs.find(e)['enhancers']['local_enhancers'])
        #contact data
        C_E1_G.append(network.vs.find(e)['Ceg'])
        #activity data
        for a in network.vs.find(e)['enhancers']['activity']:
            E1_act.append(a)

        ch, start, end = network.vs.find(e)['name'].split('_')
        midpoint = int(int(end)+int(start))/2
        E1_dist.append(midpoint - tss)

    for e in E2:
        ename = network.vs.find(e)['name']
        if ename not in enh_class:
            enh_class[ename] = [0,1,0]
        else:
            enh_class[ename][1] += 1
        #classification data
        if 'classification' in network.vs.find(e)['enhancers']:
            for c in network.vs.find(e)['enhancers']['classification']:
                E2_class[c] += 1
        else:
            E2_class['NA'] += len(network.vs.find(e)['enhancers']['local_enhancers'])

        #gather contact between various nodes
        C_E2_G.append(network.vs.find(e)['Ceg'])
        for n in network.vs.find(e).neighbors():
            if n.index in E1:
                c_e2_e1 = network.es[network.get_eid(n,e)]['contact']
                C_E2_E1.append(c_e2_e1)
                if 'classification' in network.vs.find(e)['enhancers']:
                    for clas in network.vs.find(e)['enhancers']['classification']:
                        contact_difference.append(c_e2_e1 - n['Ceg'])
                        difference_classes.append(clas)
            if n.index in E3:
                C_E3_E2.append(network.es[network.get_eid(n,e)]['contact'])
         
        #activity data
        for a in network.vs.find(e)['enhancers']['activity']:
            E2_act.append(a)

        ch, start, end = network.vs.find(e)['name'].split('_')
        midpoint = int(int(end)+int(start))/2
        E2_dist.append(midpoint - tss)

        #tfs

    for e in E3:
        ename = network.vs.find(e)['name']
        if ename not in enh_class:
            enh_class[ename] = [0,0,1]
        else:
            enh_class[ename][2] += 1
        #classification data
        if 'classification' in network.vs.find(e)['enhancers']:
            for c in network.vs.find(e)['enhancers']['classification']:
                E3_class[c] += 1
        else: 
            E3_class['NA'] += len(network.vs.find(e)['enhancers']['local_enhancers'])

        #contact data
        C_E3_G.append(network.vs.find(e)['Ceg'])
  
        #activity data
        for a in network.vs.find(e)['enhancers']['activity']:
            E3_act.append(a)

        ch, start, end = network.vs.find(e)['name'].split('_')
        midpoint = int(int(end)+int(start))/2
        E3_dist.append(midpoint - tss)

with open('enh_dual_role.tsv','w') as f:
    for ename in enh_class:
        f.write('\t'.join([ename, str(enh_class[ename][0]), str(enh_class[ename][1]), str(enh_class[ename][2])])+'\n')

#with open('binding_patterns.tsv','w') as f:
#    f.write('\t'.join(['gene','tf','promoter_only','promoter_e1','promoter_e1_e2','promoter_e2','e1_only','e1_e2','e2_only','none'])+'\n')
#    for tf in bind_patterns:
#        for i in range(len(genes)):
#            f.write('\t'.join([genes[i], tf, str(bind_patterns[tf]['p'][i]), str(bind_patterns[tf]['p_e1'][i]), str(bind_patterns[tf]['p_e1_e2'][i]), str(bind_patterns[tf]['p_e2'][i]), str(bind_patterns[tf]['e1'][i]), str(bind_patterns[tf]['e1_e2'][i]), str(bind_patterns[tf]['e2'][i]), str(bind_patterns[tf]['notf'][i])])+'\n')
########print(p_e1_e2)
########print(p_e2_e1)
########print(e2_p_e1)
########with open('E1_P_dist.txt','w') as f:
########    for d in E1_dist:
########        f.write(str(d)+'\n')

########with open('E2_P_dist.txt','w') as f:
########    for d in E2_dist:
########        f.write(str(d)+'\n')

########with open('E3_P_dist.txt','w') as f:
########    for d in E3_dist:
########        f.write(str(d)+'\n')
########with open('C_E1_G.txt','w') as f:
########    for c in C_E1_G:
########        f.write(str(c)+'\n')

########with open('C_E2_G.txt','w') as f:
########    for c in C_E2_G:
########        f.write(str(c)+'\n')

########with open('C_E3_G.txt','w') as f:
########    for c in C_E3_G:
########        f.write(str(c)+'\n')

########with open('C_E2_E1.txt','w') as f:
########    for c in C_E2_E1:
########        f.write(str(c)+'\n')

########with open('C_E3_E2.txt','w') as f:
########    for c in C_E3_E2:
########        f.write(str(c)+'\n')

########with open('activity_scores/E1_activity.txt','w') as f:
########    for a in E1_act:
########        f.write(str(a)+'\n')

########with open('activity_scores/E2_activity.txt','w') as f:
########    for a in E2_act:
########        f.write(str(a)+'\n')

########with open('activity_scores/E3_activity.txt','w') as f:
########    for a in E3_act:
########        f.write(str(a)+'\n')

########with open('contact_diff.txt','w') as f:
########    for i in range(0,len(contact_difference)):
########        f.write(str(contact_difference[i])+'\t'+difference_classes[i]+'\n')
