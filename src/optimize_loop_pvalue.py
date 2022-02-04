import pandas as pd
import pickle as pkl
import argparse
from scipy import optimize
import numpy as np
import os.path
import sys
import argparse
from copy import deepcopy
def flatten(t):
    return [item for sublist in t for item in sublist]

def filter_edges(network_orig, threshold, local_loops, pnode, pred_p_nodes=[]):
    network = deepcopy(network_orig)
    if pred_p_nodes==[]:
        neighbors = network.vs.find(pnode).neighbors()
        pred_p_nodes = list(set([x['name'] for x in neighbors]))
        filtered_loops = local_loops.loc[((local_loops['node1']==pnode)&(local_loops['node2'].isin(pred_p_nodes))&(local_loops['p-value']<threshold))|((local_loops['node2']==pnode)&(local_loops['node1'].isin(pred_p_nodes))&(local_loops['p-value']<threshold)),]
    else:
        filtered_loops = local_loops.loc[((local_loops['node1']==pnode)&(local_loops['node2'].isin(pred_p_nodes))&(local_loops['p-value']<threshold))|((local_loops['node2']==pnode)&(local_loops['node1'].isin(pred_p_nodes))&(local_loops['p-value']<threshold)),]


    #filter network by max p value
    node1s = filtered_loops['node1'].tolist()
    node2s = filtered_loops['node2'].tolist()
    p_values = filtered_loops['p-value'].tolist()
    enames_kept = [[node1s[i],node2s[i],p_values[i]] for i in range(0,len(node1s))]
    eids_kept = []
    eids_missed = []
    for e in enames_kept:
        try:
            source=network.vs.find(e[0])
            target=network.vs.find(e[1])
            eids_kept.append([source.index,target.index, e[2]])

        except:
            eids_missed.append(e)

    eids_kept = [e[:2] for e in eids_kept]

    final_eids = network.get_eids(eids_kept)
    total_eids = [e.index for e in network.es[:]]
    lost_eids = list(set(total_eids) - set(final_eids))
    #delete edges of lost loops
    network.delete_edges(lost_eids)
    #some nodes are unconnected now, so clean them off
    network.vs.select(_degree=0).delete()
    if len([v for v in network.vs]) < 2:
        return(False)
    return(network)



def filter_edges_global(network_orig, threshold, local_loops, pnode):
    network = deepcopy(network_orig)
    nodes = [v for v in network.vs]
    pred_p_nodes = list(set([x['name'] for x in nodes]))
    filtered_loops = local_loops.loc[((local_loops['node1'].isin(pred_p_nodes))&(local_loops['node2'].isin(pred_p_nodes))&(local_loops['p-value']<threshold)),]

    #filter network by max p value
    node1s = filtered_loops['node1'].tolist()
    node2s = filtered_loops['node2'].tolist()
    p_values = filtered_loops['p-value'].tolist()
    enames_kept = [[node1s[i],node2s[i],p_values[i]] for i in range(0,len(node1s))]
    eids_kept = []
    eids_missed = []
    for e in enames_kept:
        try:
            source=network.vs.find(e[0])
            target=network.vs.find(e[1])
            eids_kept.append([source.index,target.index, e[2]])

        except:
            eids_missed.append(e)

    eids_kept = [e[:2] for e in eids_kept]

    final_eids = network.get_eids(eids_kept)
    total_eids = [e.index for e in network.es[:]]
    lost_eids = list(set(total_eids) - set(final_eids))
    #delete edges of lost loops
    network.delete_edges(lost_eids)
    #some nodes are unconnected now, so clean them off
    network.vs.select(_degree=0).delete()
    if len([v for v in network.vs]) < 2:
        return(False)
    return(network)


def prune_network(gene, network, enhancers, chromosome):
    #find relevant loops
    tss = gene_tss[gene][1]
    pnode = '_'.join([chromosome, str(int(np.floor(tss/resolution)*resolution)), str(int(np.floor(tss/resolution)*resolution+resolution))])
    lbound = tss - window/2
    ubound = tss + window/2
    loops = pd.read_csv('fithic_loops/'+chromosome+'/fithic_filtered.bedpe', sep='\t')
    loops['node1'] = loops.chr1.str.cat([loops.start1.astype(int).astype(str),loops.end1.astype(int).astype(str)],sep='_')
    loops['node2'] = loops.chr2.str.cat([loops.start2.astype(int).astype(str),loops.end2.astype(int).astype(str)],sep='_')
    local_loops = loops.loc[(loops['start1']>lbound)&(loops['end1']>lbound)&(loops['start2']>lbound)&(loops['end2']>lbound),]
    local_loops = local_loops.loc[(local_loops['start1']<ubound)&(local_loops['end1']<ubound)&(local_loops['start2']<ubound)&(local_loops['end2']<ubound),]

    #find p-value that connects all pred_positive
    #what we're really looking for is lowest the p_value of the predicted true EG pairs
    local_egs = enhancers.loc[(enhancers['start']>lbound)&(enhancers['end']<ubound)&(enhancers['TargetGene']==gene)&(enhancers['classification'].isin(['TP','FP'])),].copy()
    local_egs['bin1'] = local_egs.apply(lambda row: np.floor(row['start']/resolution)*resolution, axis=1)
    local_egs['bin1'] = local_egs['bin1'].astype(int)
    local_egs['bin2'] = local_egs['bin1'] + resolution
    local_egs['node'] = local_egs.chr.str.cat([local_egs.bin1.astype(str), local_egs.bin2.astype(str)], sep='_')
    pred_p_nodes = local_egs.node.tolist()

    #identify all loops between local_egs nodes and promoter node
    #then produce their p values to find maximum
    local_ps = local_loops.loc[(local_loops['node1']==pnode)&(local_loops['node2'].isin(pred_p_nodes)), 'p-value'].tolist()
    local_ps.extend(local_loops.loc[(local_loops['node2']==pnode)&(local_loops['node1'].isin(pred_p_nodes)),'p-value'].tolist())
    #also might be interesting to see differences in tp and fp distributions of p value    
    try:
        threshold = max(local_ps)
    except ValueError as e:
        return(e)
    #all possible configurations connecting pred_p_nodes with promoter nodes below a p value threshold
    network = filter_edges(network, threshold, local_loops, pnode, pred_p_nodes)
    #somewhere in here do like an optimization?
    #when developing ABCD maybe, but for now theoretically keeping all the connections to the promoter that are pred positive and using the highest p value 
    return([gene,network])
    #add p-value??
    #with open('gene_networks_filtered/'+gene+'_network.pkl','wb') as f:
    #    pkl.dump(network, f)

def estimate_ABC(network, pnode):
    if pnode in network.vs['name']:
        promoter = network.vs.find(pnode)
        neighbors = promoter.neighbors()
        abc_nums = []
        for neighbor in neighbors:
            contact = network.es[network.get_eid(pnode,neighbor['name'])]['contact'] #contact between enhancer and gene
            abc_num = []
            for activity in neighbor['enhancers']['activity']:
                abc_num.append(activity*contact)
            neighbor['enhancers']['abc_num'] = abc_num
            abc_nums.append(abc_num)         
        #once all nums have been collected, go back through and divide by their sum to get final abc 
        abc_denom = sum(flatten(abc_nums)) 
        for neighbor in neighbors:
            abc_score = []
            for abc_num in neighbor['enhancers']['abc_num']:
                abc_score.append(abc_num/abc_denom)
            neighbor['enhancers']['abc_score'] = abc_score
    else:
        print('promoter pruned out')
        return(False)
    #now the network should be populated with abc_recalc near the promoter
    return(network)

def classify_network(gene, ground_truth, network, pnode, threshold):
    ground_truth = deepcopy(ground_truth.loc[ground_truth['TargetGene']==gene,])
    promoter = network.vs.find(pnode)
    neighbors = [v for v in network.vs] #promoter.neighbors()
    
    for neighbor in neighbors:
        sigs = []
        for local_enh in neighbor['enhancers']['local_enhancers']:
            chrm, start, end = local_enh
            sig = ground_truth.loc[(ground_truth['chr']==chrm)&(ground_truth['start']==start)&(ground_truth['end']==end),'significant']
            if len(sig.index)==1:
                sigs.append(sig.values[0])
            elif len(sig.index)==0:
                #print('no match for enh: ' + str(local_enh))
                sigs.append('NA')
            else:
                print('multiple matches?')
                print(sig)
                sigs.append('NA')
        neighbor['enhancers']['sig'] = sigs

    #sigs identified, use abc threshold to classify
    for neighbor in neighbors:
        classes = []
        if 'abc_score' not in neighbor['enhancers']:
            for enh in range(0, len(neighbor['enhancers']['activity'])):
                sig = neighbor['enhancers']['sig'][enh]
                if sig==0:
                    classes.append('TN')
                elif sig ==1:
                    classes.append('FN')
                else:
                    classes.append('NA')
        else:
            for enh in range(0,len(neighbor['enhancers']['sig'])):
                pred_sig = int(neighbor['enhancers']['abc_score'][enh]>threshold)
                sig = neighbor['enhancers']['sig'][enh]
                if (pred_sig==1) & (sig==1):
                    classes.append('TP')
                elif (pred_sig==1) & (sig==0):
                    classes.append('FP')
                elif (pred_sig==0) & (sig==1):
                    classes.append('FN')
                elif (pred_sig==0) & (sig==0):
                    classes.append('TN')
                else:
                    classes.append('NA')
            neighbor['enhancers']['classification'] = classes
    return(network)

def network_confusion_matrix(network, pnode):
    promoter = network.vs.find(pnode)
    neighbors = [v for v in network.vs] #promoter.neighbors()
    classes = {'TP':0,'TN':0,'FP':0,'FN':0,'NA':0}
    for neighbor in neighbors:
        if 'classification' in neighbor['enhancers']:
            for classification in neighbor['enhancers']['classification']:
                classes[classification]+=1
        else:
            for i in range(0, len(neighbor['enhancers']['local_enhancers'])):
                classes['NA'] += 1
    return(classes)

def total_precision(classes):
    return(classes['TP']/(classes['TP']+classes['FP']))

def total_recall(classes):
    return(classes['TP']/(classes['TP']+classes['FN']))

def total_f1_score(classes):
    precision = classes['TP']/(classes['TP']+classes['FP'])
    if classes['TP']==0:
        recall=0
    else:
        recall = classes['TP']/(classes['TP']+classes['FN'])
    if (precision+recall)==0:
        f1_score=0
    else:
        f1_score = (2*precision*recall)/(precision+recall)
    return(f1_score)

#in this script, first the list of significant loops and their p values is read in
#and subsetted to only include the window around a gene
#next the network with *all?* contacts on it is read in and filtered by a p value
#after which the ABC score for each enhancer that is directly connected to the promoter will be calculated
#and everything else will be set to 0 (below 0.02 would be anyways)
#then we try and recapture the precision-recall curve of ABC and optimize the p value so that the AUC is the greatest
#then later, we will add the ABCD score and see if we can improve the AUC

parser = argparse.ArgumentParser(description='get enhancers')
parser.add_argument('enhancers', type=str, help='1st argument must be enhancers table')
parser.add_argument('--gene_tss', type=str, help='must pass gene_tss file')


#read in classified enhancers with precomputed ABC score
efile = parser.parse_args().enhancers
enhancers = pd.read_csv(efile, header=0, sep = '\t')
#enhancers.rename(columns={'Gene':'TargetGene','Gene TSS':'TargetGeneTSS', 'Activity':'activity_base', 'ABC Score':'ABC.Score', 'Normalized HiC Contacts':'hic_contact_pl_scaled_adj'}, inplace=True)
enhancers.dropna(subset = ['chr','start','end'], inplace=True)
enhancers['start'] = enhancers.start.astype(int)
enhancers['end'] = enhancers.end.astype(int)
enhancers['TargetGeneTSS'] = enhancers.TargetGeneTSS.astype(int)


#read in loops, set relevant params
window = 5000000
resolution = 5000
abc_threshold = 0.02
chromosomes = ['chr10',  'chr12',  'chr19',  'chr3',  'chr8',  'chrX']
gene_tss = {}

#get genes/tss 
gene_tss_file = parser.parse_args().gene_tss
with open(gene_tss_file,'r') as f:
    for line in f:
        chrm, gene, tss = line.strip().split('\t')
        if chrm in chromosomes:
            gene_tss[gene] = [chrm, int(tss)]
#gene_tss = {'BAX': gene_tss['BAX'], 'CALR': gene_tss['CALR']}
#########get gene info
########parser = argparse.ArgumentParser(description='Retrieve Gene')
########parser.add_argument('gene', type=str, help='1st argument must be gene')
########gene = parser.parse_args().gene
########chromosome, tss = gene_tss[gene] 
#########find promoter node
########pnode = '_'.join([chromosome, str(int(np.floor(tss/resolution)*resolution)), str(int(np.floor(tss/resolution)*resolution+resolution))])

#load loops filter for local regions for each gene
local_loops_dict = {}
for gene in gene_tss:
    chromosome, tss = gene_tss[gene]
    if chromosome in chromosomes:
        lbound = tss - window/2
        ubound = tss + window/2
        loops = pd.read_csv('data/fithic_loops/'+chromosome+'/fithic_filtered.bedpe', sep='\t')
        loops['node1'] = loops.chr1.str.cat([loops.start1.astype(int).astype(str),loops.end1.astype(int).astype(str)],sep='_')
        loops['node2'] = loops.chr2.str.cat([loops.start2.astype(int).astype(str),loops.end2.astype(int).astype(str)],sep='_')
        local_loops = loops.loc[(loops['start1']>lbound)&(loops['end1']>lbound)&(loops['start2']>lbound)&(loops['end2']>lbound),]
        local_loops = local_loops.loc[(local_loops['start1']<ubound)&(local_loops['end1']<ubound)&(local_loops['start2']<ubound)&(local_loops['end2']<ubound),]
        local_loops_dict[gene] = deepcopy(local_loops)

#load networks with all contacts
networks = {}
to_remove = []
for gene in gene_tss:
    if os.path.isfile('data/gene_networks/'+gene+'_network.pkl'):
        with open('data/gene_networks/'+gene+'_network.pkl','rb') as f:
            network = pkl.load(f)
            network = network.simplify(multiple=True, combine_edges='first')
            networks[gene] = deepcopy(network)
    else:
        print('Gene: ' +gene+ ' not found in data/gene_networks/')
        to_remove.append(gene)
        #exit()

for gene in set(to_remove):
    del gene_tss[gene]


def objective_function(p_threshold, networks, gene_tss, abc_threshold, ground_truth, chromosomes, resolution, local_loop_dict):
    total_cm = {'TP':0,'TN':0,'FP':0,'FN':0,'NA':0}
    p_threshold = p_threshold[0]
    for gene in gene_tss:
        chromosome, tss = gene_tss[gene]
        if chromosome in chromosomes:
            pnode = '_'.join([chromosome, str(int(np.floor(tss/resolution)*resolution)), str(int(np.floor(tss/resolution)*resolution+resolution))])
            network = deepcopy(networks[gene])
            local_loops = deepcopy(local_loop_dict[gene])
            network = filter_edges_global(network, p_threshold, local_loops, pnode) 
            if network !=False:
                network = estimate_ABC(network,pnode) 
                if network != False:
                    classify_network(gene, ground_truth, network, pnode, abc_threshold)
                    local_cm = network_confusion_matrix(network, pnode)
                    for label in local_cm:
                        total_cm[label] += local_cm[label]
    return([total_f1_score(total_cm),total_precision(total_cm),total_recall(total_cm)])

def output_network(p_threshold, network, local_loops, pnode):
    network = deepcopy(network)
    network = filter_edges_global(network, p_threshold, local_loops, pnode)
    if network != False:
        network = estimate_ABC(network, pnode)
    return(network)

def p_distribution(gene_tss, local_loops_dict):
    for gene in gene_tss:
        local_loops = local_loops_dict[gene]
            

#write filtered networks to file
pcutoff = 8.71e-11
for gene in gene_tss:
    print(gene)
    chromosome, tss = gene_tss[gene]
    pnode = '_'.join([chromosome, str(int(np.floor(tss/resolution)*resolution)), str(int(np.floor(tss/resolution)*resolution+resolution))])
    network = output_network(pcutoff, networks[gene], local_loops_dict[gene], pnode)
    if network != False:
        network = classify_network(gene, enhancers, network, pnode, 0.02)
        if network!=False:
            with open('data/gene_networks_wd/'+gene+'_network.pkl','wb') as f:
                pkl.dump(network, f)
        else:
            print(gene+' error filtering edges')
    else:
        print(gene+' error filtering edges')
exit()

#print(objective_function([0.0005], networks, gene_tss, 0.02, ground_truth, chromosomes, 5000, local_loops_dict))

pvals = np.logspace(-15, -2, endpoint=False, num=100)
f1s = []
precision = []
recall = []
print('iterating through p vals')
for p in pvals:
    f1, prc, rec = objective_function([p], networks, gene_tss, 0.02, enhancers, chromosomes, 5000, local_loops_dict) #enhancers used to be = ground truth
    f1s.append(f1)
    precision.append(prc)
    recall.append(rec)
with open('p_vs_f1_pr.txt','w') as f:
    for i in range(0, len(pvals)):
        f.write(','.join([str(pvals[i]),str(f1s[i]),str(precision[i]),str(recall[i])])+'\n')

#print(pvals)
#print(f1s)
#print(optimize.minimize(objective_function, [1e-5], args=(network, pnode, 0.02, ground_truth), bounds=[[0,0.1]],method='L-BFGS-B'))
exit()
