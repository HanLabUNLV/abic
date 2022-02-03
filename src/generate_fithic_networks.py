from predictor import *
import pickle as pkl
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Retrieve Gene')
parser.add_argument('gene', type=str, help='1st argument must be gene')

#here's the plan for tomorrow, I'm pretty sure the networking works, but it was taking forever
#what I'm going to do is create an algorithm that begins with the gene, finds the chromosome that its on (from enhancers)
#then, load that bedgraph in, and add an optional argument to populate network that makes a subgraph of the network
#depending on the region of the genome, the network of interactions on a whole chromosome is quite large after all
#then parallelize it by gene, and you can focus on the genes with false negatives

#since the loops are largely the same as the ones in the networks I've already created, and those networks contain all the HiC data already
#I should just use the premade loops and filter out any edges that are not included in the fithic loop calls. 
#this will save time on networking algorithms that I have already run
#if you decide that this is the course of action to take, find out how many loops are not in the network (since we did use a threshold of our own, it may not entirely agree with fithic)

#read in tss to get window around gene
gene_tss_file = 'gene_tss.uniq.tsv'
with open(gene_tss_file, 'r') as f:
    lines = f.readlines()[1:]
gene_tss = {}
for line in lines:
    gene, tss = line.strip().split('\t')
    tss = int(tss)
    gene_tss[gene] = tss

#genes = ['GATA1', 'CCDC26', 'DNASE2', 'FTL', 'KLF1', 'NUCB1', 'FIG4', 'PVT1-TSS1', 'PQBP1', 'PRDX2', 'H1FX', 'MYC', 'JUNB', 'WDR83OS', 'HNRNPA1', 'FUT1', 'DHPS', 'BAX', 'RAE1', 'HBE1', 'CALR', 'HDAC', 'NFE2', 'PLP2', 'RAD23A', 'RPN1'] 

gene = parser.parse_args().gene
try:
    #Load extant gene network
    with open('gene_networks/'+gene+'_network.pkl','rb') as f:
        network = pkl.load(f)
    #find chromosome of gene from network
    chromosome = network.vs['name'][0].split('_')[0]
    #find tss to narrow down range of loops
    tss = gene_tss[gene]
    lbound = tss - 5000000/2
    ubound = tss + 5000000/2

    #filter gene network with bedgraph edges
    bedgraph = 'fithic_loops/'+chromosome+'/fithic_filtered.bedpe'
    filtered_loops = pd.read_csv(bedgraph, sep='\t')
    filtered_loops = filtered_loops.loc[(filtered_loops['start1']>lbound)&(filtered_loops['start2']>lbound)&(filtered_loops['end1']>lbound)&(filtered_loops['end2']>lbound),]
    filtered_loops = filtered_loops.loc[(filtered_loops['start1']<ubound)&(filtered_loops['start2']<ubound)&(filtered_loops['end1']<ubound)&(filtered_loops['end2']<ubound),]
    filtered_loops['node1_id'] = filtered_loops.chr1.str.cat([filtered_loops.start1.astype(str), filtered_loops.end1.astype(str)], sep='_')
    filtered_loops['node2_id'] = filtered_loops.chr2.str.cat([filtered_loops.start2.astype(str), filtered_loops.end2.astype(str)], sep='_')
     
    #identify which edges to keep and lose
    node1s = filtered_loops['node1_id'].tolist()
    node2s = filtered_loops['node2_id'].tolist()
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
            pass
   #print(len(eids_missed))
   #print(len(eids_kept))
   #kept_p = [e[2] if e is not None else 1  for e in eids_kept]
   #missed_p = [e[2] if e is not None else 1 for e in eids_missed]
   #########with open('ABCD_edge_p.txt', 'w') as f:
   #########    f.writelines("\n".join([str(item) for item in kept_p]))
   #########
   #########with open('significant_nonoverlapping_p.txt', 'w') as f:
   #########    f.writelines("\n".join([str(item) for item in missed_p]))
   #print(kept_p)
   #print(missed_p)
   #exit()

    eids_kept = [e[:2] for e in eids_kept]

    final_eids = network.get_eids(eids_kept)
    total_eids = [e.index for e in network.es[:]]
    lost_eids = list(set(total_eids) - set(final_eids))
    #delete edges of lost loops
    network.delete_edges(lost_eids)
    #some nodes are unconnected now, so clean them off
    network.vs.select(_degree=0).delete()

    #add p-value??
    with open('gene_networks_filtered/'+gene+'_network.pkl','wb') as f:
        pkl.dump(network, f)
except Exception as exception:
    print("ERROR FOR GENE: " + gene)
    print(exception)
