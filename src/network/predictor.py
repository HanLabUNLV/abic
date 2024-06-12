import numpy as np
import pandas as pd
from tools import *
import sys, os
import time
import pyranges as pr
from hic import *
from igraph import *
from copy import deepcopy
from scipy.stats import gmean
import pickle as pkl

def make_predictions(chromosome, enhancers, genes, args):
    pred = make_pred_table(chromosome, enhancers, genes, args)
    pred = annotate_predictions(pred, args.tss_slop)
    pred = add_powerlaw_to_predictions(pred, args)

    #if Hi-C directory is not provided, only powerlaw model will be computed
    if args.HiCdir:
        hic_file, hic_norm_file, hic_is_vc = get_hic_file(chromosome, args.HiCdir, hic_type = args.hic_type)
        pred = add_hic_to_enh_gene_table(enhancers, genes, pred, hic_file, hic_norm_file, hic_is_vc, chromosome, args)
        pred = compute_score(pred, [pred['activity_base'], pred['hic_contact_pl_scaled_adj']], "ABC")
        pred = compute_score(pred, [pred['activity_base'], pred['hic_contact_pl_scaled_adj']], "ABCD")
    
    pred = compute_score(pred, [pred['activity_base'], pred['powerlaw_contact_reference']], "powerlaw")

    return pred

def make_pred_table(chromosome, enh, genes, args):
    print('Making putative predictions table...')
    t = time.time()
 
    enh['enh_midpoint'] = (enh['start'] + enh['end'])/2
    enh['enh_idx'] = enh.index
    genes['gene_idx'] = genes.index
    enh_pr = df_to_pyranges(enh)
    genes_pr = df_to_pyranges(genes, start_col = 'TargetGeneTSS', end_col = 'TargetGeneTSS', start_slop=args.window, end_slop = args.window)

    pred = enh_pr.join(genes_pr).df.drop(['Start_b','End_b','chr_b','Chromosome','Start','End'], axis = 1)
    pred['distance'] = abs(pred['enh_midpoint'] - pred['TargetGeneTSS'])
    pred = pred.loc[pred['distance'] < args.window,:] #for backwards compatability

    #without pyranges version
    # else:
    #     enh['temp_merge_key'] = 0
    #     genes['temp_merge_key'] = 0

    #     #Make cartesian product and then subset to EG pairs within window. 
    #     #TO DO: Replace with pyranges equivalent of bedtools intersect or GRanges overlaps 
    #     pred = pd.merge(enh, genes, on = 'temp_merge_key')

    #     pred['enh_midpoint'] = (pred['start'] + pred['end'])/2
    #     pred['distance'] = abs(pred['enh_midpoint'] - pred['TargetGeneTSS'])
    #     pred = pred.loc[pred['distance'] < args.window,:]

    #     print('Done. There are {} putative enhancers for chromosome {}'.format(pred.shape[0], chromosome))
    #     print('Elapsed time: {}'.format(time.time() - t))

    return pred

def add_hic_to_enh_gene_table(enh, genes, pred, hic_file, hic_norm_file, hic_is_vc, chromosome, args):
    print('Begin HiC')
    HiC = load_hic(hic_file = hic_file, 
                    hic_norm_file = hic_norm_file,
                    hic_is_vc = hic_is_vc,
                    hic_type = args.hic_type, 
                    hic_resolution = args.hic_resolution, 
                    tss_hic_contribution = args.tss_hic_contribution, 
                    window = args.window, 
                    min_window = 0, 
                    gamma = args.hic_gamma)

    #Add hic to pred table
    #At this point we have a table where each row is an enhancer/gene pair. 
    #We need to add the corresponding HiC matrix entry.
    #If the HiC is provided in juicebox format (ie constant resolution), then we can just merge using the indices
    #But more generally we do not want to assume constant resolution. In this case hic should be provided in bedpe format
    t = time.time()
    if args.hic_type == "bedpe":
        #Use pyranges to compute overlaps between enhancers/genes and hic bedpe table
        #Consider each range of the hic matrix separately - and merge each range into both enhancers and genes. 
        #Then remerge on hic index

        HiC['hic_idx'] = HiC.index
        hic1 = df_to_pyranges(HiC, start_col='x1', end_col='x2', chr_col='chr1')
        hic2 = df_to_pyranges(HiC, start_col='y1', end_col='y2', chr_col='chr2')

        #Overlap in one direction
        enh_hic1 = df_to_pyranges(enh, start_col = 'enh_midpoint', end_col = 'enh_midpoint', end_slop = 1).join(hic1).df
        genes_hic2 = df_to_pyranges(genes, start_col = 'TargetGeneTSS', end_col = 'TargetGeneTSS', end_slop = 1).join(hic2).df
        ovl12 = enh_hic1[['enh_idx','hic_idx','hic_contact']].merge(genes_hic2[['gene_idx', 'hic_idx']], on = 'hic_idx')

        #Overlap in the other direction
        enh_hic2 = df_to_pyranges(enh, start_col = 'enh_midpoint', end_col = 'enh_midpoint', end_slop = 1).join(hic2).df
        genes_hic1 = df_to_pyranges(genes, start_col = 'TargetGeneTSS', end_col = 'TargetGeneTSS', end_slop = 1).join(hic1).df
        ovl21 = enh_hic2[['enh_idx','hic_idx','hic_contact']].merge(genes_hic1[['gene_idx', 'hic_idx']], on = ['hic_idx'])

        #Concatenate both directions and merge into preditions
        ovl = pd.concat([ovl12, ovl21]).drop_duplicates()
        pred = pred.merge(ovl, on = ['enh_idx', 'gene_idx'], how = 'left')
        pred.fillna(value={'hic_contact' : 0}, inplace=True)
    elif args.hic_type == "juicebox":
        #Merge directly using indices
        #Could also do this by indexing into the sparse matrix (instead of merge) but this seems to be slower
        #Index into sparse matrix
        #pred['hic_contact'] = [HiC[i,j] for (i,j) in pred[['enh_bin','tss_bin']].values.tolist()]
        
        pred['enh_bin'] = np.floor(pred['enh_midpoint'] / args.hic_resolution).astype(int)
        pred['tss_bin'] = np.floor(pred['TargetGeneTSS'] / args.hic_resolution).astype(int)
        if not hic_is_vc:
            #in this case the matrix is upper triangular.
            #
            pred['bin1'] = np.amin(pred[['enh_bin', 'tss_bin']], axis = 1)
            pred['bin2'] = np.amax(pred[['enh_bin', 'tss_bin']], axis = 1)
            pred = pred.merge(HiC, how = 'left', on = ['bin1','bin2'])
            pred.fillna(value={'hic_contact' : 0}, inplace=True)
        else:
            # The matrix is not triangular, its full
            # For VC assume genes correspond to rows and columns to enhancers
            pred = pred.merge(HiC, how = 'left', left_on = ['tss_bin','enh_bin'], right_on=['bin1','bin2'])

        pred.fillna(value={'hic_contact' : 0}, inplace=True)

        # QC juicebox HiC
        pred = qc_hic(pred)

        

    pred.drop(['x1','x2','y1','y2','bin1','bin2','enh_idx','gene_idx','hic_idx','enh_midpoint','tss_bin','enh_bin'], inplace=True, axis = 1, errors='ignore')
        
    print('HiC added to predictions table. Elapsed time: {}'.format(time.time() - t))

    # Add powerlaw scaling
    pred = scale_hic_with_powerlaw(pred, args)

    #Add pseudocount
    pred = add_hic_pseudocount(pred, args)

    print("HiC Complete")
    #print('Elapsed time: {}'.format(time.time() - t))

    return(pred)

def add_hic_ABCD(pairs, hic_file, hic_norm_file, hic_is_vc, chromosome, args):
	#adapted from above
	#need to add contactFrom and contactTo columns, will have to create new rows for many enhancers 
	#in order to get this information, I believe I will need to make this a directed graph toward the promoter. 
	#then I will follow a walking algorithm that will start at the outermost node and take the shortest path (somehow?) 
	#along the way I will visit nodes, each one with edges connecting to and from it
  	#for which I will collect hic calculations using this algorithm to eventually create an ABCD score for each ehancer - gene pair
    HiC = load_hic(hic_file = hic_file, 
        hic_norm_file = hic_norm_file,
        hic_is_vc = hic_is_vc,
        hic_type = args['hic_type'], 
        hic_resolution = args['hic_resolution'], 
        tss_hic_contribution = args['tss_hic_contribution'], 
        window = args['window'], 
        min_window = 0, 
        gamma = args['hic_gamma'])
    pairs['view_midpoint'] = (pairs['viewEnd'] + pairs['viewStart'])/2
    pairs['contact_midpoint'] = (pairs['contactEnd'] + pairs['contactStart'])/2
    pairs['enh_bin'] = np.floor(pairs['view_midpoint'] / args['hic_resolution']).astype(int)
    pairs['tss_bin'] = np.floor(pairs['contact_midpoint'] / args['hic_resolution']).astype(int)
    if not hic_is_vc:
        #in this case the matrix is upper triangular.
        #
        pairs['bin1'] = np.amin(pairs[['enh_bin', 'tss_bin']], axis = 1)
        pairs['bin2'] = np.amax(pairs[['enh_bin', 'tss_bin']], axis = 1)
        pairs = pairs.merge(HiC, how = 'left', on = ['bin1','bin2'])
        pairs.fillna(value={'hic_contact' : 0}, inplace=True)
    else:
        # The matrix is not triangular, its full
        # For VC assume genes correspond to rows and columns to enhancers
        pairs = pairs.merge(HiC, how = 'left', left_on = ['tss_bin','enh_bin'], right_on=['bin1','bin2'])

    pairs.fillna(value={'hic_contact' : 0}, inplace=True)

	# QC juicebox HiC since pairs != enhancers | pred, might need to do this later on in the process after populating the enh dataframe with 
	#pairs = qc_hic(pairs)
    
    return(pairs)

def scale_hic_with_powerlaw(pred, args):
    #Scale hic values to reference powerlaw

    if not args.scale_hic_using_powerlaw:
        pred['hic_contact_pl_scaled'] = pred['hic_contact']
    else:
        pred['hic_contact_pl_scaled'] = pred['hic_contact'] * (pred['powerlaw_contact_reference'] / pred['powerlaw_contact'])

    return(pred)

def add_powerlaw_to_predictions(pred, args):
    pred['powerlaw_contact'] = get_powerlaw_at_distance(pred['distance'].values, args.hic_gamma)
    pred['powerlaw_contact_reference'] = get_powerlaw_at_distance(pred['distance'].values, args.hic_gamma_reference)

    return pred

def add_hic_pseudocount(pred, args):
    # Add a pseudocount based on the powerlaw expected count at a given distance

    powerlaw_fit = get_powerlaw_at_distance(pred['distance'].values, args.hic_gamma)
    powerlaw_fit_at_ref = get_powerlaw_at_distance(args.hic_pseudocount_distance, args.hic_gamma)
    
    pseudocount = np.amin(pd.DataFrame({'a' : powerlaw_fit, 'b' : powerlaw_fit_at_ref}), axis = 1)
    pred['hic_pseudocount'] = pseudocount
    pred['hic_contact_pl_scaled_adj'] = pred['hic_contact_pl_scaled'] + pseudocount

    return(pred)

def qc_hic(pred, threshold = .01):
    # Genes with insufficient hic coverage should get nan'd

    summ = pred.loc[pred['isSelfPromoter'],:].groupby(['TargetGene']).agg({'hic_contact' : 'sum'})
    bad_genes = summ.loc[summ['hic_contact'] < threshold,:].index

    pred.loc[pred['TargetGene'].isin(bad_genes), 'hic_contact'] = np.nan

    return pred

def compute_score(enhancers, product_terms, prefix):

    scores = np.column_stack(product_terms).prod(axis = 1)
    
    #THIS IS WHERE I EDIT THE ABC FORMULA
    if prefix=='ABCD': #Activity by contact distal model
        return(run_ABCD(enhancers, prefix)) #probably need hiccups loops file here too
    else:
        enhancers[prefix + '.Score.Numerator'] = scores
        enhancers[prefix + '.Score'] = enhancers[prefix + '.Score.Numerator'] / enhancers.groupby('TargetGene')[prefix + '.Score.Numerator'].transform('sum')
        return(enhancers)

def run_ABCD(enhancers, loops, chromosome):#, product_terms, prefix):
    import collections
    #populate network with nodes and edges
    #prepare the hic data to be turned into nodes 
    hic_dir = '/data8/han_lab/dbarth/ncbi/public/jonathan/HiC/raw/5kb_resolution_intrachromosomal/'
    hic_resolution = 5000
    hic_type = 'juicebox'
    tss_hic_contribution = 100
    window = 5000000 #size of window around gene TSS to search for enhancers
    hic_gamma = .87
    args = {'hic_dir':hic_dir,'hic_resolution':hic_resolution,'hic_type':hic_type,'tss_hic_contribution':tss_hic_contribution,'window':window,'hic_gamma':hic_gamma}
    
    #hic_file, hic_norm_file, hic_is_vc = get_hic_file(chromosome, args['hic_dir'], hic_type = args['hic_type']) 
    hic_file = '/data8/han_lab/dbarth/ncbi/public/jonathan/HiC/raw/5kb_resolution_intrachromosomal/'+chromosome+'/'+chromosome+'_5kb.RAWobserved'
    hic_norm_file = '/data8/han_lab/dbarth/ncbi/public/jonathan/HiC/raw/5kb_resolution_intrachromosomal/'+chromosome+'/'+chromosome+'_5kb.KRnorm'
    hic_is_vc = True
    hic_threshold = 0.#2.788652e-03 * 0.01 #75% of contacts
    #load HIC matrix
    HiC = load_hic(hic_file = hic_file, 
        hic_norm_file = hic_norm_file,
        hic_is_vc = hic_is_vc,
        hic_type = args['hic_type'], 
        hic_resolution = args['hic_resolution'], 
        tss_hic_contribution = args['tss_hic_contribution'], 
        window = args['window'], 
        min_window = 0, 
        gamma = args['hic_gamma'])


    network = network_from_hic(enhancers, HiC, hic_threshold, hic_resolution, chromosome, window)
    network.simplify()
    with open('intermediate_network.pkl','wb') as f:
        pkl.dump(network, f)
    #detect edges given threshold and list of genes
    #print('Populating Network')
    #network = populate_network(loops)
    #nodes = network.vs['name']

    #detect communities, get membership
    #communities_ig = network.community_fastgreedy()
    #communities = igraph_community_membership(communities_ig, nodes)


    #print('...adding attributes to network')
    #tie activity and enhancer location to nodes
    add_attr_network(network, enhancers)

    with open('intermediate_network_2.pkl','wb') as f:
        pkl.dump(network, f)

    #with open('intermediate_network_2.pkl','rb') as f:
    #    network = pkl.load(f)
    nodes = network.vs['name'] 
    communities_ig = network.community_fastgreedy()
    communities = igraph_community_membership(communities_ig, nodes)
    print(collections.Counter([len(x) for x in communities]))

    

    exit()
    print('identifying valid connections')
    #which enhancer-promoter pairs have a path to them
    valid_connections = get_valid_connections(enhancers, network, communities) 

    with open('intermediate_valid_connections.pkl','wb') as f:
        pkl.dump(valid_connections, f)

    print('calculating all contacts to compute')
    #find contacts within the network to compute
    pairs = contact_network_to_df(network)

    with open('intermediate_pairs_1','wb') as f:
        pkl.dump(pairs, f)
    
    print('computing HiC scores')
    #compute them
    pairs = add_hic_ABCD(pairs, hic_file, hic_norm_file, hic_is_vc, chromosome, args)

    with open('intermediate_pairs_2.pkl','wb') as f:
        pkl.dump(pairs, f)

    print('adding HiC scores to edges')
    #add them to the network
    network = add_contact_network(network, pairs)

    with open('intermediate_network_3.pkl','wb') as f:
        pkl.dump(network, f)

    print('calculating final ABCD score')
    #calculate the final ABCD score, add to enhancers
    enhancers = add_ABCD_score(enhancers, network, valid_connections) 
    
    #print(find_secondary_enhancers(42, [173], network))
    #find enodes and their corresponding elements
    #51221837  51222430
    #print(enhancers[(enhancers['chr']=='chr22') & (enhancers['start']==51221837)]['TargetGeneTSS']-1000)
    #exit()
    #need to translate between loop anchors and enhancer coords to store data
   #enetwork = {}
   #for enode in primary_enhancer_nodes:
   #    for com in reg_communities: 
   #        if enode in com:                
   #            #find promoter through TSS of target gene
   #            chrm, start, end = node_2_coord(enode, enhancers)[0] ###change line below to include chrm when moving past example data
   #            tss = enhancers[(enhancers['start']==start) & (enhancers['chr']==chrm) & (enhancers['end']==end)].TargetGeneTSS.unique()
   #            
   #            pstarts = tss - 1000
   #            pends = tss + 1000

   #            pstart = pstarts[0]
   #            pend = pends[0]

   #            #we have many (100s) genes being mapped to one enhancer (how do they do this?)
   #            #maybe we should pick the ones that are actually connected via a loop?
   #            #print(network.vs()["name"]==nodes)
   #            enetwork[enode] = {'promoter':'_'.join([chrm,str(pstart),str(pend)]), 'secondary_enhancers':find_secondary_enhancers(enode,promoter_nodes,network)}
   #            
   #            
   #enhancers['enh_subnetwork'] = ''
   #missing = 0
   #for e in enetwork:
   #    chrm, start, end = node_2_coord(e, enhancers)[0]
   #    if(len(enhancers[(enhancers['start']==start) & (enhancers['end']==end)] )>0):
   #        enhancers.loc[(enhancers['start']==start) & (enhancers['end']==end),'enh_subnetwork'] = ','.join([node_id_2_node_name(i,nodes)  for i in enetwork[e]['secondary_enhancers']])
   #    else:
   #        missing += 1

   ##enhancers.to_csv('enhancer_buffer_file')
   ##calculate the effect of each subnetwork of enhancers
   #

   ##how many secondary enhancers in enhancers
   #subnetworks = enhancers.enh_subnetwork.unique()[1:]
   #enhancers['ABCD.Score.Numerator'] = 0
   #for subnetwork in subnetworks:
   #    distal_effect = calculate_distal_effect(enhancers, subnetwork)
   #    enhancers.loc[enhancers['enh_subnetwork']==subnetwork,'ABCD.Score.Numerator'] = enhancers.loc[enhancers['enh_subnetwork']==subnetwork,'ABC.Score.Numerator']+distal_effect
   #enhancers['ABCD.Score'] = enhancers['ABCD.Score.Numerator'] / enhancers.groupby('TargetGene')['ABCD.Score.Numerator'].transform('sum')
   #enhancers.to_csv('enhancers_abcd.csv')
    #print(enhancers[enhancers['ABCD.Score']>0])
    #exit()
    return(enhancers)

def network_from_hic(enhancers, HiC, hic_threshold, hic_resolution,chromosome, window):
    genes, genetss = [enhancers.TargetGene.tolist(), enhancers.TargetGeneTSS.tolist()]
    HiC = deepcopy(HiC.loc[HiC['hic_contact']>hic_threshold])
    HiC['chr'] = chromosome
    HiC['viewNodeStart'] = HiC.bin1 * hic_resolution
    HiC['viewNodeEnd'] = (HiC.bin1 + 1) * hic_resolution
    HiC['contactNodeStart'] = HiC.bin2 * hic_resolution
    HiC['contactNodeEnd'] = (HiC.bin2 + 1) * hic_resolution
    HiC['viewNode'] = HiC.chr.str.cat([HiC.viewNodeStart.astype(str), HiC.viewNodeEnd.astype(str)], sep='_')
    HiC['contactNode'] = HiC.chr.str.cat([HiC.contactNodeStart.astype(str), HiC.contactNodeEnd.astype(str)], sep='_')
    HiC['connectionID'] = HiC.viewNode.str.cat(HiC.contactNode, sep=':')
    
    filt_bin1 = list(set(HiC.bin1.tolist()))
    filt_bin2 = list(set(HiC.bin2.tolist()))
    #get network only 5mb around gene loci
    finished_genes = []
    nodes = []
    edges = []
    hics = []
    #for now just to 10 genes, maybe pick 30 to test it on
    for gene in list(set(genes)):
        if gene not in finished_genes:
            tss = genetss[genes.index(gene)]
            lwindow = np.floor((tss - (window/2)) / hic_resolution).astype(int) 
            rwindow = np.floor((tss + (window/2)) / hic_resolution).astype(int) 

            local_nodes = []
            local_edges = []
            for b in range(lwindow, rwindow+1):
                bnode = '_'.join([chromosome, str(b*hic_resolution), str(b*hic_resolution)])
                benh = node_2_coord(bnode, enhancers)
                if len(benh)>0:
                    if (b in filt_bin1) or  (b in filt_bin2):
                        #local_nodes.extend(['_'.join([chromosome, str(b*hic_resolution), str((b+1)*hic_resolution)]),'_'.join([chromosome, str((b-1)*hic_resolution), str(b*hic_resolution)])])
                        hic_edges = HiC.loc[(HiC['bin1']==b) | (HiC['bin2']==b),]
                        views = hic_edges.viewNode.tolist()
                        contacts = hic_edges.contactNode.tolist()
                        hic_scores = hic_edges.hic_contact.tolist()
                        for i in range(0, len(views)):
                            local_edges.append([views[i], contacts[i]])                   
                        hics.extend(hic_scores)
             
            #nodes.extend(local_nodes)
            edges.extend(local_edges)
            finished_genes.append(gene)
    
    nodes = list(set([node for edge in edges for node in edge]))
   
    #filter nodes/edges to overlap with enhancers
    filt_nodes = coord_2_node(enhancers, nodes_2_pynodes(nodes))
    print(filt_nodes)
    filt_edges = []
    filt_hics = []
    for edge in edges:
         if (edge[0] in filt_nodes) and (edge[1] in filt_nodes):
            filt_edges.append(edge)
            filt_hics.append(hics[edges.index(edge)])

    network = Graph()
    #network.add_vertices(nodes)
    #network.add_edges(edges)
    #network.es['hic'] = hics
    network.add_vertices(filt_nodes)
    network.add_edges(filt_edges)
    network.es['hic'] = filt_hics
    return(network)

def validated_network_around_gene(enhancers, gene, valid_loops):
    network = Graph()
    hic_resolution = 5000
    enhancers = deepcopy(enhancers.loc[enhancers['TargetGene']==gene,])
    
    #valid_loops should contain only loops with either a valid enhancer or the promoter at both ends
    #this makes the graph we're generating, extra valid1
    valid_loops['node1'] = valid_loops[['chr1','start1','end1']].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
    valid_loops['node2'] = valid_loops[['chr2','start2','end2']].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)

    print(valid_loops['node1'])
    print(valid_loops['node2'])
    nodes = []
    edges = []
    for idx, row in valid_loops.iterrows():
        node1 = row['node1']
        node2 = row['node2']
        if node1 not in nodes:
            nodes.append(node1)
        if node2 not in nodes:
            nodes.append(node2)
        edges.append([node1, node2])

    network.add_vertices(nodes)
    network.add_edges(edges)
    #now the network has every enhancer and promoter as a node, it's time to add activity, coordinates, and hic to the network so we can produce a picture
    if len([v for v in network.vs]) < 2:
        print(gene)
        print([v for v in network.vs])
        exit('network has less than 2 vertices')
    add_attr_network(network, enhancers, gene)
    hic_dir = 'data/raw_data/hic/5kb_resolution_intrachromosomal/'
    hic_resolution = 5000
    hic_type = 'juicebox'
    tss_hic_contribution = 100
    window = 5000000 #size of window around gene TSS to search for enhancers
    hic_gamma = .87
    args = {'hic_dir':hic_dir,'hic_resolution':hic_resolution,'hic_type':hic_type,'tss_hic_contribution':tss_hic_contribution,'window':window,'hic_gamma':hic_gamma}
    chromosome = enhancers.chr.tolist()[0]
    #hic_file, hic_norm_file, hic_is_vc = get_hic_file(chromosome, args['hic_dir'], hic_type = args['hic_type']) 
    hic_file = 'data/raw_data/hic/5kb_resolution_intrachromosomal/'+chromosome+'/'+chromosome+'_5kb.RAWobserved'
    hic_norm_file = 'data/raw_data/hic/5kb_resolution_intrachromosomal/'+chromosome+'/'+chromosome+'_5kb.KRnorm'
    hic_is_vc = True
    hic_threshold = 0.#2.788652e-03 * 0.01 #75% of contacts
    #load HIC matrix
    HiC = load_hic(hic_file = hic_file,
        hic_norm_file = hic_norm_file,
        hic_is_vc = hic_is_vc,
        hic_type = args['hic_type'],
        hic_resolution = args['hic_resolution'],
        tss_hic_contribution = args['tss_hic_contribution'],
        window = args['window'],
        min_window = 0,
        gamma = args['hic_gamma'])

    #define boundaries to trim HiC
    tss = enhancers.loc[enhancers['TargetGene']==gene,'TargetGeneTSS'].tolist()[0]
    lwindow = int(np.floor(tss - (window/2))/hic_resolution)
    rwindow = int(np.floor(tss + (window/2))/hic_resolution)
    bins = [x for x in range(lwindow, rwindow+1)]
    
    HiC = HiC.loc[(HiC['bin1'].isin(bins) & HiC['bin2'].isin(bins)),]

    #add useful cols
    HiC['chr'] = chromosome
    HiC['viewNodeStart'] = HiC.bin1 * hic_resolution
    HiC['viewNodeEnd'] = (HiC.bin1 + 1) * hic_resolution
    HiC['contactNodeStart'] = HiC.bin2 * hic_resolution
    HiC['contactNodeEnd'] = (HiC.bin2 + 1) * hic_resolution
    HiC['viewNode'] = HiC.chr.str.cat([HiC.viewNodeStart.astype(str), HiC.viewNodeEnd.astype(str)], sep='_')
    HiC['contactNode'] = HiC.chr.str.cat([HiC.contactNodeStart.astype(str), HiC.contactNodeEnd.astype(str)], sep='_')
    HiC['connectionID'] = HiC.viewNode.str.cat(HiC.contactNode, sep=':')
    #now iter over edges, get all pairs of connections between bins you need
    hics = []
    for edge in edges:
        h = HiC.loc[((HiC['viewNode']==edge[0]) & (HiC['contactNode']==edge[1])),]
        hlist = h.hic_contact.tolist()
        if len(hlist)==1:
            hics.append(hlist[0])
        elif len(hlist)>1:
            hics.append(h['hic_contact'].mean())
        else:
            hics.append(0)

    network.es['contact']=hics

    #with open('chr19/gene_networks/'+gene+'_network.pkl','wb') as f:
    #    pkl.dump(network, f)
 
    #visual_style = {}
    #visual_style["vertex_size"] = 10
    #visual_style["edge_width"] = hics
    #plot(network, target='chr19/'+gene+'_reg_network.png', **visual_style)
    return(network)




def network_around_gene(enhancers, gene):
    network = Graph()
    hic_resolution = 5000
    enhancers = deepcopy(enhancers.loc[enhancers['TargetGene']==gene,])
    enhancers['nodeStart'] = np.floor(enhancers['start'] / hic_resolution)*hic_resolution
    enhancers['nodeEnd'] = (np.floor(enhancers['start'] / hic_resolution) + 1) * hic_resolution
    enhancers['enhNode'] = enhancers.chr.str.cat([enhancers.nodeStart.astype(int).astype(str), enhancers.nodeEnd.astype(int).astype(str)], sep='_')
    
    enhancers['pnodeStart'] = np.floor(enhancers['TargetGeneTSS'] / hic_resolution)*hic_resolution
    enhancers['pnodeEnd'] = (np.floor(enhancers['TargetGeneTSS'] / hic_resolution) + 1) * hic_resolution
    enhancers['pNode'] = enhancers.chr.str.cat([enhancers.pnodeStart.astype(int).astype(str), enhancers.pnodeEnd.astype(int).astype(str)], sep='_')

    if len(enhancers.index) < 2:
        return(False)

    enodes = list(set(enhancers.enhNode.tolist()))
    pnodes = list(set(enhancers.pNode.tolist()))
    nodes = enodes + pnodes
    #print(enhancers['start'])
    #print(len(enodes+pnodes))
    #print(enodes[0:5])
    edges = []
    for i in nodes:
        for j in nodes:
            if i!=j:
                edges.append([i,j])
    network.add_vertices(nodes)
    network.add_edges(edges)
    #now the network has every enhancer and promoter as a node, it's time to add activity, coordinates, and hic to the network so we can produce a picture
    if len([v for v in network.vs]) < 2:
        print(gene)
        print([v for v in network.vs])
        exit('network has less than 2 vertices')
    add_attr_network(network, enhancers, gene)
    hic_dir = 'data/raw_data/hic/5kb_resolution_intrachromosomal/'
    hic_resolution = 5000
    hic_type = 'juicebox'
    tss_hic_contribution = 100
    window = 5000000 #size of window around gene TSS to search for enhancers
    hic_gamma = .87
    args = {'hic_dir':hic_dir,'hic_resolution':hic_resolution,'hic_type':hic_type,'tss_hic_contribution':tss_hic_contribution,'window':window,'hic_gamma':hic_gamma}
    chromosome = enhancers.chr.tolist()[0]
    #hic_file, hic_norm_file, hic_is_vc = get_hic_file(chromosome, args['hic_dir'], hic_type = args['hic_type']) 
    hic_file = 'data/raw_data/hic/5kb_resolution_intrachromosomal/'+chromosome+'/'+chromosome+'_5kb.RAWobserved'
    hic_norm_file = 'data/raw_data/hic/5kb_resolution_intrachromosomal/'+chromosome+'/'+chromosome+'_5kb.KRnorm'
    hic_is_vc = True
    hic_threshold = 0. #2.788652e-03 * 0.01 #75% of contacts
    #load HIC matrix
    HiC = load_hic(hic_file = hic_file,
        hic_norm_file = hic_norm_file,
        hic_is_vc = hic_is_vc,
        hic_type = args['hic_type'],
        hic_resolution = args['hic_resolution'],
        tss_hic_contribution = args['tss_hic_contribution'],
        window = args['window'],
        min_window = 0,
        gamma = args['hic_gamma'])

    #define boundaries to trim HiC
    tss = enhancers.loc[enhancers['TargetGene']==gene,'TargetGeneTSS'].tolist()[0]
    lwindow = int(np.floor(tss - (window/2))/hic_resolution)
    rwindow = int(np.floor(tss + (window/2))/hic_resolution)
    bins = [x for x in range(lwindow, rwindow+1)]
    
    HiC = HiC.loc[(HiC['bin1'].isin(bins) & HiC['bin2'].isin(bins)),]

    #add useful cols
    HiC['chr'] = chromosome
    HiC['viewNodeStart'] = HiC.bin1 * hic_resolution
    HiC['viewNodeEnd'] = (HiC.bin1 + 1) * hic_resolution
    HiC['contactNodeStart'] = HiC.bin2 * hic_resolution
    HiC['contactNodeEnd'] = (HiC.bin2 + 1) * hic_resolution
    HiC['viewNode'] = HiC.chr.str.cat([HiC.viewNodeStart.astype(str), HiC.viewNodeEnd.astype(str)], sep='_')
    HiC['contactNode'] = HiC.chr.str.cat([HiC.contactNodeStart.astype(str), HiC.contactNodeEnd.astype(str)], sep='_')
    HiC['connectionID'] = HiC.viewNode.str.cat(HiC.contactNode, sep=':')
    #now iter over edges, get all pairs of connections between bins you need
    hics = []
    for edge in edges:
        h = HiC.loc[((HiC['viewNode']==edge[0]) & (HiC['contactNode']==edge[1])),]
        hlist = h.hic_contact.tolist()
        if len(hlist)==1:
            hics.append(hlist[0])
        elif len(hlist)>1:
            hics.append(h['hic_contact'].mean())
        else:
            hics.append(0)

    network.es['contact']=hics

    #with open('chr19/gene_networks/'+gene+'_network.pkl','wb') as f:
    #    pkl.dump(network, f)
 
    #visual_style = {}
    #visual_style["vertex_size"] = 10
    #visual_style["edge_width"] = hics
    #plot(network, target='chr19/'+gene+'_reg_network.png', **visual_style)
    return(network)
    
def calculate_dijkstra_contact(network, promoter):
    Cd = []
    Ceg = []
    pid = network.vs.find(promoter)
    try:    
        max_contact = max(network.es['contact']) + 0.001
    except ValueError as e:
        print(network.es['contact'])
        print('contact not intact, none found')
        exit()

    dist = [max_contact - c for c in network.es['contact'] ]
    network.es['dist'] = dist
    for v in network.vs:
        vid = network.vs.find(v['name'])
        paths = network.get_all_shortest_paths(v, promoter, weights='dist')
        if len(paths)>0:
            path = paths[0]
            contacts = []
            for i in range(0, len(path) -1):
                contacts.append(network.es.find(network.get_eid(path[i], path[i+1]))['contact'])
            Cd.append(mean(contacts))
            try:
                Ceg.append(network.es.find(network.get_eid(vid, pid))['contact'])
            except: #no edge found above
                Ceg.append(0)

        else: #no path found??i
            print('no path')
            Cd.append(0)
            try:
                Ceg.append(network.es.find(network.get_eid(vid, pid))['contact'])
            except: #no edge found above
                Ceg.append(0)
    network.vs['Cd'] = Cd
    network.vs['Ceg'] = Ceg    

def add_validated_attr(network, validation):
    #TODO: add effect size and significance -- filter out unvalidated? just mark with NA? 
    #initialize validation pyranges object 
    validation = validation.rename(columns={'chrEnhancer':'Chromosome','startEnhancer':'Start','endEnhancer':'End'})
    valpr = pr.PyRanges(validation)
    #iter through nodes
    for v in network.vs:
        #iter through enhancers within node
        fx_sz = []
        sig = []
        for enh in v['enhancers']['local_enhancers']:
            chrm, start, stop = enh
            prtmp = pr.from_dict({'Chromosome':[chrm], 'Start':[start], 'End':[stop]})
            ov = valpr.overlap(prtmp)
            if ov.__len__()>0:
                fx_sz.append(ov.as_df().EffectSize.values[0])
                sig.append(ov.as_df().Significant.values[0])
            else:
                fx_sz.append('NA')
                sig.append('NA')
        network.vs.find(v['name'])['effect_size'] = fx_sz
        network.vs.find(v['name'])['sig'] = sig

        
    return network


def add_attr_network(network, enhancers, gene):
    vs = network.vs['name']
    for i in vs:
        local_enhancer_attr = {'local_enhancers':node_2_coord(i, enhancers)}
        activity = []
        abc_scores = []
        for enh in local_enhancer_attr['local_enhancers']:
            chrm, start, end = enh
            if start!=end:
                abc_scores.append(enhancers.loc[(enhancers['chr']==chrm) & (enhancers['start']==start) & (enhancers['end']==end)& (enhancers['TargetGene']==gene),'ABC.Score'].tolist()[0])
                activity.append(enhancers.loc[(enhancers['chr']==chrm) & (enhancers['start']==start) & (enhancers['end']==end)& (enhancers['TargetGene']==gene),'activity_base'].mean())
            else: #is a promoter, activity not taken into account?? could use promoter activity quantile to get estimate
                abc_scores.append(0)
                activity.append(0)
        local_enhancer_attr['activity'] = activity
        local_enhancer_attr['abc_score'] = abc_scores
        if len(local_enhancer_attr['local_enhancers'])>0:
            print(local_enhancer_attr)
        network.vs.find(i)['enhancers'] = local_enhancer_attr

    #add a mean activity to the node as an attribute
    np.seterr(all='raise')
    activity = []
    for v in network.vs:
        if v['enhancers'] is not None:
            if (len(v['enhancers']['activity'])>0) and (sum(v['enhancers']['activity'])>0):
                try:
                    activity.append(gmean(v['enhancers']['activity']))
                except:
                    activity.append(mean(v['enhancers']['activity']))
            else:
                activity.append(0)
        else:
            activity.append(0)
    network.vs['activity'] = activity

def contact_network_to_df(network):
    pairs = pd.DataFrame(columns = ['chr', 'viewStart', 'viewEnd', 'contactStart', 'contactEnd', 'view_node','contact_node'])
    for i in network.vs['name']:
        node = network.vs.find(i)
        viewpoint_enh = node['enhancers']['local_enhancers']      
        for vp in viewpoint_enh:
            for nb in node.neighbors():
                for con in nb['enhancers']['local_enhancers']:
                    #add the values to the pairs dataframe, then pass that into the HIC calculator
                    pairs = pairs.append({'chr':vp[0], 'viewStart':vp[1], 'viewEnd':vp[2], 'contactStart':con[1], 'contactEnd':con[2], 'view_node':node['name'],'contact_node':nb['name']}, ignore_index=True)
    pairs['edgeID'] = pairs.view_node.str.cat(pairs.contact_node, sep=':')  #pairs['view_node'] + ':' pairs['contact_node']
    pairs['viewID'] = pairs.chr.str.cat([pairs.viewStart.astype(str), pairs.viewEnd.astype(str)], sep='_')
    pairs['contactID'] = pairs.chr.str.cat([pairs.contactStart.astype(str), pairs.contactEnd.astype(str)], sep='_')
    pairs['connectionID'] = pairs.viewID.str.cat(pairs.contactID, sep=':') 
    return pairs

def calculate_distal_contact(enh, prm, path, network):
    contacts = []
    for i in range(len(path)-1):
        view_node = path[i]   
        contact_node = path[i+1]
        edge = network.es.select(_source=view_node, _target=contact_node)
        contacts.append(edge['avg_contact'][0])
    if ((len(contacts)>0) and (len(contacts)>0)):
        return(gmean(contacts))
    else:
        return(0)

def get_valid_connections(enhancers, network, communities):
    pynodes = nodes_2_pynodes(network.vs['name'])
    #takes a year to finish
    enhancers = deepcopy(enhancers.loc[enhancers['class']!='promoter',])
    enhancers['enhID'] = enhancers.chr.str.cat([enhancers.start.astype(str), enhancers.end.astype(str)], sep='_')
    enhancers['promoterID'] = enhancers.chr.str.cat([enhancers.TargetGeneTSS.astype(str), enhancers.TargetGeneTSS.astype(str)], sep='_')
    enhIDs = enhancers.enhID.tolist()
    promoterIDs = enhancers.promoterID.tolist()
    valid_connections = []
    noloop_enh = []

    for i in range(0, len(enhIDs)):
        if enhIDs[i] not in noloop_enh:        
            enh_coord = coord_2_node(enhIDs[i], pynodes)
            prom_coord = coord_2_node(promoterIDs[i], pynodes)
            path = [[]]
            if len(enh_coord)>0 and len(prom_coord)>0:    
                enh_df = enh_coord.filt_nodes = filt_nodes.to_as_df()
                prom_df = enh_coord.as_df()
                enh_node_id = list(set(enh_df.Chromosome.str.cat([enh_df.Start.astype(str),enh_df.End.astype(str)], sep='_').tolist()))[0]
                prom_node_id = list(set(prom_df.Chromosome.str.cat([prom_df.Start.astype(str),prom_df.End.astype(str)], sep='_').tolist()))[0]
                for com in communities:
                    if (enh_node_id in com) and (prom_node_id in com):
                        path = network.get_shortest_paths(v=enh_node_id, to=prom_node_id)
                        print(path)
            else:
                noloop_enh.append(enhIDs[i])

            if len(path[0])<=1:
                print([enh_coord, prom_coord])
    print(noloop_enh)

    return(valid_connections)

def add_contact_network(network, pairs):
    for e in network.es:
        v1 = network.vs.find(e.source)['name']
        v2 = network.vs.find(e.target)['name']
        edge_data = pairs.loc[((pairs['view_node']==v1)&(pairs['contact_node']==v2))]# & pairs['contact_node']==v2) | (pairs['view_node']==v2 & pairs['contact_node']==v1)]
        connections = {'connectionID':edge_data['connectionID'].tolist(), 'contact':edge_data['hic_contact'].tolist()}
        e['connections'] = connections
        if len(connections['contact'])>0:
            e['avg_contact'] = sum(connections['contact'])
        else:
            e['avg_contact'] = 0
    return(network)

def populate_network(bedpe):
    nodes = []
    edges = []
    with open(bedpe, 'r') as f:
        f.readline() #header
        f.readline()
        #convert bedpe format to edge format
        for i in f:
            viewnode = '_'.join(i.strip().split('\t')[0:3])
            contactnode = '_'.join(i.strip().split('\t')[3:6])
            nodes.append(viewnode)
            nodes.append(contactnode)
            edges.append([viewnode,contactnode])

    #print('19_13090000_13100000' in nodes)
    #print('19_13200000_13210000' in nodes)
    #exit()
    for i in range(0, len(nodes)):
        nodes[i] = 'chr'+nodes[i]

    for i in range(0,len(edges)):
        edges[i][0] = 'chr'+edges[i][0]
        edges[i][1] = 'chr'+edges[i][1]


    reg_network = Graph()
    reg_network.add_vertices(nodes)
    reg_network.add_edges(edges)

    #e1 = edge_from_nodeid(reg_network, ['chr19_13090000_13100000','chr19_13200000_13210000'])
    
    #detect communities, get membership
    #reg_communities_ig = reg_network.community_fastgreedy()
    #reg_communities = igraph_community_membership(reg_communities_ig, nodes)

    #view, contact = edge_get_nodeids(reg_network, e1)
    #for com in reg_communities:
    #    if (view in com) and (contact in com): 
    #        print(com)
    #        exit()
    #exit()
    return(reg_network) 

def edge_from_nodeid(network, nodes):
    viewnode, contactnode = nodes
    viewidx = network.vs['name'].index(viewnode)
    contactidx = network.vs['name'].index(contactnode)
    e1 = network.es[network.get_eid(viewidx, contactidx)]
    return(e1)

def edge_get_nodeids(network, edge):
    return([network.vs['name'][edge.source], network.vs['name'][edge.target]])


def calculate_distal_effect(enhancers, subnetwork):
    #not implemented
    subnetwork = subnetwork.split(',')
    subnet_activity = []
    subnet_contact = []
    for node_name in subnetwork:    
        #look up node in enhancers to find rows with relevant data
        node_coord = node_2_coord(node_name, enhancers)
        if len(node_coord)>0:
            uniq_node_coord = list(set([tuple(x) for x in node_coord]))
            node_coord = uniq_node_coord[0]
            chrm, start, end = node_coord

            print([chrm, start, end])
            #pred['activity_base'], pred['hic_contact_pl_scaled_adj']        
            #add activity data
            subnet_activity.append(enhancers.loc[(enhancers['chr']==chrm) & (enhancers['start']==start) & (enhancers['end']==end),'activity_base'].mean())               
             #add contact data
            subnet_contact.append(enhancers.loc[(enhancers['chr']==chrm) & (enhancers['start']==start) & (enhancers['end']==end),'hic_contact_pl_scaled_adj'].mean())
        else:
            print([node_name,node_coord])
            
    if len(subnet_activity)!=0 and len(subnet_contact)!=0:
        subnet_activity_mean = gmean(subnet_activity)
        subnet_contact_mean = gmean(subnet_contact)
        return(subnet_activity_mean*subnet_contact_mean)
    else:
        return(0)

def add_ABCD_score(enhancers, network, valid_connections):
    enhancers['hic.ABCD'] = enhancers['hic_contact_pl_scaled_adj']
    enhancers['geodesic_dist'] = 0
    pynodes = nodes_2_pynodes(network.vs['name'])
    for ep in valid_connections:
        enh, prm = ep
        chrm, enh_start, enh_end = enh.split('_')
        tss = prm.split('_')[1]
        enh_df = coord_2_node(enh, pynodes).as_df()
        enh_node = enh_df.Chromosome.str.cat([enh_df.Start.astype(str), enh_df.End.astype(str)], sep='_').tolist()[0]
        prm_df = coord_2_node(prm, pynodes).as_df()
        prm_node = prm_df.Chromosome.str.cat([prm_df.Start.astype(str), prm_df.End.astype(str)], sep='_').tolist()[0]
        path = network.get_shortest_paths(v=enh_node, to=prm_node)
        cnx1 = ':'.join([prm,enh])
        cnx2 = ':'.join([enh,prm])
        enhancers.loc[(enhancers['chr']==chrm) & (enhancers['start']==int(enh_start)) & (enhancers['end']==int(enh_end)) & (enhancers['TargetGeneTSS']==int(tss)),'geodesic_dist']=len(path[0])-1
        print(path[0])
        #what I need is to walk the path of the network and get all avg_contact for each edge along the way
        if len(path[0])>1:
            dc = calculate_distal_contact(enh, prm, path[0],network)
            if dc > enhancers.loc[(enhancers['chr']==chrm) & (enhancers['start']==int(enh_start)) & (enhancers['end']==int(enh_end)) & (enhancers['TargetGeneTSS']==int(tss)),'hic.ABCD']:
                #then find the entry in enhancers and add the distal contact to those who need it
                enhancers.loc[(enhancers['chr']==chrm) & (enhancers['start']==int(enh_start)) & (enhancers['end']==int(enh_end)) & (enhancers['TargetGeneTSS']==int(tss)),'hic.ABCD'] = dc
                          
    enhancers['ABCD.Score.Numerator'] = np.column_stack([enhancers['activity_base'], enhancers['hic.ABCD']]).prod(axis = 1)
    enhancers['ABCD.Score'] = enhancers['ABCD.Score.Numerator'] / enhancers.groupby('Gene')['ABCD.Score.Numerator'].transform('sum')

    return(enhancers)

def node_name_2_node_id(node, nodes):
    return(nodes.index(node))

def node_id_2_node_name(node, nodes):
    return(nodes[node])

def find_secondary_enhancers(enhancer, promoters, network):
    cnetwork = deepcopy(network)
    nodes = cnetwork.vs()["name"]
    #first find promoters in enhancer community, detach them from all neighbors
    #then find the subcomponent that enhancer falls in
    community = cnetwork.subcomponent(enhancer, mode='all')
    for node in community:
        if node in promoters:
            promoter = node
            pneighbors = cnetwork.neighbors(promoter)
            for neighbor in pneighbors:
                cnetwork.delete_edges((neighbor, promoter))
    secondary_enhancers = cnetwork.subcomponent(enhancer,mode='all')
    return(secondary_enhancers)

def igraph_community_membership(communities, nodes):
    cluster = communities.as_clustering()
    membership = []
    for i in range(0,len(cluster)):
        members = []
        for j in range(0, len(cluster[i])):
            members.append(nodes[cluster[i][j]])
        membership.append(members)
    return(membership)

def nodes_2_pynodes(nodes):
    chrms, starts, ends = [[],[],[]]
    for i in nodes:
        chrm, start, end = i.split('_')
        chrms.append(chrm)
        starts.append(int(float(start)))
        ends.append(int(float(end)))
    return(pr.from_dict({'Chromosome': chrms, 'Start':starts, 'End':ends}))

def coord_2_node(data, loop_anchors):
    #loop_anchors = pr.read_bed('loop_anchors.bed')
    if isinstance(data, str):
        chrm, start, end = data.split('_')
        data = pr.from_dict({'Chromosome':[chrm], 'Start':[int(float(start))], 'End':[int(float(end))]})
        return(loop_anchors.overlap(data))
    primary_enhancers = data.rename(columns = {'chr':'Chromosome', 'start':'Start','end':'End'}, inplace = False) #convert to pyranges for overlap
    primary_enhancer_loops = loop_anchors.overlap(pr.PyRanges(primary_enhancers))
    node_df = primary_enhancer_loops.as_df()
    if not node_df.empty:
        node_df['node_id'] = node_df.Chromosome.str.cat([node_df.Start.astype(str), node_df.End.astype(str)], sep='_')
        nodes = node_df.node_id.tolist()
    else:
        nodes=[]
    return nodes

def node_2_coord(nodename, enhancers):
    targets = deepcopy(enhancers)
    if isinstance(nodename, str):
        chrm, start, end = nodename.split('_')
        data = pr.from_dict({'Chromosome':[chrm], 'Start':[start], 'End':[end]})
        targets = targets.rename(columns = {'chr':'Chromosome', 'start':'Start','end':'End'}, inplace = False)
        targets = pr.PyRanges(targets)
        node_names = targets.overlap(data) #can we get a link between these files?

        #also do promoters such that start=end (to identify them)
        promoters = deepcopy(enhancers)
        promoters['start'] = promoters['TargetGeneTSS']
        promoters['end'] = promoters['TargetGeneTSS']
        promoters = promoters.rename(columns = {'chr':'Chromosome', 'start':'Start','end':'End'}, inplace = False)
        promoters['promoterID'] = promoters.Chromosome.str.cat([promoters.Start.astype(str), promoters.End.astype(str)], sep='_')
        promoters = pr.PyRanges(promoters)
        node_names_p = promoters.overlap(data)
        

        node_df = node_names.as_df()
        node_df_p = node_names_p.as_df()

        #if len(node_names)<1:            
        #    return([])
        #if promoter overlap found
        if not node_df_p.empty:
            if not node_df.empty:
                chrs = node_df['Chromosome'].tolist()
                starts = node_df['Start'].tolist()
                ends = node_df['End'].tolist()
                coords = [[chrs[i],starts[i],ends[i]] for i in range(0,len(chrs))]
            else:
                coords = []

            promoterIDs = list(set(node_df_p['promoterID'].tolist())) #make sure this is added at the beginning of the process
            for pid in promoterIDs:
                i = pid.split('_')
                coords.append([i[0], int(float(i[1])), int(float(i[2]))])      

        #else, just enhancers
        elif not node_df.empty:
            node_df = node_names.as_df()
            chrs = node_df['Chromosome'].tolist()
            starts = node_df['Start'].tolist()
            ends = node_df['End'].tolist()
            coords = [[chrs[i],starts[i],ends[i]] for i in range(0,len(chrs))]
        else:
            return([])
        return list(set([tuple(x) for x in coords]))
    elif isinstance(nodename, list):
        #doesnt work yet.. pyranges will not pass any information from the overlapping dataframe
        #the answer to the question above and below is NO UNFORTUNATELY
        chrms = []
        starts = []
        ends = []
        for i in nodename:
            chrm, start, end = i.split('_')
            chrms.append(chrm)
            starts.append(start)
            ends.append(end)
        data = pr.from_dict({'Chromosome':chrms, 'Start':starts, 'End':ends})

        targets = targets.rename(columns = {'chr':'Chromosome', 'start':'Start','end':'End'}, inplace = False)
        targets['enh_coords_id'] = targets['Chromosome'] + '_' + targets['Start'].astype(str) +'_'+ targets['End'].astype(str)
        targets = pr.PyRanges(targets)
        node_names = targets.overlap(data) #can we get a link between these files?


        promoters = deepcopy(enhancers)
        promoters['start'] = promoters['TargetGeneTSS']
        promoters['end'] = promoters['TargetGeneTSS']
        promoters = promoters.rename(columns = {'chr':'Chromosome', 'start':'Start','end':'End'}, inplace = False)
        promoters = pr.PyRanges(promoters)
        node_names_p = promoters.overlap(data)
        

        if len(node_names.as_df().Start.tolist() + node_names_p.as_df().Start.tolist())<1:            
            return([])
        node_df = node_names.as_df()
        node_df_p = node_names_p.as_df()

        #if promoter overlap found
        if len(node_df_p.Chromosome.tolist())>0:
            chrs = node_df['Chromosome'].tolist() + node_df_p['Chromosome'].tolist()
            starts = node_df['Start'].tolist() + node_df_p['Start'].tolist()
            ends = node_df['End'].tolist() + node_df_p['End'].tolist()
        #else, just enhancers
        else:
            node_df = node_names.as_df()
            chrs = node_df['Chromosome'].tolist()
            starts = node_df['Start'].tolist()
            ends = node_df['End'].tolist()

        coords = [[chrs[i],int(starts[i]),int(ends[i])] for i in range(0,len(chrs))]

        return coords


def annotate_predictions(pred, tss_slop=500):
    #TO DO: Add is self genic
    pred['isSelfPromoter'] = np.logical_and.reduce((pred['class'] == 'promoter' , pred.start - tss_slop < pred.TargetGeneTSS, pred.end + tss_slop > pred.TargetGeneTSS))

    return(pred)

def make_gene_prediction_stats(pred, args):
    summ1 = pred.groupby(['chr','TargetGene','TargetGeneTSS']).agg({'TargetGeneIsExpressed' : lambda x: set(x).pop(), args.score_column : lambda x: all(np.isnan(x)) ,  'name' : 'count'})
    summ1.columns = ['geneIsExpressed', 'geneFailed','nEnhancersConsidered']

    summ2 = pred.loc[pred['class'] != 'promoter',:].groupby(['chr','TargetGene','TargetGeneTSS']).agg({args.score_column : lambda x: sum(x > args.threshold)})
    summ2.columns = ['nDistalEnhancersPredicted']
    summ1 = summ1.merge(summ2, left_index=True, right_index=True)

    summ1.to_csv(os.path.join(args.outdir, "GenePredictionStats.txt"), sep="\t", index=True)
