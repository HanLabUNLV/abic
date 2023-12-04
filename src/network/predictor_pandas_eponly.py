import numpy as np
import pandas as pd
from tools import *
import sys, os
import time
import pyranges as pr
from hic import *
from igraph import *
import dask.dataframe as dd
import networkx as nx
import itertools


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

    enhancers[prefix + '.Score.Numerator'] = scores
    enhancers[prefix + '.Score'] = enhancers[prefix + '.Score.Numerator'] / enhancers.groupby('TargetGene')[prefix + '.Score.Numerator'].transform('sum')

    return(enhancers)

def make_gene_prediction_stats(pred, args):
    summ1 = pred.groupby(['chr','TargetGene','TargetGeneTSS']).agg({'TargetGeneIsExpressed' : lambda x: set(x).pop(), args.score_column : lambda x: all(np.isnan(x)) ,  'name' : 'count'})
    summ1.columns = ['geneIsExpressed', 'geneFailed','nEnhancersConsidered']

    summ2 = pred.loc[pred['class'] != 'promoter',:].groupby(['chr','TargetGene','TargetGeneTSS']).agg({args.score_column : lambda x: sum(x > args.threshold)})
    summ2.columns = ['nDistalEnhancersPredicted']
    summ1 = summ1.merge(summ2, left_index=True, right_index=True)

    summ1.to_csv(os.path.join(args.outdir, "GenePredictionStats.txt"), sep="\t", index=True)



def network_from_hic(chromosome, args):
    #(enh, genes, pred, hic_file, hic_norm_file, hic_is_vc, chromosome, args):
    tmp = time.time()
    hic_file, hic_norm_file, hic_is_vc = get_hic_file(chromosome, args.HiCdir, hic_type = args.hic_type)
    #hic_file = 'raw_data/hic/5kb_resolution_intrachromosomal/'+chromosome+'/'+chromosome+'_5kb.RAWobserved'
    #hic_norm_file = 'raw_data/hic/5kb_resolution_intrachromosomal/'+chromosome+'/'+chromosome+'_5kb.KRnorm'
    #hic_is_vc = True
    #hic_type = 'juicebox'
    #hic_resolution = 5000
    #tss_hic_contribution = 100 
    #hic_gamma = 0.87
    #window = 5000000 #size of window around gene TSS to search for enhancers
    hic_type = args.hic_type
    hic_resolution = args.hic_resolution
    tss_hic_contribution = args.tss_hic_contribution 
    hic_gamma = args.hic_gamma
    window = args.window #5000000 #size of window around gene TSS to search for enhancers
    print('Begin HiC')
    HiC = load_hic(hic_file = hic_file, 
                    hic_norm_file = hic_norm_file,
                    hic_is_vc = hic_is_vc,
                    hic_type = hic_type, 
                    hic_resolution = hic_resolution, 
                    tss_hic_contribution = tss_hic_contribution, 
                    window = window, 
                    min_window = 0, 
                    gamma = hic_gamma)


    HiC['hic_contact'] = pd.to_numeric(HiC['hic_contact'], downcast='float')
    #print(HiC.info())
    #HiC.to_csv("HiC.csv")

    print("load HiC:", time.time()-tmp)
    tmp = time.time()

    edgelist = HiC
    edgelist = edgelist.rename(columns={"bin1": "source", "bin2": "target", "hic_contact": "weight"})
    edgelist["type"] = np.repeat(["contact"], [len(edgelist)], axis=0)

    maxbin = HiC[['bin1', 'bin2']].max().max()
    vertices_hic = pd.DataFrame(index=[i for i in range(maxbin+1)])
    #vertices_hic["type"] = np.repeat(["hic"], [len(vertices_hic)], axis=0)
    vertices_hic.to_csv(os.path.join(args.outdir, "vertices_hic."+chromosome+".txt"), sep="\t", index=True)
    edgelist.to_csv(os.path.join(args.outdir, "edgelist_hic."+chromosome+".txt"), sep="\t", index=True)

    print("hic to pandas edgelist:", time.time()-tmp)
    tmp = time.time()
    return(edgelist)


def network_from_gene_enhancer(chromosome, edgelist_hic, args):

    genes_file = args.genes
    enhancers_file = args.enhancers
    #genes_file = "./ABC_output/Neighborhoods/GeneList.txt"
    #enhancers_file = "./ABC_output/Neighborhoods/EnhancerList.txt"
    expression_cutoff = args.expression_cutoff # 1
    promoter_activity_quantile_cutoff = args.promoter_activity_quantile_cutoff #0.4
    window = args.window #5000000
    hic_resolution = args.hic_resolution #5000
    #hic_is_vc = args.hic_is_vc  #True

    print("reading genes") 
    genes = pd.read_csv(genes_file, sep = "\t") 
    genes = determine_expressed_genes(genes, expression_cutoff, promoter_activity_quantile_cutoff)
    genes = genes.loc[:,['chr', 'name', 'strand', 'symbol','tss','Expression','Expression.quantile', 'PromoterActivityQuantile','isExpressed']]
    genes.columns = ['chr','TargetGene', 'TargetGeneStrand', 'TargetGeneSymbol', 'TargetGeneTSS', 'TargetGeneExpression', 'TargetGeneExpressionQuantile', 'TargetGenePromoterActivityQuantile','TargetGeneIsExpressed']
    genes = genes.loc[genes['chr'] == chromosome]

    print("reading enhancers")
    enhancers_full = pd.read_csv(enhancers_file, sep = "\t") 
    #TO DO  
    #Think about which columns to include 
    enhancers = enhancers_full.loc[:,['chr','start','end','name','class', 'promoterSymbol', 'genicSymbol', 'normalized_h3K27ac', 'normalized_dhs', 'activity_base']]
    enhancers = enhancers.loc[enhancers['chr'] == chromosome]

    print('connecting elements to places..')
    t = time.time()

    enhancers['enh_idx'] = enhancers.index
    enhancers['id'] = enhancers['name']
    enhancers['pos'] = (enhancers['start'] + enhancers['end'])/2
    enhancers['type'] = enhancers['class']
    genes['gene_idx'] = genes.index
    genes['id'] = genes['TargetGene']
    genes['pos'] = genes['TargetGeneTSS']
    genes['type'] = np.repeat(["TSS"], [genes.shape[0]], axis=0)
    enhancers_pr = df_to_pyranges(enhancers)
    genes_pr = df_to_pyranges(genes, start_col = 'TargetGeneTSS', end_col = 'TargetGeneTSS', start_slop=window, end_slop = window)

    pred = enhancers_pr.join(genes_pr).df.drop(['Start_b','End_b','chr_b','Chromosome','Start','End'], axis = 1)
    pred['distance'] = abs(pred['pos'] - pred['TargetGeneTSS'])
    pred = pred.loc[pred['distance'] < window,:] #for backwards compatability

    # create edges between enhancer/promoters and hic windows 
    #edgelist_ep = pd.concat([enhancers_pr[["id","pos", "type"]].as_df(),genes_pr[["id","pos", "type"]].as_df()])
    #edgelist_ep = enhancers[["id","pos", "type"]]
    edgelist_ep_bin = np.floor(enhancers['pos'] / hic_resolution).astype(int)
    # create vertices for enhancers and promoters
    maxbin = edgelist_hic[['source', 'target']].max().max()
    vertices_ep = pd.DataFrame(index=[i for i in range(maxbin+1,maxbin+1+len(enhancers))])
    vertices_ep["type"] = list(enhancers["type"])
    edgelist_ep_vid = vertices_ep.index
    #edgelist_TSS = genes[["id","pos", "type"]]
    edgelist_TSS_bin = np.floor(genes['pos'] / hic_resolution).astype(int)
    # create vertices for enhancers and promoters
    maxbin = edgelist_hic[['source', 'target']].max().max()
    vertices_TSS = pd.DataFrame(index=[i for i in range(maxbin+1+len(enhancers),maxbin+1+len(enhancers)+len(genes))])
    vertices_TSS["type"] = list(genes["type"])
    edgelist_TSS_vid = vertices_TSS.index


    # bind features to vertices
    vertices_ep = pd.concat([vertices_ep.reset_index(), enhancers[["normalized_h3K27ac", "normalized_dhs", "activity_base", "id"]].reset_index(drop=True)], axis=1)      
    vertices_ep.to_csv(os.path.join(args.outdir, "vertices_ep_hic."+chromosome+".txt"), sep="\t", index=False)
    vertices_TSS = pd.concat([vertices_TSS.reset_index(), genes[["TargetGeneExpression", "TargetGeneExpressionQuantile", "TargetGenePromoterActivityQuantile", "TargetGeneIsExpressed", "id"]].reset_index(drop=True)], axis=1)      
    vertices_TSS.to_csv(os.path.join(args.outdir, "vertices_TSS_hic."+chromosome+".txt"), sep="\t", index=False)

    # create edgelist pandas and save
    edgelist_ep_weight = np.repeat([1], [len(enhancers)], axis=0)
    edgelist_ep_type = np.repeat(["place"], [len(enhancers)], axis=0)
    edgelist_TSS_weight = np.repeat([1], [len(genes)], axis=0)
    edgelist_TSS_type = np.repeat(["place"], [len(genes)], axis=0)

    edgelist_ep = pd.DataFrame(data = {'source': edgelist_ep_bin, 'target': edgelist_ep_vid, 'weight': edgelist_ep_weight, 'type': edgelist_ep_type})
    edgelist_TSS = pd.DataFrame(data = {'source': edgelist_TSS_bin, 'target': edgelist_TSS_vid, 'weight': edgelist_TSS_weight, 'type': edgelist_TSS_type})
    edgelist = pd.concat([edgelist_hic, edgelist_ep, edgelist_TSS], ignore_index=True)

    edgelist_ep.to_csv(os.path.join(args.outdir, "edgelist_ep_hic."+chromosome+".txt"), sep="\t", index=True)
    edgelist_TSS.to_csv(os.path.join(args.outdir, "edgelist_TSS_hic."+chromosome+".txt"), sep="\t", index=True)
    edgelist.to_csv(os.path.join(args.outdir, "edgelist_ep_TSS_hic."+chromosome+".txt"), sep="\t", index=True)

    return edgelist


def network_remove_hic(chromosome, edgelist_ep, args):

    print("reading network") 
    vertices_hic = pd.read_csv(os.path.join(args.outdir, "vertices_hic."+chromosome+".txt"), sep="\t", index_col=0)
    vertices_ep = pd.read_csv(os.path.join(args.outdir, "vertices_ep_hic."+chromosome+".txt"), sep="\t", index_col=0)
    vertices_TSS = pd.read_csv(os.path.join(args.outdir, "vertices_TSS_hic."+chromosome+".txt"), sep="\t", index_col=0)
    vertices_elements = pd.concat([vertices_ep, vertices_TSS], sort='False')
    edgelist_hic = pd.read_csv(os.path.join(args.outdir, "edgelist_hic."+chromosome+".txt"), sep="\t", index_col=0)
    edgelist_ep = pd.read_csv(os.path.join(args.outdir, "edgelist_ep_hic."+chromosome+".txt"), sep="\t", index_col=0)
    edgelist_TSS = pd.read_csv(os.path.join(args.outdir, "edgelist_TSS_hic."+chromosome+".txt"), sep="\t", index_col=0)
    edgelist_elements = pd.concat([edgelist_ep, edgelist_TSS], sort='True')

    print("leave hic bins with elements")
    hic_bins_occupied = set(edgelist_elements['source'].unique()) 
    vertices_hic_new = vertices_hic.loc[vertices_hic.index.isin(sorted(hic_bins_occupied))].copy()
    edgelist_hic_new = edgelist_hic.loc[edgelist_hic['source'].isin(hic_bins_occupied) & edgelist_hic['target'].isin(hic_bins_occupied)].copy()

    t = time.time()

    print('connecting complete graphs based on within hic bin relationship..')
    first_of_pairs = list()
    second_of_pairs = list()
    weights = list()
    for i, row in vertices_hic_new.iterrows():
      # find all elements connected to hic bin i
      element_ids = edgelist_elements.loc[edgelist_elements['source'] == i,'target']
      # create pairs from element list n choose 2
      pairs = list(itertools.combinations(element_ids, 2))
      if pairs:
        first_of_pairs.extend([x[0] for x in pairs])
        second_of_pairs.extend([x[1] for x in pairs])
        # find diagonal hic value
        weightval = edgelist_hic.loc[(edgelist_hic['source']== i ) & (edgelist_hic['target'] == i),'weight' ].to_list()
        if len(weightval) == 0: 
          weightval = [None]
        weight_rep_arr = np.repeat(weightval, [len(pairs)], axis=0)
        weights.extend(weight_rep_arr)

    #weights = np.repeat([1], [len(first_of_pairs)], axis=0)
    types = np.repeat(["within"], [len(first_of_pairs)], axis=0)
    #edgelist_elements_within = pd.DataFrame(data = {'source': first_of_pairs, 'target': second_of_pairs, 'weight': weights, 'type': types})
    edgelist_elements_within = pd.DataFrame(data = {'source': first_of_pairs, 'target': second_of_pairs})
    edgelist_elements_within['weight'] = weights
    edgelist_elements_within['type'] = types


    print('connecting edges based on between hic bin contact..')
    first_of_pairs = list()
    second_of_pairs = list()
    weights = list()
    for i, row in edgelist_hic_new.iterrows():
      element_ids_bin1 = edgelist_elements.loc[edgelist_elements['source'] == row['source'],'target']
      element_ids_bin2 = edgelist_elements.loc[edgelist_elements['source'] == row['target'],'target']
      pairs = list(itertools.product(element_ids_bin1, element_ids_bin2))
      if pairs:
        first_of_pairs.extend([x[0] for x in pairs])
        second_of_pairs.extend([x[1] for x in pairs])
        weights.extend(np.repeat([row['weight']], [len(pairs)], axis=0))

    types = np.repeat(["between"], [len(first_of_pairs)], axis=0)
    #edgelist_elements_between = pd.DataFrame(data = {'source': first_of_pairs, 'target': second_of_pairs, 'weight': weights, 'type': types})
    edgelist_elements_between = pd.DataFrame(data = {'source': first_of_pairs, 'target': second_of_pairs})
    edgelist_elements_between['weight'] = weights
    edgelist_elements_between['type'] = types

    edgelist_elements_new = pd.concat([edgelist_elements_within,edgelist_elements_between], ignore_index=False)

##############################
    vertices_elements.to_csv(os.path.join(args.outdir, "vertices_ep_TSS."+chromosome+".txt"), sep="\t", index=True)
    edgelist_elements_new.to_csv(os.path.join(args.outdir, "edgelist_ep_TSS."+chromosome+".txt"), sep="\t", index=True)

    return edgelist_elements_new


