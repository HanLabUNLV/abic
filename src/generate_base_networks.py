from predictor import *
import pandas as pd
import os.path
import argparse

parser = argparse.ArgumentParser(description='Retrieve Gene')
parser.add_argument('gene', type=str, help='1st argument must be gene')

#first, read in pre-calculated ABC data
enhancers = pd.read_csv('data/enhancers.gas.class.tsv', header=0, sep = '\t')
#enhancers.rename(columns={'Gene':'TargetGene','Gene TSS':'TargetGeneTSS', 'Activity':'activity_base', 'ABC Score':'ABC.Score', 'Normalized HiC Contacts':'hic_contact_pl_scaled_adj'}, inplace=True)
#then filter for chromosome, etc
enhancers.dropna(subset = ['chr','start','end'], inplace=True)
enhancers['start'] = enhancers.start.astype(int)
enhancers['end'] = enhancers.end.astype(int)
enhancers['TargetGeneTSS'] = enhancers.TargetGeneTSS.astype(int)

#genes = ['PQBP1', 'PRDX2', 'H1FX', 'MYC', 'JUNB', 'WDR83OS', 'HNRNPA1', 'FUT1', 'DHPS', 'BAX', 'RAE1', 'HBE1',  'HDAC6', 'NFE2', 'PLP2', 'RAD23A', 'RPN1'] #list of genes that have false negatives (from enhancer_classification.py)
#get tss and chr for gene
genes = {}
with open('data/gene_tss.uniq.tsv','r') as f:
    for line in f:
        chrm, gene, tss = line.strip().split('\t')
        genes[gene] = [chrm,tss]
valid_chr = ['chr10',  'chr12',  'chr19',  'chr3',  'chr8',  'chrX']

#get gene from cmdline
gene = parser.parse_args().gene
if genes[gene][0] in valid_chr:
    print(gene)
    network = network_around_gene(enhancers, gene)
    if network!=False:
        with open('data/gene_networks/'+gene+'_network.pkl','wb') as f:
            pkl.dump(network, f)

############################################################################
# calculate a distance along the contact network, not useful here really   #
############################################################################
#        chromosome = enhancers.loc[enhancers['TargetGene']==gene, 'chr'].tolist()[0]
#        tss = enhancers.loc[enhancers['TargetGene']==gene, 'TargetGeneTSS'].tolist()[0]
#        hic_resolution = 5000
#        promoter_node = '_'.join([chromosome,str(int(np.floor(tss/hic_resolution)*hic_resolution)), str(int(np.floor(tss/hic_resolution)*hic_resolution + hic_resolution))])
#
#        calculate_dijkstra_contact(network, promoter_node)
#
#
#        print(gene)
#        with open('data/gene_networks/'+gene+'_network.pkl','wb') as f:
#            pkl.dump(network, f)
