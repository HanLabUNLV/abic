from predictor import *
import pyranges as pr
import pandas as pd
import os.path
import argparse
import time
import joblib as jl

#this script will generate the networks in a more stringent fashion, by only considering edges that had fithic fdr < 0.01 and filtering out any enhancers not validated in gasperini
tstart = time.time()
resolution = 5000
#get gene from cmdline
parser = argparse.ArgumentParser(description='Retrieve Gene')
parser.add_argument('gene', type=str, help='1st argument must be gene')

#load in tss
gene_tss_file = 'data/gene_tss.gas.long.tsv'
genes = {}
with open(gene_tss_file,'r') as f:
    for line in f:
        chrm, gene, tss = line.strip().split('\t')
        genes[gene] = [chrm,tss]
chrm, tss = ['','']
valid_chr = ['chr1','chr10','chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr19','chr2','chr20','chr21','chr22','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chrX']

#get gene from cmdline
gene = parser.parse_args().gene

#make sure it's in the tss file, otherwise we can't locate the promoter
if gene not in genes:
    exit(gene+' is not in gene_tss file: '+gene_tss_file)

#otherwise, get chromosome, tss
chrm = genes[gene][0]
tss = int(genes[gene][1])

#load in all validation data
validation = pd.read_csv('data/Gasperini2019.at_scale_screen.cand_enhancer_x_exprsd_genes.200503.csv',quotechar='"')
#trim validation data for enhancers validated for this gene only
validation = validation.loc[validation['GeneSymbol']==gene,].copy()

#add node / bin values to validation
validation['bin1'] = np.floor(validation['startEnhancer']/resolution)*resolution
validation['bin2'] = np.floor(validation['startEnhancer']/resolution)*resolution+resolution
validation['bin1'] = validation['bin1'].astype(int)
validation['bin2'] = validation['bin2'].astype(int)

#load in chromosome loops
loops = pd.read_csv('data/fithic_loops/'+chrm+'/fithic_filtered.bedpe',sep='\t')
#filter loops to contain only fdr (q value) < 0.01 (and have a window encompassing our validation enhancers?)
loops = loops.loc[loops['q-value']<0.01,].copy()

#filter loops by validation

validated_loops = pd.DataFrame(columns=loops.columns.tolist().extend(['sig','effect_size']))
bin1s = validation.bin1.tolist()
bin1s.append(int(np.floor(tss/resolution)*resolution)) #also need to include the promoter node as a valid connection, duh
for idx, row in validation.iterrows():
    bin1 = row['bin1']
    bin2 = row['bin2']
    
    #get loops with bins that encompass this validated enhancer on either side
    forward_loops = loops.loc[(loops['start1']==bin1),].copy()
    backward_loops = loops.loc[(loops['start2']==bin1),].copy()

    #iter through both forward and backward loops, check for other validated enhancers
    for idx2, row2 in forward_loops.iterrows():
        if row2['start2'] in bin1s:
            #that means one bin (start1) encompasses a validated enhancer, and the looped bin (start2) also encompasses a validated enhancer
            row2['sig'] = row['Significant']
            row2['effect_size'] = row['EffectSize']
            validated_loops = validated_loops.append(row2, ignore_index=True)
    for idx2, row2 in backward_loops.iterrows():
        if row2['start1'] in bin1s:
            row2['sig'] = row['Significant']
            row2['effect_size'] = row['EffectSize']
            #that means one bin (start2) encompasses a validated enhancer, and the looped bin (start1) also encompasses a validated enhancer
            validated_loops = validated_loops.append(row2, ignore_index=True)

if len(validated_loops)==0:
    exit('No enhancers found for gene: ' + gene)

validated_loops['start1'] = validated_loops['start1'].astype(int)
validated_loops['end1'] = validated_loops['end1'].astype(int)
validated_loops['start2'] = validated_loops['start2'].astype(int)
validated_loops['end2'] = validated_loops['end2'].astype(int)

#read in pre-calculated ABC data
enhancers = pd.read_csv('data/enhancers.gas.class.tsv', header=0, sep = '\t')
#filter by gene
enhancers = enhancers.loc[enhancers['TargetGene']==gene,]
#then filter for chromosome, etc
enhancers.dropna(subset = ['chr','start','end'], inplace=True)
enhancers['start'] = enhancers.start.astype(int)
enhancers['end'] = enhancers.end.astype(int)
enhancers['TargetGeneTSS'] = enhancers.TargetGeneTSS.astype(int)


#next up, fix the validated_network_around_gene() function to only include validated_loops
if genes[gene][0] in valid_chr:
    print(gene)
    network = validated_network_around_gene(enhancers, gene, validated_loops)
    if network!=False:
        network = add_validated_attr(network, validation)
        #NOTE TO FUTURE SELF: copy this and src/predictor.py back over to /data8/
        with open('data/gene_networks_validated_2/'+gene+'_network.pkl','wb') as f:
            jl.dump(network, f)

tend = time.time()
print('Execution time: ' + str(tend - tstart))

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
