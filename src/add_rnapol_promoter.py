#this script will accomplish a few simple things
#it will start by filtering the metadata.tsv file downloaded from ENCODE
#then from those filtered files, it will do a bedtools overlap with Enhancers.txt to generate a binary matrix of overlaps with narrowpeak ChIPSeq data
#if multiple datasets pass the filtering steps, then get a consensus of >=50% of the remaining data
#right now it doesn't take a consensus, just looks for presence/absence in any dataset

import pandas as pd
import os

####POLR2A
####POLR2AphosphoS2
####POLR2AphosphoS5
####POLR2B
####POLR2G
####POLR2H
####POLR3A
####POLR3G


data = pd.read_csv('data/full_feature_matrix.revalidated.final2.tsv',sep='\t', header=0)
####data = data.loc[:,['chr','tss','gene']]
####print(data.shape)
####data = data.drop_duplicates()
####data['start'] = [int(i - 500) for i in data.tss.tolist()]
####data['stop'] = [int(i + 500) for i in data.tss.tolist()]
####data = data[['chr','start','stop','gene','tss']]
####print(data.shape)
####data.to_csv('data/promoter.subset2.bed',sep='\t', index=False)

#pbed = pd.read_csv('data/promoter.subset2.bed',sep='\t')
#load in enhancers
enhancers = []
#made with above commented out section
efile = 'data/promoter.subset2.bed'
gtrans = {}
with open(efile,'r') as f:
    for line in f:
        ename = '_'.join(line.strip().split('\t')[0:3])
        if ename not in enhancers:
            enhancers.append(ename)
            gtrans[ename] = line.strip().split('\t')[3]
#read in metadata filter out crispr experiments
metadata = pd.read_csv('raw_data/ENCODE_ChIP/metadata.tsv',sep='\t')
metadata = metadata.loc[metadata['Biosample genetic modifications categories']!='insertion',]

#filter data?

#gather tf-experiment duplicates only if
TF_experiments = {}
for index, row in metadata.iterrows():
    tf = row['Experiment target'].split('-')[0]
    if tf not in TF_experiments:
        TF_experiments[tf] = [row['File accession']]
    else:
        TF_experiments[tf].append(row['File accession'])

#add cols to data matrix
rnapols = ['POLR2A','POLR2AphosphoS2','POLR2AphosphoS5','POLR2B','POLR2G','POLR3A','POLR3G']
for tf in rnapols:
    data[tf+'_promoter'] = 0


#find overlaps, change 0s to values
for tf in rnapols:

    #init to aggregate experiments 
    gene_pol = {}
    for i in set(data.gene.tolist()):
        gene_pol[i] = []

    for exp in TF_experiments[tf]:
        result = os.system('bedtools intersect -b raw_data/ENCODE_ChIP/'+exp+'.bed -a '+efile+' -wa -wb > data/tmp.bed')
        try:
            ov = pd.read_csv('data/tmp.bed',sep='\t',header=None)
        except pd.errors.EmptyDataError:
            continue
        if len(ov)>0: #for each gene, report last column and append to gene_pol
            #find enhancers, set values to 1
            for zdx, row in ov.iloc[:,[3,11]].iterrows():
                gene, signal = row
                gene_pol[gene].append(signal)
                
    for gene in gene_pol:
        if len(gene_pol[gene])>0:
            data.loc[data['gene']==gene,tf+'_promoter'] = sum(gene_pol[gene])/len(gene_pol[gene])

data.to_csv('data/full_feature_matrix.promPOL.subset2.v2.tsv',sep='\t', index=False)
