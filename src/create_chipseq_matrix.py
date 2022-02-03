#this script will accomplish a few simple things
#it will start by filtering the metadata.tsv file downloaded from ENCODE
#then from those filtered files, it will do a bedtools overlap with Enhancers.txt to generate a binary matrix of overlaps with narrowpeak ChIPSeq data
#if multiple datasets pass the filtering steps, then get a consensus of >=50% of the remaining data
#right now it doesn't take a consensus, just looks for presence/absence in any dataset

import pandas as pd
import os


#load in enhancers
enhancers = []
with open('ENCODE_ChIP/PutativeEnhancer.bed','r') as f:
    for line in f:
        enhancers.append('_'.join(line.strip().split('\t')))

#read in metadata filter out crispr experiments
metadata = pd.read_csv('ENCODE_ChIP/metadata.tsv',sep='\t')
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
#create matrix
tf_matrix = pd.DataFrame({'enhancer':enhancers})
for tf in TF_experiments:
    tf_matrix[tf] = 0

#find overlaps, change 0s to 1s
for tf in TF_experiments:
    for exp in TF_experiments[tf]:
        result = os.system('bedtools intersect -b ENCODE_ChIP/'+exp+'.bed -a ENCODE_ChIP/PutativeEnhancer.bed -u > tmp.bed')
        with open('tmp.bed','r') as f:
            enh_names = ['_'.join(l.strip().split('\t')) for l in  f.readlines()]
        if len(enh_names)>0:
            #find enhancers, set values to 1
            tf_matrix.loc[tf_matrix['enhancer'].isin(enh_names),tf] = 1

tf_matrix.to_csv('enhancer_chipseq_featurematrix.tsv',sep='\t')
