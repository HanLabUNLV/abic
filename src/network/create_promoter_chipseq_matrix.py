import pandas as pd
import os


#load in enhancers, just promoters for this script... not sure why it didn't work in the first place but whatever...
enhancers = []

#promoters too
pfile = 'data/promoters.bed'
with open(pfile,'r') as f:
    for line in f:
        chrm, start, end, gene = line.strip().split('\t')
        pname = '_'.join([chrm, start, end])
        if pname not in enhancers:
            enhancers.append(pname)

#read in metadata filter out crispr experiments
metadata = pd.read_csv('raw_data/ENCODE_ChIP/metadata.tsv',sep='\t')
metadata = metadata.loc[metadata['Biosample genetic modifications categories']!='insertion',]

#gather tf-experiment duplicates only if
TF_experiments = {}
for index, row in metadata.iterrows():
    tf = row['Experiment target'].split('-')[0]
    if tf not in TF_experiments:
        TF_experiments[tf] = [row['File accession']]
    else:
        TF_experiments[tf].append(row['File accession'])

#create matrix
tf_matrix = pd.DataFrame({'promoter':enhancers})
for tf in TF_experiments:
    tf_matrix[tf] = 0

#find overlaps, change 0s to 1s
for tf in TF_experiments:
    for exp in TF_experiments[tf]:
        result = os.system('bedtools intersect -b raw_data/ENCODE_ChIP/'+exp+'.bed -a '+pfile+' -u > data/tmp.bed')
        with open('data/tmp.bed','r') as f:
            enh_names = ['_'.join(l.strip().split('\t')[0:3]) for l in  f.readlines()]
        if len(enh_names)>0:
            #find enhancers, set values to 1
            tf_matrix.loc[tf_matrix['promoter'].isin(enh_names),tf] = 1

tf_matrix.to_csv('data/promoter_chipseq_featurematrix.tsv',sep='\t', index=False)
