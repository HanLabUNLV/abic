import pandas as pd
import multiprocessing as mp
import argparse
import os
parser = argparse.ArgumentParser(description='Retrieve Gene')
parser.add_argument('gene', type=str, help='1st argument must be gene')

###
# create promoter chipseq matrix with other script, check matrix for cobound data with this script
###


def detect_cobound_p(index, row, newdata):
    tss = row['tss']
    chrm = row['chr']
    gene = row['gene']
    
    pname = '_'.join([chrm, str(int(tss-500)), str(int(tss+500))])
    prow = newdata.loc[newdata['promoter']==pname]
    sigtf = []
    if len(prow.index) >= 1:
        for tf in tfs:            
            if (prow.iloc[0][tf]==1) and (row[tf]==1):
                sigtf.append([str(index),tf])
    return(sigtf)

gene = parser.parse_args().gene
if os.path.isfile('data/gene_networks_validated_2/'+gene+'_network.pkl'):
    data = pd.read_csv('data/full_feature_matrix.promPOL.subset2.tsv',sep='\t')
    data = data.loc[data['gene']==gene,]
    pdata = pd.read_csv('data/promoter_chipseq_featurematrix.tsv',sep='\t')
    tfs = data.columns[12:-7].tolist()
    cotf = [x + '_p_cobound' for x in tfs]

    results = []
    for index, row in data.iterrows():
        results.extend(detect_cobound_p(index, row, pdata))


    out = open('data/cobound/'+gene+'_cobound_tf','w')
    for result in results:
        if len(result)>0:
            out.write('\t'.join(result)+'\n')
