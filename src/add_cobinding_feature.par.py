import pandas as pd
import multiprocessing as mp
import argparse
import os
parser = argparse.ArgumentParser(description='Retrieve Gene')
parser.add_argument('gene', type=str, help='1st argument must be gene')

def detect_cobound_p(index, row, newdata):
    tss = row['tss']
    chrm = row['chr']
    prow = newdata.loc[(newdata['chr']==chrm)&(newdata['start']==tss)&(newdata['stop']==tss) ]
    sigtf = []
    if len(prow.index) >= 1:
        for tf in tfs:
            
            if (prow.iloc[0][tf]==1) and (row[tf]==1):
                sigtf.append([str(index),tf])
    return(sigtf)

gene = parser.parse_args().gene
if os.path.isfile('data/gene_networks_wd/'+gene+'_network.pkl'):

    data = pd.read_csv('data/full_feature_matrix.tsv',sep='\t')
    data = data.loc[data['gene']==gene,]
    tfs = data.columns[11:].tolist()
    cotf = [x + '_p_cobound' for x in tfs]

    #newdata = pd.concat([data, pd.DataFrame(columns=cotf)])
    #for col in cotf:
    #    newdata[col] = 0

    #parallelize
    #results = [pool.apply(detect_cobound_p, args=(index, row, newdata.loc[(newdata['chr']==row['chr'])&(newdata['start']==row['tss'])&(newdata['stop']==row['tss'])])) for index, row in newdata.iterrows()]

    results = []
    for index, row in data.iterrows():
        results.extend(detect_cobound_p(index, row, data))


    out = open('data/cobound/'+gene+'_cobound_tf','w')
    for result in results:
        if len(result)>0:
            out.write('\t'.join(result)+'\n')
