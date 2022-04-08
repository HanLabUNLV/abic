import pandas as pd
import os
#this script will take an input of the entire feature matrix, then output a version where all of the 1s are looked up in the original chipseq data to get the corresponding continuous signal

data1 = pd.read_csv('data/full_feature_matrix.coboundp.dataset1.tsv',sep='\t', header=0)
data2 = pd.read_csv('data/full_feature_matrix.coboundp.merged.validated.tsv',sep='\t', header=0)
data1['dataset'] = 'fulco'
data2['dataset'] = 'gasperini'

#for some reason MCM5 didnt run on the new dataset, so remove that col
data1 = data1.loc[:,data1.columns != 'MCM5']
data1 = data1.loc[:,data1.columns != 'MCM5_p_cobound']

#put them together, free memory
data = pd.concat([data1,data2])
data1 = ''
data2 = ''

tf_2_file = pd.read_csv('raw_data/ENCODE_ChIP/filtered_experiment_targets.txt',sep='\t',header=0)
#remove '-organism' from tf name
tf_2_file['Experiment target'] = tf_2_file['Experiment target'].str.split('-').str[0]

featured_tfs = data.columns[11:321].tolist()
#print(featured_tfs)
for tf in featured_tfs:
    exps = tf_2_file.loc[tf_2_file['Experiment target']==tf,'File accession'].values

    pos_enhancers = data.loc[data[tf]==1,['chr','start','stop',tf]]
    #create tmp bedfile of these enhancers
    pos_enhancers['start'] = pos_enhancers['start'].astype(int).astype(str)
    pos_enhancers['stop'] = pos_enhancers['stop'].astype(int).astype(str)
    pos_enhancers['bed'] = pos_enhancers[['chr','start','stop']].agg('\t'.join, axis=1)
    lines = [i+'\n' for i in pos_enhancers['bed'].tolist()]
    if len(lines)==0:
            continue
    else:
        with open('data/tmp_tf.bed','w') as f:
            f.writelines(lines)

    #init pandas df to hold all values from all experiments
    colnames = ['chr','start','stop']
    colnames.extend(exps)
    #print(colnames)
    total_data = pd.DataFrame(columns = ['chr','start','stop'].extend(exps))
    total_data[['chr','start','stop']] = pos_enhancers[['chr','start','stop']]
    total_data[exps] = None
    total_data.drop_duplicates(inplace=True)
    total_data.set_axis(colnames, axis=1, inplace=True)
    total_data['start'] = total_data['start'].astype(int)
    total_data['stop'] = total_data['stop'].astype(int)

    #print(total_data)
    for exp in exps:
        #identify enh w overlap (1 in tf column)
        fname = 'raw_data/ENCODE_ChIP/'+ exp +'.bed'
        os.system('bedtools intersect -b '+fname+' -a data/tmp_tf.bed -wa -wb > data/tmp_results.bed')
        try:
            result = pd.read_csv('data/tmp_results.bed',sep='\t',header=None)[[0,1,2,3,4,5,9]]
        except pd.errors.EmptyDataError:
            continue
        result = result.drop_duplicates()
        for idx, row in result.iterrows():
            chrm = row[0]
            start = row[1]
            stop = row[2]
            signal = row[9]
            total_data.loc[((total_data['chr']==chrm)&(total_data['start']==start)&(total_data['stop']==stop)),exp] = signal

    total_data['avg'] = total_data[exps].mean(axis=1)
    #print(data.loc[data[tf]>1])
    for idx, row in total_data.iterrows():
        chrm = row['chr']
        start = row['start']
        stop = row['stop']
        signal = row['avg']
        data.loc[((data['chr']==chrm)&(data['start']==start)&(data['stop']==stop)),tf] = signal
    
    #print(data.loc[data[tf]>1,tf])
data['start'] = data['start'].astype(int)
data['stop'] = data['stop'].astype(int)
data['tss'] = data['tss'].astype(int)

data.to_csv('data/full_feature_matrix.continuous.total_merged.tsv',sep='\t',index=False)

