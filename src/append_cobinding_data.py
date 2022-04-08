import pandas as pd
from os import listdir as ls

#import data, add columns
data = pd.read_csv('data/full_feature_matrix.subset1.tsv',sep='\t')
tfs = data.columns[11:].tolist()
cotf = [x + '_p_cobound' for x in tfs]

newdata = pd.concat([data, pd.DataFrame(columns=cotf)])
for col in cotf:
    newdata[col] = 0

#aggregate the results of the parallel cobinding code
cobind_dir = 'data/cobound/'
results = []
for fn in ls(cobind_dir):
    with open(cobind_dir+fn,'r') as f:
        lines = f.readlines()
        if len(lines) > 0:
            results.extend([line.strip().split('\t') for line in lines])

#flip 0s to 1s            
for result in results:
   newdata.at[int(result[0]),result[1]+'_p_cobound'] = 1 
newdata.to_csv('data/full_feature_matrix.subset1.coboundp.tsv',sep='\t',index=False)
