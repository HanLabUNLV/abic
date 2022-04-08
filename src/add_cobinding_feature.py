import pandas as pd
data = pd.read_csv('data/full_feature_matrix.tsv',sep='\t')
tfs = data.columns[11:].tolist()
cotf = [x + '_p_cobound' for x in tfs]

newdata = pd.concat([data, pd.DataFrame(columns=cotf)])
for col in cotf:
    newdata[col] = 0

for index, row in newdata.iterrows():
    tss = row['tss']
    chrm = row['chr']
    prow = newdata.loc[(newdata['chr']==chrm)&(newdata['start']==tss)&(newdata['stop']==tss) ]
    if len(prow.index) >= 1:
        for tf in tfs:
            if (prow.iloc[0][tf]==1) and (row[tf]==1):
                newdata.loc[index, (tf + '_p_cobound')] = 1
newdata.to_csv('data/full_feature_matrix.coboundp.tsv',sep='\t',index=False)
