import pandas as pd
data = pd.read_csv('data/full_feature_matrix.revalidated.final.tsv',sep='\t')

idx_to_del = []
for idx, row in data.iterrows():
    if idx not in idx_to_del:
        tmp = data.loc[(data['chr']==row['chr'])&(data['start']==row['start'])&(data['stop']==row['stop'])&(data['gene']==row['gene']),]
        if len(tmp)>1:
            for i in tmp.index.to_list()[1:]:
                if i not in idx_to_del:
                    idx_to_del.append(i) 
print(len(idx_to_del))
data.drop(index=idx_to_del,inplace=True)

#make some other minor adjustments to differences in syntax
data.loc[data['abc_score'].isna(), 'abc_score'] = 0
data.loc[data['role'].isna(), 'role'] = 'None'
data.drop(columns=['classification'], inplace=True)

data.to_csv('data/full_feature_matrix.revalidated.final2.tsv',sep='\t', index=False)
