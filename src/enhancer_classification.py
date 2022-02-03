import pandas as pd

abc_threshold = 0.02
enhancers = pd.read_csv('ABC_perturbed_K562_EGpairs.csv', header=0, sep = ',')
enhancers.dropna(subset = ['chr','start','end'], inplace=True)
enhancers['start'] = enhancers.start.astype(int)
enhancers['end'] = enhancers.end.astype(int)
#enhancers['Gene TSS'] = enhancers.TSS.astype(int)

enhancers['pred_sig'] = 0
enhancers.loc[enhancers['ABC Score']>abc_threshold,'pred_sig'] = 1

enhancers['classification'] = ''
#TP
enhancers.loc[(enhancers['Significant']==1) & (enhancers['pred_sig']==1), 'classification'] = 'TP'
#FP
enhancers.loc[(enhancers['Significant']==0) & (enhancers['pred_sig']==1), 'classification'] = 'FP'
#TN
enhancers.loc[(enhancers['Significant']==0) & (enhancers['pred_sig']==0), 'classification'] = 'TN'
#FN
enhancers.loc[(enhancers['Significant']==1) & (enhancers['pred_sig']==0), 'classification'] = 'FN'

 
#print(len(enhancers.loc[enhancers['classification']=='FN', 'Gene'].tolist()))
#enhancers.to_csv('ABC_perturbed_K562_EGpairs.classified.csv', index=False, sep=',')

#only keep the chr, start, end, gene, and classification
enhancers.rename(columns={'Gene':'TargetGene'}, inplace=True)
enhancers.drop(enhancers.columns.difference(['chr','start','end','TargetGene','classification']), 1, inplace=True)


#that classified the enhancers from the study, we need to overlap them with our enhancers to add the classification column to our predicted enhancers
pred_enhancers = pd.read_csv('/data8/han_lab/dbarth/ncbi/public/jonathan/han_rep/Predictions/EnhancerPredictionsAllPutative.txt', header=0, sep='\t')

print('ground_truth_rows')
print(enhancers.shape[0])
print('predict_rows')
print(pred_enhancers.shape[0])

intersect = pred_enhancers.merge(enhancers, how='inner', on=['chr','start','end','TargetGene'])

print(intersect)
print(intersect.shape[0])

intersect.to_csv('EnhancerPredictionsAllPutative.classified.txt',sep='\t', index=False)
