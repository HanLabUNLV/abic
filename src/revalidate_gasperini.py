import numpy as np
import pandas as pd
import pyranges as pr
import pickle as pkl

valid = pd.read_csv('data/Gasperini2019.at_scale_screen.cand_enhancer_x_exprsd_genes.200503.csv',quotechar='"')
enhancers = pd.read_csv('data/full_feature_matrix.coboundp.merged.tsv', sep='\t')
valid = valid.rename(columns={'chrEnhancer':'Chromosome','startEnhancer':'Start', 'endEnhancer':'End','GeneSymbol':'gene'})

#enhancers  = enhancers.rename(columns={'end':'stop'})
#s2a = pd.read_csv('data/s2a.bed', sep='\t', header=None)
#s2a = s2a.rename(columns={0: 'Chromosome', 1:'Start', 2:'End'})
#s2b = pd.read_csv('data/s2b.bed', sep='\t',header=None)
#s2b = s2b.rename(columns={0: 'Chromosome', 1:'Start', 2:'End',3:'gene'})

#use pyranges to get overlap, only enhancers that were tested

edf = enhancers.loc[:,['chr','start','stop','gene']]
edf.dropna(axis=0, how='any', inplace=True)
epr = pr.PyRanges(edf.rename(columns={'chr':'Chromosome','start':'Start','stop':'End'}))

valpr = pr.PyRanges(valid)
#s2apr

#s2bpr = pr.PyRanges(s2b)

es2a = epr.overlap(valpr).as_df().rename(columns={'Chromosome':'chr','Start':'start','End':'stop'})
#es2b = epr.overlap(s2bpr).as_df().rename(columns={'Chromosome':'chr','Start':'start','End':'stop'})

#how many of the enhancer-gene pairs in gasperini do we have in our matrix?
gas_in_us = 0
us_in_gas = 0
gas_not_in_us = 0
gas_missing_eg = []
us_not_in_gas = 0
keep_idx = []
for idx, row in valid.iterrows(): 
    chrm = row['Chromosome']
    start = row['Start']
    end = row['End']
    gene = row['gene']

    #convert to pyranges, overlap 
    tpr = pr.from_dict({'Chromosome':[chrm], 'Start':[start], 'End':[end]})
    ov = epr.overlap(tpr)
    if len(ov) > 0:
        glist = ov.as_df().gene.tolist()
        if gene in glist:
            gas_in_us += 1
            keep_idx.append(idx)
        else:
            gas_not_in_us += 1
            gas_missing_eg.append([chrm, start, end, gene])
    else:
        gas_missing_eg.append([chrm, start, end, gene])
        gas_not_in_us += 1

lines = []
for i in gas_missing_eg:
    lines.append('\t'.join([str(j) for j in i])+'\n')

with open('data/gasperini_missing_eg_pairs.tsv','w') as f:
    f.writelines(lines)

missing_found = 0
eg_pairs_found = []
resolution = 5000
for i in gas_missing_eg:
    chrm, start, stop, gene = i
    node = '_'.join([chrm, str(int(np.floor(start/resolution)*resolution)), str(int(np.floor(start/resolution)*resolution+resolution))])
    try: 
        network = pkl.load(open('data/gene_networks/'+gene+'_network.pkl','rb'))
    except:
        continue
    if node in network.vs['name']:
        missing_found += 1        
        eg_pairs_found.append(i)

print('Number of Gasperini EG pairs in our data: ' + str(us_in_gas))
print('Number of Gasperini EG pairs NOT in our data: ' + str(us_not_in_gas))
print('Total: ' + str(len(enhancers)))

print('Number of missing EG pairs found in unpruned networks: ' + str(missing_found))


lines1 = []
for i in eg_pairs_found:
    lines1.append('\t'.join([str(j) for j in i])+'\n')

with open('data/found_eg_pairs.tsv','w') as f:
    f.writelines(lines)

#print(str(len(enhancers.iloc[keep_idx])))
#out = enhancers.iloc[keep_idx].copy()
#out.to_csv('data/full_feature_matrix.coboundp.merged.extra_validated.tsv',sep='\t',index=False)
