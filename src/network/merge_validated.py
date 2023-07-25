import pandas as pd
import numpy as np
import pyranges as pr
import pickle as pkl
import time
pd.set_option('display.max_columns', None)

tstart = time.time()
#read in total feature matrix and gasperini data
total = pd.read_csv('data/full_feature_matrix.revalidated2.tsv',sep='\t')
gas = pd.read_csv('data/Gasperini2019.at_scale_screen.cand_enhancer_x_exprsd_genes.200503.csv',quotechar='"') 
abc = pd.read_csv('/data8/han_lab/mhan/ABC-Enhancer-Gene-Prediction/Gasperini/ABC_output/Predictions/EnhancerPredictionsAllPutative.txt',sep='\t')

#translate to pyranges 
gas = gas.rename(columns={'chrEnhancer':'Chromosome','startEnhancer':'Start', 'endEnhancer':'End','GeneSymbol':'gene'})   
gas['Start'] = gas.Start.astype(int)
gas['End'] = gas.End.astype(int)
feature_eg = total.loc[:,['chr','start','stop','gene']].copy()
feature_eg = feature_eg.rename(columns={'chr':'Chromosome','start':'Start', 'stop':'End'})
feature_eg.dropna(axis=0, how='any', inplace=True) 
#gas.dropna(axis=0, how='any', inplace=True) 
#print(feature_eg.shape)
#print(gas.columns)
#print(feature_eg.columns)
feat_pr = pr.PyRanges(feature_eg)

#####iter through gasperini data then use the revalidate script / pyranges to check if feature matrix includes each EG pair
####newrows = []
####eg_found = 0
####eg_mismatch = 0
####eg_mismatches = []
####no_ov = 0
####for idx, row in gas.iterrows():
####    chrm = row['Chromosome']
####    start = row['Start']
####    end = row['End']
####    gene = row['gene']
####    #convert to pyranges, overlap 
####    tpr = pr.from_dict({'Chromosome':[chrm], 'Start':[start], 'End':[end]})     
####    ov = feat_pr.overlap(tpr)
####    if len(ov)>0:
####        glist = ov.gene.tolist()
####        #those that are present, pass, but otherwise, append to a list of EG pairs to gather data for later
####        if gene in glist:
####            eg_found += 1
####            rstart = ov.Start.tolist()[0]
####            rstop = ov.End.tolist()[0]
####            #set significance in feature matrix, (some were missing, so it never hurts to revalidate)
####            total.loc[(total['chr']==chrm)&(total['start']==rstart)&(total['stop']==rstop)&(total['gene']==gene),'sig'] = row['Significant']
####            total.loc[(total['chr']==chrm)&(total['start']==rstart)&(total['stop']==rstop)&(total['gene']==gene),'effect_size'] = row['EffectSize']
####            #print(row)
####            #print(ov)
####            #print(gene)
####            #print(glist)
####        else:
####            eg_mismatch +=1
####            #since eg is all that's mismatched, use data from matrix, set role, contact, and degree to 0
####            ovdf = ov.as_df()
####            nchr = ovdf.Chromosome.values[0]
####            nstart = ovdf.Start.values[0]
####            nend = ovdf.End.values[0]
####            tgene = ovdf.gene.values[0]
####            eg_mismatches.append([[chrm, start, end, gene], [nchr,nstart,nend,tgene]])
####            newrows.append([chrm, start, end, gene])
####    else:
####        no_ov += 1
####        newrows.append([chrm, start, end, gene])

####print(gas.shape)
####print(feature_eg.shape)
####print(eg_found)
####print(eg_mismatch)
####print(no_ov)

#with open('data/egmismatch.tmp.txt','w') as f:
#    for r in eg_mismatches:
#        f.write('\t'.join([str(x) for x in r[0]])+'\t'+'\t'.join([str(x) for x in r[1]])+'\n')

eg_mismatches = []
with open('data/egmismatch.tmp.txt','r') as f:
    for line in f.readlines():
        a, b, c, d, e, f, g, h = line.strip().split('\t')
        eg_mismatches.append([[a,int(b),int(c),d],[e,int(f),int(g),h]])

        

#create a df with the same cols as total and nrow=(eg_mismatch + no_ov) 
colnames = total.columns
chrms = [x[1][0] for x in eg_mismatches]
starts = [int(x[1][1]) for x in eg_mismatches]
ends = [int(x[1][2]) for x in eg_mismatches]
genes = [x[0][3] for x in eg_mismatches]

#newdf = pd.DataFrame({'chr':chrms,'start':starts,'stop':ends,'gene':genes})
#newdf = newdf.reindex(newdf.columns.union(colnames), axis=1)
newdf = pd.DataFrame(columns = colnames)

#gather the data from various sources 
#for the eg pairs with e in feature matrix, but not to the right g, add in row from existing data, make contact, role, degree = 0
idx = 0
gas_missing = 0
for row, ov in eg_mismatches:
    chrm, start, stop, tgene = ov
    gene = row[3]

    nurow = total.loc[(total['chr']==chrm)&(total['start']==start)&(total['stop']==stop)&(total['gene']==tgene), ].copy()
    #since the network isn't available for these EG pairs, set role and degree = 0
    nurow['gene'] = gene
    nurow['contact'] = 0
    nurow['role'] = 'None'
    nurow['degree'] = 0
    nurow['abc_score'] = 'NA'

    #before assigning to final matrix, lookup in original abc matrix to see if you can find missing vals
    abc_tmp = abc.loc[(abc['chr']==chrm)&(abc['start']==start)&(abc['end']==stop)&(abc['TargetGene']==gene),]
    if len(abc_tmp)>0:
        nurow['contact'] = abc_tmp.hic_contact_pl_scaled_adj.values[0] 
        nurow['abc_score'] = abc_tmp['ABC.Score'].values[0]      
    #lookup in gasperini matrix to fill in more missing vals
    gchr, gstart, gstop, ggene = row
    gas_tmp = gas.loc[(gas['Chromosome']==gchr)&(gas['Start']==int(gstart))&(gas['End']==int(gstop))&(gas['gene']==gene),]
    if len(gas_tmp)>0:
        nurow['sig'] = gas_tmp.Significant.values[0]
        nurow['effect_size'] = gas_tmp.EffectSize.values[0]
        newdf = newdf.append(nurow, ignore_index=True) 
    else:
        gas_missing+=1
        continue

print(gas_missing)
print(newdf.head())

#we validated the data in newdf, but in total, there's still some missing that we can get from gasperini
gaspr = pr.PyRanges(gas)
idx_to_del = []
for idx, row in total.iterrows():
    chrm = row['chr']
    start = row['start']
    end = row['stop']
    gene = row['gene']
    #convert to pyranges, overlap 
    tpr = pr.from_dict({'Chromosome':[chrm], 'Start':[start], 'End':[end],'gene':[gene]})     
    ov = gaspr.overlap(tpr)
    if len(ov)>0:
        glist = ov.gene.tolist()
        #those that are present, pass, but otherwise, append to a list of EG pairs to gather data for later
        if gene in glist:
            #set significance in feature matrix, (some were missing, so it never hurts to revalidate)
            total.loc[(total['chr']==chrm)&(total['start']==start)&(total['stop']==end)&(total['gene']==gene),'sig'] = ov.Significant.values[glist.index(gene)]
            total.loc[(total['chr']==chrm)&(total['start']==start)&(total['stop']==end)&(total['gene']==gene),'effect_size'] = ov.EffectSize.values[glist.index(gene)]
        else:
            idx_to_del.append(idx)
    else:
        idx_to_del.append(idx)
total.drop(idx_to_del, inplace=True)

outdf = pd.concat([total,newdf], ignore_index=True)
outdf.to_csv('data/full_feature_matrix.revalidated.final.tsv',sep='\t',index=False)
print('final df shape: ' + str(outdf.shape))
print('time elapsed: '+ str(time.time()-tstart))
exit('done')
