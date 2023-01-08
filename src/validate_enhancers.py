#this script will validate enh from an experiment and get the overlapped coords from the output of ABC
import pandas as pd
import pyranges as pr

#data in / out
out = open('data/validated_enhancers.bed','w')
abc = pd.read_csv('/data8/han_lab/mhan/ABC-Enhancer-Gene-Prediction/Gasperini/ABC_output/Predictions/EnhancerPredictionsAllPutative.txt', sep='\t')
val = pd.read_csv('data/Gasperini2019.at_scale_screen.cand_enhancer_x_exprsd_genes.200503.csv',quotechar='"')

#init pyranges
pr_abc = pr.PyRanges(abc.rename(columns={'chr':'Chromosome','start':'Start','end':'End'}))
val = val.rename(columns={'chrEnhancer':'Chromosome','startEnhancer':'Start','endEnhancer':'End'})

#init tracking params
eg_found = 0
eg_missing = 0
abc_coords = []
for index, row in val.iterrows():
    pr_tmp = pr.from_dict({'Chromosome':[row['Chromosome']],'Start':[row['Start']],'End':[row['End']]})
    ov = pr_abc.overlap(pr_tmp)
    if len(ov) > 0:
        eg_found += 1
        abc_coords.append(ov.as_df().values.tolist()[0][0:3])
    else:
        eg_missing += 1
print(eg_found)
print(eg_missing)
print(abc_coords)

for enh in abc_coords:
    chrm, start, stop = enh
    start = str(int(start))
    stop = str(int(stop))

    out.write('\t'.join([chrm, start, stop]) + '\n')
    
