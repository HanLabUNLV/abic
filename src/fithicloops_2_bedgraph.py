import pandas as pd

#chromosome = 'chr10'
p_cutoff = 0.05 #corresponds to roughly 0.0`1 percentile of p values

for chromosome in ['chr10',  'chr12',  'chr19',  'chr3',  'chr8',  'chrX']:
    #change format to bedpe  
    resolution = 5000
    loops =  pd.read_csv('fithic_loops/'+chromosome+'/FitHiC.spline_pass1.res5000.significances.txt.gz', compression='gzip',header=0,sep='\t')
    loops = loops.rename(columns={"fragmentMid1":"start1","fragmentMid2":"start2"})
    loops.insert(2, 'end1', loops['start1'].tolist())
    loops.insert(5, 'end2', loops['start2'].tolist())
    loops['end1'] = loops['end1'] + resolution
    loops['end2'] = loops['end2'] + resolution


    #filter by p value (taken from FDR song et al)
    loops = loops.loc[(loops['p-value'] < p_cutoff) & (loops['p-value']>0)]

    #write to file
    loops.to_csv('fithic_loops/'+chromosome+'/fithic_filtered.bedpe', sep='\t', index=False)




