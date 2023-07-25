import pandas as pd
import argparse
from numpy import floor


parser = argparse.ArgumentParser(description = "Description for my parser")
parser.add_argument("-i", "--feature_matrix", help = "feature matrix of EG pair rows (use -i or --feature_matrix)", required = True)
argument = parser.parse_args()

#read in file from cmd line
fm_file = parser.parse_args().feature_matrix

#read file into pandas df
fm = pd.read_csv(fm_file, sep='\t')

#def coord translater
def coords_to_hic_bin(l1):
    hic_res = 5000
    out = []
    for eid in l1:
        chrm, start, stop = eid.split('_')
        start = int(start)
        stop = int(stop)
        hstart = int(floor(start/hic_res)*hic_res)
        hstop = int(floor(start/hic_res)*hic_res+hic_res)
        out.append('_'.join([chrm, str(hstart), str(hstop)]))

    return(out)

def compare_bins(l1):
    return(int(l1[0]==l1[1]))


#init enhancer id col
fm['eid'] = fm[['chrEnhancer','startEnhancer' ,'endEnhancer']].applymap(str).agg('_'.join, axis=1)

#init promoter id col
fm['pid'] = fm[['chrTSS','startTSS','endTSS']].applymap(str).agg('_'.join, axis=1)

 
#init e0 col
e0 = [compare_bins(coords_to_hic_bin([x,y])) for x, y in zip(fm['eid'], fm['pid'])]
e1_idx = fm.columns.tolist().index('e1')
fm.insert(e1_idx, 'e0', e0)

#check no double counting
fm['esum'] = fm[['e0','e1','e2','e3']].agg(sum, axis=1)
#print(fm.loc[fm['esum']>1,['e0','e1','e2','e3','esum','TargetGene','eid','pid','Significant']])
#there's 7 insignificant double counted, upon review, these are in fact e0, not sure how they got the e1 label as they're in the same hic node as their promoter
fm.loc[fm['esum']>1,'e1'] = 0

fm.drop(['eid','pid','esum'], axis=1, inplace=True)
fm.to_csv('data/Gasperini2019.at_scale.ABC.TF.roles.updated.e0.txt',sep='\t')
