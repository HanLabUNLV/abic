import pandas as pd
import pyranges as pr
from numpy import floor
import os
import joblib as jl

#import mira's dataset
#import my dataset
mira = pd.read_csv('data/Gasperini2019.at_scale.ABC.TF.txt',sep='\t')
gene_networks_dir = 'data/gene_networks_validated_2/'
####mine = pd.read_csv('data/full_feature_matrix.revalidated.final2.tsv',sep='\t')

#####make pyranges out of my data bc the coords are not matching 
####classpr = pr.from_dict({'Chromosome':mine['chr'], 'Start':mine['start'], 'End':mine['stop'], 'Gene':mine['gene'], 'role':mine['role']})


#####iter through mira's dataset, get coords, look for overlap in my dataset as well as matching gene and then return the class
mira['e1']=0
mira['e2']=0
mira['e3']=0

####for idx, row in mira.iterrows():
####    tmp = pr.from_dict({'Chromosome':[row['chrEnhancer']], 'Start':[row['startEnhancer']], 'End':[row['endEnhancer']], 'Gene':[row['GeneSymbol']]})

####    matches = classpr.overlap(tmp)

####    if len(matches)>0: 
####        genes = matches.Gene.tolist()
####    else:
####        genes=[]


####    if row['GeneSymbol'] in genes:
####        result = matches.df.iloc[genes.index(row['GeneSymbol'])]
####        role = result['role']
####        if role is not None:
####            if role=='E1':
####                mira.loc[idx,'e1']=1
####            elif role=='E2':
####                mira.loc[idx,'e2']=1
####            elif role=='E3':
####                mira.loc[idx,'e3']=1

res = 5000
for idx, row in mira.iterrows():
    chrm, start, stop, gene = row.loc[['chrEnhancer','startEnhancer','endEnhancer','GeneSymbol']]
    node = '_'.join([chrm, str(int(floor(start/res)*res)), str(int(floor(stop/res)*res+ res))])

    if os.path.isfile(gene_networks_dir+gene+'_network.pkl'):
        network = jl.load(gene_networks_dir+gene+'_network.pkl')
        nodes = [v['name'] for v in network.vs]
        if node in nodes:
            try:
                role = network.vs.find(node)['role']
            except:
                role = 'NA'

            if role=='E1':
                mira.loc[idx, 'e1'] = 1
            elif role=='E2':
                mira.loc[idx, 'e2'] = 1
            elif role=='E3':
                mira.loc[idx, 'e3'] = 1
            

mira.to_csv('data/Gasperini2019.at_scale.ABC.TF.roles.updated.txt',sep='\t')

