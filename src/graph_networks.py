from igraph import *
from predictor import *
from scipy.stats.mstats import gmean
import numpy as np
import pandas as pd
import argparse

#read in gene info
gene_tss = {}
with open('gene_tss.uniq.tsv','r') as f:
    for line in f:
        chrm, gene, tss = line.strip().split('\t')
        gene_tss[gene] = [chrm, int(tss)]

#get gene from cmd line
parser = argparse.ArgumentParser(description='Retrieve Gene')
parser.add_argument('gene', type=str, help='1st argument must be gene')
gene = parser.parse_args().gene

#get gene info, set promoter node
chromosome, tss = gene_tss[gene]
hic_resolution = 5000
promoter_node = '_'.join([chromosome,str(int(np.floor(tss/hic_resolution)*hic_resolution)), str(int(np.floor(tss/hic_resolution)*hic_resolution + hic_resolution))])

#load network
with open('gene_networks_optimized/'+gene+'_network.pkl','rb') as f:
    network = pkl.load(f)

#gather activity to scale color
activity = []
for v in network.vs['name']:
    activity.append(network.vs.find(v)['activity'])

#color activity
pal = GradientPalette("#73d7ff", "#34fa02", int(max(activity))+2)
colors = []
for a in activity:
    colors.append(pal.get(int(a)))

#find neighbors to color
neighbors = network.vs.find(promoter_node).neighbors()
neigh_idx = [v.index for v in neighbors]
pidx = network.vs.find(promoter_node).index
neigh_edges = [network.get_eid(pidx, n) for n in neigh_idx]

ecolor = []
for e in network.es:
    if e.index in neigh_edges:
        ecolor.append('red')
    else:
        ecolor.append('black')

label = []
for v in network.vs:
    if any(x in ['TP','FN','FP'] for x in v['enhancers']['classification']):
        label.append(' '.join(set(v['enhancers']['classification'])))
    else:
        label.append('')

#define shape
shape = []
#define fn
label = []
idx = 0
for node in network.vs['name']:
    if node==promoter_node:
        shape.append('square')
        colors[idx] = 'red'
    elif network.vs.find(node)['Cd'] > network.vs.find(node)['Ceg']:
        shape.append('triangle-up')
    else:    
        shape.append('circle')
    
    idx +=1

visual_style = {}
visual_style["vertex_size"] = 20
visual_style["edge_color"] = ecolor
#visual_style["vertex_label"] = label
visual_style["vertex_color"] = colors
visual_style["vertex_shape"] = shape
plot(network, target='optimized_network_figs/'+gene+'_reg_network.png', **visual_style)
