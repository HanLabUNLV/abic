from igraph import *
from predictor import *
from scipy.stats.mstats import gmean
import numpy as np
import pandas as pd
import argparse

#read in gene info
gene_tss = {}
with open('data/gene_tss.uniq.tsv','r') as f:
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
with open('data/gene_networks_validated_2/'+gene+'_network.pkl','rb') as f:
    network = pkl.load(f)

#gather activity to scale color
activity = []
roles = []
for v in network.vs['name']:
    activity.append(network.vs.find(v)['activity'])
    
#color activity
pal = GradientPalette("#73d7ff", "#34fa02", int(max(activity))+2)
colors = []
for a in activity:
    colors.append(pal.get(int(a)))

#remove non-role enhancers
#print([v['role'] for v in network.vs if v['role']==None])

#find neighbors to color
network.simplify(combine_edges='median')
pidx = network.vs.find(promoter_node).index
#to_delete_ids = [v.index for v in network.vs if (v['role']==None)]
#to_delete_ids.remove(pidx)
#network.delete_vertices(to_delete_ids)

pidx = network.vs.find(promoter_node).index
neighbors = network.vs.find(promoter_node).neighbors()
neigh_idx = [v.index for v in neighbors]
neigh_edges = [network.get_eid(pidx, n) for n in neigh_idx]

ecolor = []
width = []
for e in network.es:
    print(e)
    if e.index in neigh_edges:
        ecolor.append('red')
    else:
        ecolor.append('black')

    width.append(1*e['contact'])
width = 1000*width
print(np.percentile(np.array([x['contact'] for x in network.es]), 75))

label = []
for v in network.vs:
    label.append(v['role'])
    
#define shape
shape = []
#define fn
idx = 0
for node in network.vs['name']:
    if node==promoter_node:
        shape.append('square')
        colors[idx] = 'red'
    else:    
        shape.append('circle')
    
    idx +=1

layout = network.layout('reingold_tilford_circular')
layout[pidx] = [0,0]
visual_style = {}
visual_style["vertex_size"] = 20
visual_style["edge_color"] = ecolor
visual_style["edge_width"] = [100*x +.000001 for x in network.es['contact']]
visual_style["vertex_label"] = label
visual_style["vertex_color"] = colors
visual_style["vertex_shape"] = shape
plot(network, target='data/figs/'+gene+'_network.png', **visual_style, layout=layout)
