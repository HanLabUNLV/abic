from igraph import *
from predictor import *
from scipy.stats.mstats import gmean
import numpy as np
import pandas as pd
import argparse
 
######################################
# first graph network with all edges #
######################################

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

#########load network
########with open('data/gene_networks/'+gene+'_network.pkl','rb') as f:
########    network = pkl.load(f)

#########remove dup nodes
########seen = []
########to_delete = []
########for v in network.vs:
########    if v['name'] in seen:
########        to_delete.append(v.index)
########    else:
########        seen.append(v['name'])
########network.delete_vertices(to_delete) 

#########gather activity to scale color
########activity = []
########roles = []
########for v in network.vs['name']:
########    activity.append(network.vs.find(v)['activity'])
########    
#########color activity
########pal = GradientPalette("#73d7ff", "#34fa02", int(max(activity))+2)
########colors = []
########for a in activity:
########    colors.append(pal.get(int(a)))

#########remove non-role enhancers
#########print([v['role'] for v in network.vs if v['role']==None])

#########find neighbors to color

#########pidx = network.vs.find(promoter_node).index
#########to_delete_ids = [v.index for v in network.vs if ((v['role']==None) or (v['role']=='E3'))]
#########to_delete_ids.remove(pidx)
#########network.delete_vertices(to_delete_ids)

########pidx = network.vs.find(promoter_node).index
########neighbors = network.vs.find(promoter_node).neighbors()
########neigh_idx = [v.index for v in neighbors]
########neigh_edges = [network.get_eid(pidx, n) for n in neigh_idx]
########ecolor = []
########width = []
########for e in network.es:
########    if e.index in neigh_edges:
########        ecolor.append('red')
########    else:
########        ecolor.append('black')

#########label = []
#########for v in network.vs:
#########    if any(x in ['TP','FN','FP'] for x in v['enhancers']['classification']):
#########        label.append(' '.join(set(v['enhancers']['classification'])))
#########    else:
#########        label.append('')

#########define shape
########shape = []
#########define fn
########label = []
########idx = 0
########for node in network.vs['name']:
########    if node==promoter_node:
########        shape.append('square')
########        colors[idx] = 'red'
########    else:    
########        shape.append('circle')
########    
########    idx +=1


#########layout = network.layout('reingold_tilford_circular')
########network = network.simplify(multiple=True, combine_edges='max')
########layout = network.layout('large')
########layout[pidx] = [0,0]
########visual_style = {}
########visual_style["vertex_size"] = 20
########visual_style["edge_color"] = ecolor
########visual_style["edge_width"] = np.log10([50*x +.000001 for x in network.es['contact']])
#########visual_style["vertex_label"] = label
########visual_style["vertex_color"] = colors
########visual_style["vertex_shape"] = shape
########plot(network, target='data/figs/filtered_networks/'+gene+'_full_network.png', **visual_style, layout=layout)
########print('full plot done')



###########################################################
######### second graph network with fithic2 filtered edges #
############################################################

########lbound = tss - 5000000/2
########ubound = tss + 5000000/2

#########
#########filter gene network with bedgraph edges
#########

########bedgraph = 'data/fithic_loops/'+chromosome+'/fithic_filtered.bedpe'
########filtered_loops = pd.read_csv(bedgraph, sep='\t')
########filtered_loops = filtered_loops.loc[(filtered_loops['start1']>lbound)&(filtered_loops['start2']>lbound)&(filtered_loops['end1']>lbound)&(filtered_loops['end2']>lbound),]
########filtered_loops = filtered_loops.loc[(filtered_loops['start1']<ubound)&(filtered_loops['start2']<ubound)&(filtered_loops['end1']<ubound)&(filtered_loops['end2']<ubound),]
########filtered_loops['node1_id'] = filtered_loops.chr1.str.cat([filtered_loops.start1.astype(str), filtered_loops.end1.astype(str)], sep='_')
########filtered_loops['node2_id'] = filtered_loops.chr2.str.cat([filtered_loops.start2.astype(str), filtered_loops.end2.astype(str)], sep='_')

#########identify which edges to keep and lose
########node1s = filtered_loops['node1_id'].tolist()
########node2s = filtered_loops['node2_id'].tolist()
########p_values = filtered_loops['p-value'].tolist()
########enames_kept = [[node1s[i],node2s[i],p_values[i]] for i in range(0,len(node1s))]
########eids_kept = []
########eids_missed = []
########for e in enames_kept:
########    try:
########        source=network.vs.find(e[0])
########        target=network.vs.find(e[1])
########        eids_kept.append([source.index,target.index, e[2]])

########    except:
########        eids_missed.append(e)
########        pass

########eids_kept = [e[:2] for e in eids_kept]

########final_eids = network.get_eids(eids_kept)
########total_eids = [e.index for e in network.es[:]]
########lost_eids = list(set(total_eids) - set(final_eids))
#########delete edges of lost loops
########network.delete_edges(lost_eids)
#########some nodes are unconnected now, so clean them off
########network.vs.select(_degree=0).delete()
########print('edges fithic2 filtered')

#########
#########graph it
#########

#########gather activity to scale color
########activity = []
########roles = []
########for v in network.vs['name']:
########    activity.append(network.vs.find(v)['activity'])
########    
#########color activity
########pal = GradientPalette("#73d7ff", "#34fa02", int(max(activity))+2)
########colors = []
########for a in activity:
########    colors.append(pal.get(int(a)))

#########remove non-role enhancers
#########print([v['role'] for v in network.vs if v['role']==None])

#########find neighbors to color

########pidx = network.vs.find(promoter_node).index
########neighbors = network.vs.find(promoter_node).neighbors()
########neigh_idx = [v.index for v in neighbors]
########neigh_edges = [network.get_eid(pidx, n) for n in neigh_idx]
########ecolor = []
########width = []
########for e in network.es:
########    if e.index in neigh_edges:
########        ecolor.append('red')
########    else:
########        ecolor.append('black')

#########define shape
########shape = []
#########define fn
########label = []
########idx = 0
########for node in network.vs['name']:
########    if node==promoter_node:
########        shape.append('square')
########        colors[idx] = 'red'
########        print(node.index)
########    else:    
########        shape.append('circle')
########    
########    idx +=1


#########layout = network.layout('reingold_tilford_circular')
########layout = network.layout('kk')
########layout[pidx] = [-1,0]
########visual_style = {}
########visual_style["vertex_size"] = 20
########visual_style["edge_color"] = ecolor
########visual_style["edge_width"] = np.log10([50*x +.000001 for x in network.es['contact']])
#########visual_style["vertex_label"] = label
########visual_style["vertex_color"] = colors
########visual_style["vertex_shape"] = shape
########plot(network.simplify(multiple=True), target='data/figs/filtered_networks/'+gene+'_fithic2_network.png', **visual_style, layout=layout)
########print('fithic filtered graph plotted')


###############################################################
# third graph network with gasperini validated enhancers only #
###############################################################

#instead of actually recreating the final network, just load in the saved version

#load network
with open('data/gene_networks_validated_2/'+gene+'_network.pkl','rb') as f:
    network = pkl.load(f)

#remove dup nodes
seen = []
to_delete = []
for v in network.vs:
    if v['name'] in seen:
        to_delete.append(v.index)
    else:
        seen.append(v['name'])
network.delete_vertices(to_delete) 

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

#pidx = network.vs.find(promoter_node).index
#to_delete_ids = [v.index for v in network.vs if ((v['role']==None) or (v['role']=='E3'))]
#to_delete_ids.remove(pidx)
#network.delete_vertices(to_delete_ids)

pidx = network.vs.find(promoter_node).index
neighbors = network.vs.find(promoter_node).neighbors()
neigh_idx = [v.index for v in neighbors]
neigh_edges = [network.get_eid(pidx, n) for n in neigh_idx]
ecolor = []
width = []

#label = []
#for v in network.vs:
#    if any(x in ['TP','FN','FP'] for x in v['enhancers']['classification']):
#        label.append(' '.join(set(v['enhancers']['classification'])))
#    else:
#        label.append('')

#define shape
shape = []
#define fn
label = []
idx = 0
for node in network.vs['name']:
    if node==promoter_node:
        shape.append('square')
        colors[idx] = 'red'
        print(node.index)
    else:    
        shape.append('circle')
    
    idx +=1


layout = network.layout('reingold_tilford_circular')
#layout = network.layout('large')
network = network.simplify(multiple=True, combine_edges='max')
layout[pidx] = [0,0]
visual_style = {}
visual_style["vertex_size"] = 20
visual_style["edge_color"] = ecolor
visual_style["edge_width"] = np.log10([100*x +.000001 for x in network.es['contact']])
#visual_style["vertex_label"] = label
visual_style["vertex_color"] = colors
visual_style["vertex_shape"] = shape
#plot(network, target='data/figs/filtered_networks/'+gene+'_pruned_network.png', **visual_style, layout=layout)
print('gasperini filtered plot done')

#
#lastly, choose one e2 to demonstrate on the network
#

pos_e2 = 'chr6_26475000_26480000'
path = []
e2idx = network.vs.find(pos_e2).index
pidx = network.vs.find(promoter_node).index
neighbors = network.vs.find(pos_e2).neighbors()
neigh_idx = [v.index for v in neighbors]
neigh_edges = [network.get_eid(e2idx, n) for n in neigh_idx]

p_neigh = [v.index for v in network.vs.find(promoter_node).neighbors()]

connecting = list(set(p_neigh).intersection(set(neigh_idx)))
contacts = [network.es.find(network.get_eid(e2idx, c))['contact'] for c in connecting]
e1 = contacts.index(max(contacts))
e1idx = connecting[e1]
path.append(network.get_eid(e1idx, e2idx)) #edge between e2 - 1
path.append(network.get_eid(e1idx, pidx)) #edge between e1 - p




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

pidx = network.vs.find(promoter_node).index
neighbors = network.vs.find(promoter_node).neighbors()
neigh_idx = [v.index for v in neighbors]
neigh_edges = [network.get_eid(pidx, n) for n in neigh_idx]
ecolor = []

width = []
for e in network.es:
    if e.index in path:
        ecolor.append('#d45e0f')
        width.append(e['contact'] * 30)
    else:
        ecolor.append('black')
        width.append(e['contact'])

#define shape
shape = []
#define fn
label = []
idx = 0
for node in network.vs['name']:
    if node==promoter_node:
        shape.append('square')
        colors[idx] = 'red'
        print(node.index)
    else:    
        shape.append('circle')
    
    idx +=1


layout = network.layout('reingold_tilford_circular')
#layout = network.layout('large')
network = network.simplify(multiple=True, combine_edges='max')
layout[pidx] = [0,0]
visual_style = {}
visual_style["vertex_size"] = 20
visual_style["edge_color"] = ecolor
visual_style["edge_width"] = np.log10([100*x +.000001 for x in width])
#visual_style["vertex_label"] = label
visual_style["vertex_color"] = colors
visual_style["vertex_shape"] = shape
plot(network, target='data/figs/filtered_networks/'+gene+'_e2_ex_network.png', **visual_style, layout=layout)
print('e2 example plot done')
