from igraph import *
from predictor import *
from scipy.stats.mstats import gmean
import numpy as np


#  vertices= [0,1,2,3,4,5,6,7,8]
#  edges = [(8,5),(0,5),(0,8),(1,5),(1,2),(2,3),(3,4),(2,4),(1,6),(6,7),(7,8)]
#  weights = [1,4,1,4,1,1,1,1,1,1,1]
#  ecolor = ['black','green','black','green','black','black','black','black','black','black','black']

#  network = Graph()
#  network.add_vertices(vertices)
#  network.add_edges(edges)

#  color = []
#  shape = []
#  for i in vertices:
#      if i==0:
#          color.append('green')
#          shape.append('circle')
#      elif i==1:
#          color.append('red')
#          shape.append('square')
#      else:
#          color.append('blue')
#          shape.append('circle')



#  visual_style = {}
#  #visual_style["vertex_label"] = [str(x) for x in vertices]
#  visual_style["vertex_size"] = 20
#  visual_style["edge_width"] = weights
#  visual_style["edge_color"] = ecolor
#  visual_style["vertex_color"] = color
#  visual_style["vertex_shape"] = shape
#  plot(network, target='chr19/ABC_ex2.png', **visual_style)

#  exit()
with open('chr19/gene_networks/CALR_network_cd.pkl','rb') as f:
    network = pkl.load(f)

tss = 13049413
hic_resolution = 5000
promoter_node = '_'.join(['chr19',str(int(np.floor(tss/hic_resolution)*hic_resolution)), str(int(np.floor(tss/hic_resolution)*hic_resolution + hic_resolution))])

####### activity = []
####### for v in network.vs:
#######     if v['enhancers'] is not None:
#######         if len(v['enhancers']['activity'])>0:
#######             activity.append(gmean(v['enhancers']['activity']))
#######         else:
#######             activity.append(0)
#######     else:
#######         activity.append(0)
gene='CALR'

singletons = network.vs.select(_degree = 0)
network.delete_vertices(singletons)

#color activity
pal = GradientPalette("#73d7ff", "#34fa02", int(max(activity))+2)
colors = []
for a in activity:
    colors.append(pal.get(int(a)))

#no select define width of edges
width = []
ecolor = []
cutoff1 = np.quantile(network.es['contact'], 0.9911)
cutoff2 = np.quantile(network.es['contact'], 0.95)
for c in network.es['contact']:
    if c>cutoff1:
        width.append(2*c)
        ecolor.append('black')
    elif c>cutoff2:
        width.append(1.5*c)
        ecolor.append('black')
    else:
        width.append(c)
        ecolor.append('black')
width = 2*width
#       width = []
#       for c in network.es:
#           v1 = network.vs[c.source]['name']
#           v2 = network.vs[c.target]['name']
#           if v1==promoter_node or v2==promoter_node:
#               width.append(1.25)
#           else:
#               width.append(c['contact'])

#define shape
shape = []
idx = 0
for node in network.vs['name']:
    if node==promoter_node:
        shape.append('square')
        colors[idx] = 'red'
    elif network.vs.find(node)['Cd'] > network.vs.find(node)['Ceg']:
        shape.append('triangle-up')
    else:    
        shape.append('circle')
    idx+=1
    


visual_style = {}
visual_style["vertex_size"] = 20
visual_style["edge_width"] = np.log10([20*x +.000001 for x in network.es['contact']])
visual_style["edge_color"] = ecolor
visual_style["vertex_color"] = colors
visual_style["vertex_shape"] = shape
plot(network, target='chr19/'+gene+'_reg_network_cd.png', **visual_style)
 
