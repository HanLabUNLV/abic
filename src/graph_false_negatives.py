from igraph import *
from predictor import *
from scipy.stats.mstats import gmean
import numpy as np
import pandas as pd


def add_node_2_enh(enhancers):
    enhancers['bin1'] = [int(np.floor(tss/hic_resolution)*hic_resolution) for tss in enhancers.start.tolist()]
    enhancers['bin2'] = [int(np.floor(tss/hic_resolution)*hic_resolution + hic_resolution) for tss in enhancers.start.tolist()]
    enhancers['node'] = enhancers.chr.str.cat([enhancers.bin1.astype(str), enhancers.bin2.astype(str)], sep='_')

enhancers = pd.read_csv('ABC_perturbed_K562_EGpairs.classified.csv', header=0)
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
 
#genes = ['GATA1', 'CCDC26', 'DNASE2', 'FTL', 'KLF1', 'NUCB1', 'FIG4', 'PVT1-TSS1', 'PQBP1', 'PRDX2', 'H1FX', 'MYC', 'JUNB', 'WDR83OS', 'HNRNPA1', 'FUT1', 'DHPS', 'BAX', 'RAE1', 'HBE1', 'CALR', 'HDAC6', 'NFE2', 'PLP2', 'RAD23A', 'RPN1']

genes = ['BAX','BCAT2','C19orf43','CALR','CCDC26','CNBP','COPZ1','DHPS','DNASE2','FTL','FUT1','GATA1','H1FX','HNRNPA1','ITGA5','JUNB','KCNN4','KLF1','LYL1','MYC','NFE2','NUCB1','PPP1R15A','PQBP1','PRDX2','RAB7A','RAD23A','RNASEH2A','RPN1','SEC61A1','SUOX','UROS','WDR83OS']
for gene in genes:
    try:
        with open('gene_networks_filtered/'+gene+'_network.pkl','rb') as f:
            network = pkl.load(f)

        tss = enhancers.loc[enhancers['Gene']==gene, 'Gene TSS'].tolist()[0]
        chromosome = enhancers.loc[enhancers['Gene']==gene, 'chr'].tolist()[0]
        hic_resolution = 5000
        promoter_node = '_'.join([chromosome,str(int(np.floor(tss/hic_resolution)*hic_resolution)), str(int(np.floor(tss/hic_resolution)*hic_resolution + hic_resolution))])

        activity = []
        for v in network.vs:
            if v['enhancers'] is not None:
                if len(v['enhancers']['activity'])>0:
                    activity.append(gmean(v['enhancers']['activity']))
                else:
                    activity.append(0)
            else:
                activity.append(0)
        #singletons = network.vs.select(_degree = 0)
        #network.delete_vertices(singletons)
        if len(activity)>0:
        #find FN in this gene, on the network
            fn = deepcopy(enhancers.loc[(enhancers['Gene']==gene) & (enhancers['classification']=='FN'), ])
            add_node_2_enh(fn)
            fn_nodes = fn.node.tolist()

            tp = deepcopy(enhancers.loc[(enhancers['Gene']==gene) & (enhancers['classification']=='TP'), ])
            add_node_2_enh(tp)
            tp_nodes = tp.node.tolist()

            fp = deepcopy(enhancers.loc[(enhancers['Gene']==gene) & (enhancers['classification']=='FP'), ])
            add_node_2_enh(fp)
            fp_nodes = fp.node.tolist()
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
                    width.append(c)
                    ecolor.append('black')
                elif c>cutoff2:
                    width.append(c)
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

                if node in fn_nodes:
                    label.append('FN')
                elif node in tp_nodes:
                    label.append('TP')
                elif node in fp_nodes:
                    label.append('FP')
                else:
                    label.append('')
                
                idx +=1

            visual_style = {}
            visual_style["vertex_size"] = 20
            #visual_style["edge_width"] = np.log10([20*x +.000001 for x in network.es['contact']])
            visual_style["edge_color"] = ecolor
            visual_style["vertex_label"] = label
            visual_style["vertex_color"] = colors
            visual_style["vertex_shape"] = shape
            plot(network, target='false_negative_networks/'+gene+'_reg_network.png', **visual_style)
    except FileNotFoundError:
        print(gene)
