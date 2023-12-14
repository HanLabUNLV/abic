from predictor_pandas_eponly import *
import pandas as pd
import os.path
import argparse
import time

def get_model_argument_parser():
    class formatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
        pass

    parser = argparse.ArgumentParser(description='Predict enhancer relative effects.',
                                     formatter_class=formatter)
    readable = argparse.FileType('r')

    #Basic parameters
    parser.add_argument('--netdir', required=True, help="network graph directory")
    parser.add_argument('--dir', required=True, help="output directory")
    parser.add_argument('--chromosomes', default="all", help="chromosomes to make predictions for. Defaults to intersection of all chromosomes in --genes and --enhancers")
    parser.add_argument('--infile', required=False, help="gasperini infile name")

    return parser


def get_predict_argument_parser():
    parser = get_model_argument_parser()
    return parser


def compute_score(enhancers, product_terms):

    scores = np.column_stack(product_terms).prod(axis = 1) 
    
    enhancers['ABC.Score.Numerator'] = scores
    enhancers['ABC.Score'] = enhancers['ABC.Score.Numerator'] / enhancers.groupby('TargetGene')['ABC.Score.Numerator'].transform('sum')
    return(enhancers)





def network_from_eps(chromosome, netdir):

    print("reading network") 
    vertices = pd.read_csv(os.path.join(netdir, "vertices_ep_TSS."+chromosome+".txt"), sep="\t", index_col=0)
    edgelist = pd.read_csv(os.path.join(netdir, "edgelist_ep_TSS."+chromosome+".txt"), sep="\t", index_col=0)

    t = time.time()

    g = Graph.TupleList(edgelist.itertuples(index=False), directed=False, weights=True )

    for column in vertices:
      g.vs[column] = vertices.loc[g.vs['name'],column]
    #g.write_edgelist("g.gene.enhancer.edgelist")
    #g.write_graphml("gene.enhancer."+chromosome+".graphml")
    print("graph from pandas:", time.time()-tmp)

    tmp = time.time()

    print('connecting complete graphs based on within hic bin relationship..')
    first_of_pairs = list()
    second_of_pairs = list()
    for i, row in vertices_hic_new.iterrows():
      element_ids = edgelist_elements.loc[edgelist_elements['source'] == i,'target']
      pairs = list(itertools.combinations(element_ids, 2))
      if pairs:
        first_of_pairs.extend([x[0] for x in pairs])
        second_of_pairs.extend([x[1] for x in pairs])

    weights = np.repeat([1], [len(first_of_pairs)], axis=0)
    types = np.repeat(["within"], [len(first_of_pairs)], axis=0)
    edgelist_elements_within = pd.DataFrame(data = {'source': first_of_pairs, 'target': second_of_pairs, 'weight': weights, 'type': types})


    print('connecting edges based on between hic bin contact..')
    first_of_pairs = list()
    second_of_pairs = list()
    weights = list()
    for i, row in edgelist_hic_new.iterrows():
      element_ids_bin1 = edgelist_elements.loc[edgelist_elements['source'] == row['source'],'target']
      element_ids_bin2 = edgelist_elements.loc[edgelist_elements['source'] == row['target'],'target']
      pairs = list(itertools.product(element_ids_bin1, element_ids_bin2))
      if pairs:
        first_of_pairs.extend([x[0] for x in pairs])
        second_of_pairs.extend([x[1] for x in pairs])
        weights.extend(np.repeat([row['weight']], [len(pairs)], axis=0))

    types = np.repeat(["between"], [len(first_of_pairs)], axis=0)
    edgelist_elements_between = pd.DataFrame(data = {'source': first_of_pairs, 'target': second_of_pairs, 'weight': weights, 'type': types})
    edgelist_elements_new = pd.concat([edgelist_elements_within,edgelist_elements_between], ignore_index=False, sort=True)

##############################
    vertices_elements.to_csv(os.path.join(args.dir, "vertices_ep_TSS."+chromosome+".txt"), sep="\t", index=True)
    edgelist_elements_new.to_csv(os.path.join(args.dir, "edgelist_ep_TSS."+chromosome+".txt"), sep="\t", index=True)

    return edgelist_elements_new





if __name__ == '__main__':
  start = time.time()
  tmp = time.time()
  parser = get_predict_argument_parser()
  args = parser.parse_args()


  if not os.path.exists(args.dir):
      os.makedirs(args.dir)

  infile_base = os.path.splitext(args.infile)[0]

  ABC_orig = pd.read_csv(os.path.join(args.dir, args.infile),sep='\t', header=0)
  ABC = ABC_orig.copy()
  ABC = ABC.loc[:,~ABC.columns.str.match("Unnamed")]
  if 'chrEnhancer' not in ABC.columns:
    ABC['chrEnhancer'] = ABC['chr']
  ABC[['ABC_enhancer','ABC_gene']] = ABC['ABC.id'].str.split('_',expand=True)
  ABC['path.step'] = 0
  ABC['path.ABC.mean'] = 0
  ABC['path.ABC.product'] = 0

  if args.chromosomes == "all":
      chromosomes = set(ABC['chrEnhancer'])
  else:
      chromosomes = args.chromosomes.split(",")

  for chromosome in chromosomes:

    print("reading network") 
    vertices = pd.read_csv(os.path.join(args.netdir, "vertices_ep_TSS."+chromosome+".txt"), sep="\t", index_col=0)
    edgelist = pd.read_csv(os.path.join(args.netdir, "edgelist_ep_TSS."+chromosome+".txt"), sep="\t", index_col=0)
    vertices[['ABC.type','ABC_enhancer']] = vertices['id'].str.split('|',expand=True)
    vertices['ABC_gene'] = None
    vertices.loc[vertices['ABC_enhancer'].isnull(),'ABC_gene'] = vertices.loc[vertices['ABC_enhancer'].isnull(),'id'].to_list()

    t = time.time()

    g = Graph.TupleList(edgelist.itertuples(index=False), directed=False, weights=True )
    g.vs['id'] = vertices.loc[g.vs['name'],'id'].to_list()
    g.vs['ABC_enhancer'] = vertices.loc[g.vs['name'],'ABC_enhancer'].to_list()
    g.vs['ABC_gene'] = vertices.loc[g.vs['name'],'ABC_gene'].to_list()
    g.vs['ABC_activity_base'] = vertices.loc[g.vs['name'],'activity_base'].to_list()

    es_hic = pd.DataFrame(g.es['weight'], columns=['hic'])
    es_hic.loc[es_hic['hic'].isna()] = es_hic['hic'].mean()
    hic_arr = es_hic['hic'].to_numpy()
    g.es['hic'] = hic_arr
    g.es['weight'] = np.max(hic_arr) - (hic_arr)  # negate the weight, so that shorted_path will find the path with largest hic value
    print("graph from pandas:", time.time()-tmp)

    tmp = time.time()
    for index, row in ABC.loc[ABC['chrEnhancer'] == chromosome].iterrows():
      ve = g.vs.select(ABC_enhancer=row['ABC_enhancer'])
      vg = g.vs.select(id=row['ABC_gene'])
      thisedge = None
      # if enhancer has no connection (hic value zero) to any element in the graph, the enhancer will not exist as a node.
      if (len(ve) != 1 or len(vg) != 1):
        continue 
      ve = ve[0]
      vg = vg[0] 
      # find shortest path from e to p with highest HiC*activity_base 
      results = g.get_shortest_paths(ve, to=vg, weights=g.es["weight"], output="epath")
      results_df = pd.DataFrame(results[0], columns=['path_edges'])
      if len(results) > 0: 
        if (len(results[0]) == 1) and (set([ve.index, vg.index]) == set([g.es[results[0][0]].source, g.es[results[0][0]].target])):   #skip if the shortest path is equal to the edge. 
          continue
        #print(results_df)
        last_to = ve
        for e in results[0]:
          #print(g.es[e]["weight"])
          if (g.vs[g.es[e].source] == last_to):
            current_from = g.vs[g.es[e].source]
            current_to = g.vs[g.es[e].target]
            last_to = g.vs[g.es[e].target]
          else: 
            current_from = g.vs[g.es[e].target]
            current_to = g.vs[g.es[e].source]
            last_to = g.vs[g.es[e].source]
          
          #print("from:") 
          #print(current_from) 
          #print("to:") 
          #print(current_to) 
          score = current_from['ABC_activity_base'] * g.es[e]["weight"]
          results_df['score'] = score

        #print(results_df)
        ABC.loc[index,'path.step'] = len(results[0])
        if len(results[0]) > 0:
          ABC.loc[index,'path.ABC.mean'] = results_df['score'].mean()
          ABC.loc[index,'path.ABC.product'] = results_df['score'].product()



 
    #print(ABC.loc[ABC['chrEnhancer'] == chromosome,['e0','e1', 'e2', 'e3', 'path.step','ABC.Score.Numerator','path.ABC.sum','path.ABC.mean','path.ABC.max','path.ABC.product']])

    #network = network_from_eps(chromosome, args.netdir)
    print("network_from_eps :", time.time()-tmp)
    print("total time :", time.time()-start)

  ABC_orig['path.step'] = ABC['path.step'] 
  ABC_orig['path.ABC.mean'] = ABC['path.ABC.mean'] 
  ABC_orig['path.ABC.product'] = ABC['path.ABC.product'] 
  ABC_orig.to_csv(os.path.join(args.dir, infile_base+".ABCpath.txt"),sep='\t')

