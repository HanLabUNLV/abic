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
    parser.add_argument('--outdir', required=True, help="output directory")
    parser.add_argument('--chromosomes', default="all", help="chromosomes to make predictions for. Defaults to intersection of all chromosomes in --genes and --enhancers")
    parser.add_argument('--infile', required=False, help="gasperini infile name")

    return parser


def get_predict_argument_parser():
    parser = get_model_argument_parser()
    return parser






def network_from_eps(chromosome, netdir):

    print("reading network") 
    vertices = pd.read_csv(os.path.join(netdir, "vertices_nohic."+chromosome+".txt"), sep="\t", index_col=0)
    edgelist = pd.read_csv(os.path.join(netdir, "edgelist_nohic."+chromosome+".txt"), sep="\t", index_col=0)

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
    vertices_elements.to_csv(os.path.join(args.outdir, "vertices_nohic."+chromosome+".txt"), sep="\t", index=True)
    edgelist_elements_new.to_csv(os.path.join(args.outdir, "edgelist_nohic."+chromosome+".txt"), sep="\t", index=True)

    return edgelist_elements_new





if __name__ == '__main__':
  start = time.time()
  tmp = time.time()
  parser = get_predict_argument_parser()
  args = parser.parse_args()


  if not os.path.exists(args.outdir):
      os.makedirs(args.outdir)

  write_params(args, os.path.join(args.outdir, "parameters.predict.txt"))

  ABC = pd.read_csv(args.infile,sep='\t', header=0)
  ABC = ABC.loc[:,~ABC.columns.str.match("Unnamed")]
  ABC[['ABC_enhancer','ABC_gene']] = ABC['ABC.id'].str.split('_',expand=True)

  if args.chromosomes == "all":
      chromosomes = set(genes['chr']).intersection(set(enhancers['chr'])) 
  else:
      chromosomes = args.chromosomes.split(",")

  for chromosome in chromosomes:

    print("reading network") 
    vertices = pd.read_csv(os.path.join(args.netdir, "vertices_nohic."+chromosome+".txt"), sep="\t", index_col=0)
    edgelist = pd.read_csv(os.path.join(args.netdir, "edgelist_nohic."+chromosome+".txt"), sep="\t", index_col=0)
    vertices[['ABC.type','ABC_enhancer']] = vertices['id'].str.split('|',expand=True)
    vertices['ABC_gene'] = None
    vertices.loc[vertices['ABC_enhancer'].isnull(),'ABC_gene'] = vertices.loc[vertices['ABC_enhancer'].isnull(),'id'].to_list()

    t = time.time()

    g = Graph.TupleList(edgelist.itertuples(index=False), directed=False, weights=True )
    g.vs['id'] = vertices.loc[g.vs['name'],'id'].to_list()
    g.vs['ABC_enhancer'] = vertices.loc[g.vs['name'],'ABC_enhancer'].to_list()
    g.vs['ABC_gene'] = vertices.loc[g.vs['name'],'ABC_gene'].to_list()
    #g.write_edgelist("g.gene.enhancer.edgelist")
    #g.write_graphml("gene.enhancer."+chromosome+".graphml")
    print("graph from pandas:", time.time()-tmp)

    tmp = time.time()
    ABC_bychr = ABC.loc[ABC['chrEnhancer'] == chromosome]
    for index, row in df.iterrows():
      ve = g.vs.find(ABC_enhancer=row['ABC_enhancer'])
      vg = g.vs.find(id=row['ABC_gene'])

    #network = network_from_eps(chromosome, args.netdir)
    print("network_from_eps :", time.time()-tmp)
    print("total time :", time.time()-start)



