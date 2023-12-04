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
    parser.add_argument('--enhancers', required=True, help="Candidate enhancer regions. Formatted as the EnhancerList.txt file produced by run.neighborhoods.py")
    parser.add_argument('--genes', required=True, help="Genes to make predictions for. Formatted as the GeneList.txt file produced by run.neighborhoods.py")
    parser.add_argument('--outdir', required=True, help="output directory")
    parser.add_argument('--window', type=int, default=5000000, help="Make predictions for all candidate elements within this distance of the gene's TSS")
    parser.add_argument('--score_column', default='ABC.Score', help="Column name of score to use for thresholding")
    parser.add_argument('--threshold', type=float, required=True, default=.022, help="Threshold on ABC Score (--score_column) to call a predicted positive")
    parser.add_argument('--cellType', help="Name of cell type")
    parser.add_argument('--chrom_sizes', help="Chromosome sizes file")

    #hic
    #To do: validate params
    parser.add_argument('--HiCdir', default=None, help="HiC directory")
    parser.add_argument('--hic_resolution', type=int, help="HiC resolution")
    parser.add_argument('--tss_hic_contribution', type=float, default=100, help="Weighting of diagonal bin of hic matrix as a percentage of the maximum of its neighboring bins")
    parser.add_argument('--hic_pseudocount_distance', type=int, default=1e6, help="A pseudocount is added equal to the powerlaw fit at this distance")
    parser.add_argument('--hic_type', default = 'juicebox', choices=['juicebox','bedpe'], help="format of hic files")
    parser.add_argument('--hic_is_doubly_stochastic', action='store_true', help="If hic matrix is already DS, can skip this step")

    #Power law
    parser.add_argument('--scale_hic_using_powerlaw', action="store_true", help="Scale Hi-C values using powerlaw relationship")
    parser.add_argument('--hic_gamma', type=float, default=.87, help="powerlaw exponent of hic data. Must be positive")
    parser.add_argument('--hic_gamma_reference', type=float, default=.87, help="powerlaw exponent to scale to. Must be positive")

    #Genes to run through model
    parser.add_argument('--run_all_genes', action='store_true', help="Do not check for gene expression, make predictions for all genes")
    parser.add_argument('--expression_cutoff', type=float, default=1, help="Make predictions for genes with expression higher than this value")
    parser.add_argument('--promoter_activity_quantile_cutoff', type=float, default=.4, help="Quantile cutoff on promoter activity. Used to consider a gene 'expressed' in the absence of expression data")

    #Output formatting
    parser.add_argument('--make_all_putative', action="store_true", help="Make big file with concatenation of all genes file")
    parser.add_argument('--use_hdf5', action="store_true", help="Write AllPutative file in hdf5 format instead of tab-delimited")

    #Other
    parser.add_argument('--tss_slop', type=int, default=500, help="Distance from tss to search for self-promoters")
    parser.add_argument('--chromosomes', default="all", help="chromosomes to make predictions for. Defaults to intersection of all chromosomes in --genes and --enhancers")
    parser.add_argument('--include_chrY', '-y', action='store_true', help="Make predictions on Y chromosome")

    return parser


def get_predict_argument_parser():
    parser = get_model_argument_parser()
    return parser


def validate_args(args):
    if args.HiCdir and args.hic_type == 'juicebox':
        assert args.hic_resolution is not None, 'HiC resolution must be provided if hic_type is juicebox'

    if not args.HiCdir:
        print("WARNING: Hi-C directory not provided. Model will only compute ABC score using powerlaw!")


#python src/generate_networks.py --enhancers ./ABC_output/Neighborhoods/EnhancerList.txt --genes ./ABC_output/Neighborhoods/GeneList.txt --outdir . --threshold 0.0 --HiCdir raw_data/hic/5kb_resolution_intrachromosomal/ --hic_resolution 5000 --chromosomes chrX 

#cat chromosomes.txt | parallel python src/generate_networks.py --enhancers ./ABC_output/Neighborhoods/EnhancerList.txt --genes ./ABC_output/Neighborhoods/GeneList.txt --outdir . --threshold 0.0 --HiCdir raw_data/hic/5kb_resolution_intrachromosomal/ --hic_resolution 5000 --chromosomes {}


if __name__ == '__main__':

  start = time.time()
  tmp = time.time()
  parser = get_predict_argument_parser()
  args = parser.parse_args()

  validate_args(args)

  if not os.path.exists(args.outdir):
      os.makedirs(args.outdir)

  write_params(args, os.path.join(args.outdir, "parameters.predict.txt"))

  if args.chromosomes == "all":
      chromosomes = set(genes['chr']).intersection(set(enhancers['chr'])) 
      if not args.include_chrY:
          chromosomes.discard('chrY')
  else:
      chromosomes = args.chromosomes.split(",")

  for chromosome in chromosomes:
    network = network_from_hic(chromosome, args)
    print("network_from_hic :", time.time()-tmp)
    print("total time :", time.time()-start)
    network = network_from_gene_enhancer(chromosome, network, args)
    print("network_from_gene_enhancer :", time.time()-tmp)
    print("total time :", time.time()-start)
    network = network_remove_hic(chromosome, network, args)
    print("network_remove_hic :", time.time()-tmp)
    print("total time :", time.time()-start)



