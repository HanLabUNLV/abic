import pandas as pd
import os.path
import argparse
import time








if __name__ == '__main__':
  start = time.time()
  tmp = time.time()
  parser = argparse.ArgumentParser()

  #Basic parameters
  parser.add_argument('--dir', required=True, help="output directory")
  parser.add_argument('--infile', required=False, help="gasperini infile name")
  parser.add_argument('--target', required=False, help="target, in case file is test")
 
  args = parser.parse_args()

  infile_base = os.path.splitext(args.infile)[0]
  ABC = pd.read_csv(args.dir+'/'+args.infile,sep='\t', header=0)

  if args.target is not None:
    target_base = os.path.splitext(args.target)[0]
    target = pd.read_csv(args.dir+'/'+args.target,sep='\t', header=0)

  mask = (ABC['Enhancer.count.near.TSS']<600) & (ABC['TSS.count.near.enhancer']<50)
  ABCsimple = ABC.loc[mask]
  ABCsimple.to_csv(args.dir+'/'+infile_base+".simple.txt", sep='\t')
  if args.target is not None:
    targetsimple = target.loc[mask]
    targetsimple.to_csv(args.dir+'/'+target_base+".simple.txt", sep='\t')
  mask = (ABC['Enhancer.count.near.TSS']>=600) | (ABC['TSS.count.near.enhancer']>=50)
  ABCcomplex = ABC.loc[mask]
  ABCcomplex.to_csv(args.dir+'/'+infile_base+".complex.txt", sep='\t')
  if args.target is not None:
    targetcomplex = target.loc[mask]
    targetcomplex.to_csv(args.dir+'/'+target_base+".complex.txt", sep='\t')

