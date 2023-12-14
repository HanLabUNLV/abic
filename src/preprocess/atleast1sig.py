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
 
  args = parser.parse_args()

  infile_base = os.path.splitext(args.infile)[0]

  ABC = pd.read_csv(args.dir+'/'+args.infile,sep='\t', header=0)
  ABC_atleast1sig = ABC.loc[ABC['atleast1Sig'] == True,]
  ABC_atleast1sig.to_csv(args.dir+'/'+infile_base+".atleast1sig.txt",sep='\t')


