import pandas as pd
import numpy as np
import joblib
from sklearn.decomposition import NMF
import seaborn as sns
import matplotlib.pyplot as plt
import os.path
import argparse
import time











def DR_NMF_features_transform(TFmatrix, NMF_dir, data_dir, NMFprefix):

    nmf_dump = NMF_dir+'/'+NMFprefix+'.gz'
    nmf_model = joblib.load(nmf_dump)
    W = nmf_model.transform(TFmatrix)
    Wdf = pd.DataFrame(W, index=TFmatrix.index, columns =  ["TF_NMF" + str(i+1) for i in range(nmf_model.n_components)])
    #Wdf.to_csv(data_dir+'/'+NMFprefix+'.TF.W.txt', index=False, sep='\t')
    H = nmf_model.components_
    Hdf = pd.DataFrame(H, columns=TFmatrix.columns)
    #Hdf.to_csv(data_dir+'/'+NMFprefix+'.TF.H.txt', index=False, sep='\t')
    return (Wdf)







if __name__ == '__main__':
  start = time.time()
  tmp = time.time()
  parser = argparse.ArgumentParser()

  #Basic parameters
  parser.add_argument('--dir', required=True, help="output directory")
  parser.add_argument('--infile', required=True, help="test data infile name")
  parser.add_argument('--NMFdir', required=True, help="NMF model dir")
 
  args = parser.parse_args()

  infile_base = os.path.splitext(args.infile)[0]
  infile_base = infile_base.removesuffix('.beforeNMF')
  infile_base = infile_base.removesuffix('.test')


  # ABC
  data_dir = args.dir+'/'
  NMF_dir = args.NMFdir+'/'
  Xtest = pd.read_csv(data_dir+args.infile, sep='\t', index_col=0)
  ytest = Xtest['Significant'] 


  # apply NMF to test
  e_TFfeatures = Xtest.filter(regex='(_e)').copy()
  NMFprefix='Gasperini2019.eTF.NMF'
  eNMFinputfeatures = pd.read_csv(NMF_dir+NMFprefix+'.featureinput.txt', sep='\t')
  missingTF = np.setdiff1d(eNMFinputfeatures['TF'].tolist(), e_TFfeatures.columns.tolist()).tolist()
  for TF in missingTF:
    e_TFfeatures[TF] = 0 
  e_TFfeatures = e_TFfeatures[eNMFinputfeatures['TF']]
  e_TFfeatures = e_TFfeatures.fillna(0)
  eTF_nmf_reduced_features = DR_NMF_features_transform(e_TFfeatures, NMF_dir, data_dir, NMFprefix)
  eTF_nmf_reduced_features = eTF_nmf_reduced_features.add_suffix('_e')

  TSS_TFfeatures = Xtest.filter(regex='(_TSS)').copy()
  NMFprefix='Gasperini2019.TSSTF.NMF'
  TSSNMFinputfeatures = pd.read_csv(NMF_dir+NMFprefix+'.featureinput.txt', sep='\t')
  missingTF = np.setdiff1d(TSSNMFinputfeatures['TF'].tolist(), TSS_TFfeatures.columns.tolist()).tolist()
  for TF in missingTF:
    TSS_TFfeatures[TF] = 0 
  TSS_TFfeatures = TSS_TFfeatures[TSSNMFinputfeatures['TF']]
  TSSTF_nmf_reduced_features = DR_NMF_features_transform(TSS_TFfeatures, NMF_dir, data_dir, NMFprefix)
  TSSTF_nmf_reduced_features = TSSTF_nmf_reduced_features.add_suffix('_TSS')

  Xtest = pd.concat([Xtest, eTF_nmf_reduced_features, TSSTF_nmf_reduced_features], axis=1)
  Xtest = Xtest.loc[:,~Xtest.columns.str.match("Unnamed")]
  Xtest.to_csv(data_dir+infile_base+".test.txt", sep='\t')
  ytest.to_csv(data_dir+infile_base+".test.target.txt", sep='\t')


