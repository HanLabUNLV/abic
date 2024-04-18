import pandas as pd
import numpy as np
import joblib
from sklearn.decomposition import NMF
import seaborn as sns
import matplotlib.pyplot as plt
import os.path
import argparse
import time










def DR_NMF_features_fit(TFmatrix,outdir, study_name_prefix):
    n_components = 12 
    init = "nndsvd"
    nmf_model = NMF(
        solver='cd',
        n_components=n_components,
        random_state=1,
        init=init,
        beta_loss="frobenius",
        alpha_W=0.005,
        alpha_H=0.00005,
        l1_ratio=0.7,
        max_iter=500
    )
    nmf_model.fit(TFmatrix)
    joblib.dump(nmf_model, outdir+'/'+study_name_prefix+'.gz')
    W = nmf_model.transform(TFmatrix)
    Wdf = pd.DataFrame(W, index=TFmatrix.index, columns =  ["TF_NMF" + str(i+1) for i in range(n_components)])
    Wdf.to_csv(outdir+'/'+study_name_prefix+'.TF.W.txt', index=False, sep='\t')
    H = nmf_model.components_
    Hdf = pd.DataFrame(H, columns=TFmatrix.columns)
    Hdf.to_csv(outdir+'/'+study_name_prefix+'.TF.H.txt', index=False, sep='\t')
    # heatmap for NMF features.
    sns.set(font_scale=2)
    fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(400, 100))
    hm = sns.heatmap(data = Hdf)
    plt.title("Heatmap NMF H")
    plt.savefig(outdir+'/'+study_name_prefix+'.heatmap.NMF.H.pdf')
    plt.close(fig)
    plt.show()
    return (Wdf)



def DR_NMF_features_transform(TFmatrix, NMFdir, outdir, prefix):

    nmf_dump = NMFdir+'/'+prefix+'.gz'
    nmf_model = joblib.load(nmf_dump)
    W = nmf_model.transform(TFmatrix)
    Wdf = pd.DataFrame(W, index=TFmatrix.index, columns =  ["TF_NMF" + str(i+1) for i in range(nmf_model.n_components)])
    Wdf.to_csv(outdir+'/'+prefix+'.TF.W.txt', index=False, sep='\t')
    H = nmf_model.components_
    Hdf = pd.DataFrame(H, columns=TFmatrix.columns)
    Hdf.to_csv(outdir+'/'+prefix+'.TF.H.txt', index=False, sep='\t')
    return (Wdf)







if __name__ == '__main__':
  start = time.time()
  tmp = time.time()
  parser = argparse.ArgumentParser()

  #Basic parameters
  parser.add_argument('--dir', required=True, help="output directory")
  parser.add_argument('--infile', required=True, help="gasperini infile name")
 
  args = parser.parse_args()

  infile_base = os.path.splitext(args.infile)[0]


  # ABC
  data_dir = args.dir+'/'
  Gasperini_atscale_ABC = pd.read_csv(data_dir+args.infile, sep='\t', index_col=0)
  Gasperini_atscale_ABC = Gasperini_atscale_ABC.loc[:,~Gasperini_atscale_ABC.columns.str.match("Unnamed")]

  # split train and test before DR.
  mask = np.isin(Gasperini_atscale_ABC['chr'], ['chr5','chr10','chr15','chr20']) # chromosomes for test dataset
  Xtrain = Gasperini_atscale_ABC[np.logical_not(mask)]
  #Xtrain.to_csv(data_dir+"Xtrain.txt", sep="\t")
  Xtest = Gasperini_atscale_ABC[mask]
  #Xtest.to_csv(data_dir+"Xtest.txt", sep="\t")
  ytrain = Xtrain['Significant']  
  #ytrain.to_csv(data_dir+"ytrain.txt", sep="\t")
  ytest = Xtest['Significant'] 
  #ytest.to_csv(data_dir+"ytest.txt", sep="\t")


  # fit DR on the train and apply to the test
  #enhancer_TF_pivot = pd.read_csv(data_dir+"Gasperini2019.enhancer.TF.txt", sep='\t', index_col='ID')
  #TSS_TF_pivot = pd.read_csv(data_dir+"Gasperini2019.TSS.TF.txt", sep='\t', index_col='gene')

  e_TFfeatures = Xtrain.filter(regex='(_e)').copy()
  if not e_TFfeatures.empty:
    #e_TFfeatures =  Xtrain.loc[:,list(enhancer_TF_pivot.columns)]
    NMFprefix='Gasperini2019.eTF.NMF'
    #pd.DataFrame(enhancer_TF_pivot.columns, columns=['TF']).to_csv(data_dir+NMFprefix+'.featureinput.txt', sep='\t')
    pd.DataFrame(e_TFfeatures.columns, columns=['TF']).to_csv(data_dir+NMFprefix+'.featureinput.txt', sep='\t')
    eTF_nmf_reduced_features = DR_NMF_features_fit(e_TFfeatures, data_dir, NMFprefix)
    eTF_nmf_reduced_features = eTF_nmf_reduced_features.add_suffix('_e')

  TSS_TFfeatures = Xtrain.filter(regex='(_TSS)').copy()
  if not TSS_TFfeatures.empty:
    #TSS_TFfeatures =  Xtrain.loc[:,list(TSS_TF_pivot.columns)]
    NMFprefix='Gasperini2019.TSSTF.NMF'
    #pd.DataFrame(TSS_TF_pivot.columns, columns=['TF']).to_csv(data_dir+NMFprefix+'.featureinput.txt', sep='\t')
    pd.DataFrame(TSS_TFfeatures.columns, columns=['TF']).to_csv(data_dir+NMFprefix+'.featureinput.txt', sep='\t')
    TSSTF_nmf_reduced_features = DR_NMF_features_fit(TSS_TFfeatures, data_dir, 'Gasperini2019.TSSTF.NMF')
    TSSTF_nmf_reduced_features = TSSTF_nmf_reduced_features.add_suffix('_TSS')

  if not e_TFfeatures.empty:
    Xtrain = pd.concat([Xtrain, eTF_nmf_reduced_features], axis=1)
  if not TSS_TFfeatures.empty:
    Xtrain = pd.concat([Xtrain, TSSTF_nmf_reduced_features], axis=1)

  Xtrain.to_csv(data_dir+infile_base+".train.txt", sep='\t')


  # save test
  Xtest.to_csv(data_dir+infile_base+".beforeNMF.txt", sep='\t')
  ytest.to_csv(data_dir+infile_base+".beforeNMF.target.txt", sep='\t')


