import argparse
import time, os
import joblib
os.environ["OMP_NUM_THREADS"] = "8" # export OMP_NUM_THREADS=8
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
import optuna
import glob
import shap
from BorutaShap import BorutaShap
import json
from pathlib import Path




RANDOM_SEED = 42

tstart = time.time()
pid = os.getpid()





# python -i src/runshap.py --modeldir run.newtrain/run.newtrain10 --studyname newtrain10 --outdir run.applymodel/newtrain10/
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--modeldir', required=True, help="directory containing data and model files")
  parser.add_argument('--studyname', required=True, help="studyname prefix for data files") 
  parser.add_argument('--outdir', default='.', help="directory to save shap results")
  parser.add_argument('--chr', default='all', help="chromosome")
  parser.add_argument('--nfold', default=4, help="n outer fold")
  parser.add_argument('--ID', default='ABC.id', help="n outer fold")

  args=parser.parse_args()
  pid = os.getpid()

  nfold = int(args.nfold)
  modeldir = args.modeldir
  studyname = args.studyname
  outdir = args.outdir
  chromosome = args.chr
  scalerfile = ''
  modelfile = ''
  features = ''
  targets = ''
  featurenames = ''
  Path(outdir).mkdir(parents=True, exist_ok=True)

  filenamesuffix = ''


  plt.rcParams.update({'font.size': 14})

  #################################
  #import our data, then format it #
  ##################################

#  eroles = pd.read_csv('data/Gasperini/Gasperini2019.at_scale.erole.txt', sep="\t", index_col=0)
#  print(eroles)
  outer_index = 0
  modelFilenamesList = glob.glob(modeldir+'/'+studyname+'.save*.json')
  configFilenamesList = glob.glob(modeldir+'/'+studyname+'.config*.json')
  scalerFilenamesList = glob.glob(modeldir+'/'+studyname+'.scaler*.gz')

  list_importances = []
  list_shap_values = []
  list_shap_interactions = []
  X_test = pd.read_csv(modeldir +'/'+studyname+'.Xtest.0.txt', sep='\t', index_col=0)
  X_test = X_test.drop(columns = [IDcolname])
  print(X_test)
  X = pd.DataFrame(columns = X_test.columns)
  X_scaled = pd.DataFrame(columns = X_test.columns)
  print(X)
  IDdf = pd.DataFrame(columns = ['ID'])
  y = pd.DataFrame(columns = ['Significant'])

  for outer_index in range(nfold):
      modelfile = modelFilenamesList[outer_index]
      print(modelfile)

      X_test = pd.read_csv(modeldir +'/'+studyname+'.Xtest.'+str(outer_index)+'.txt', sep='\t', index_col=0)
      X_test = X_test.drop(columns = [IDcolname])
      y_test = pd.read_csv(modeldir +'/'+studyname+'.ytest.'+str(outer_index)+'.txt', sep='\t', index_col=0)
      IDdf_test = pd.read_csv(modeldir +'/'+studyname+'.IDtest.'+str(outer_index)+'.txt', sep='\t', index_col=0)

      X = pd.concat([X, X_test])
      y = pd.concat([y, y_test])
      IDdf = pd.concat([IDdf, IDdf_test])
      


      # copied from applymodel start

      # load preprocessor preprocess x_test 
      scaler = joblib.load(scalerFilenamesList[outer_index])
      features = scaler.get_feature_names_out()
      print(features)
      X_test = X_test.reindex(columns = scaler.get_feature_names_out())
      X_test = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns)
      print(X_test.columns)
      X_scaled = pd.concat([X_scaled, X_test])

      # preprocess y_test
      y_test = y_test.loc[:,~y_test.columns.str.match("Unnamed")]
      y_test = y_test['Significant'].astype(int)
      print(y_test)
      print(str(sum(y_test))+'/'+str(len(y_test)))




      # load model
      xgb_clf_tuned_2 = xgb.Booster()
      xgb_clf_tuned_2.load_model(modelfile)    
      print(xgb_clf_tuned_2)
      best_iteration = xgb_clf_tuned_2.best_iteration
      print(best_iteration)
      lst_vars_in_model = xgb_clf_tuned_2.feature_names
      print(lst_vars_in_model)
      featurenames = pd.DataFrame({"features":lst_vars_in_model})
      print(featurenames)
      X_test = X_test.reindex(columns = featurenames['features'])
        
     
      dtest = xgb.DMatrix(X_test) 
      y_pred_prob = xgb_clf_tuned_2.predict(dtest, ntree_limit=best_iteration)
      print(y_pred_prob)
      y_pred = [round(value) for value in y_pred_prob]
      test_pd = pd.DataFrame({'y_pred':y_pred, 'y_prob':y_pred_prob}, index=y_test.index)
      y_res = pd.merge(y_test, test_pd, left_index=True, right_index=True)
      y_res.to_csv(outdir+'/shap_confusion.'+str(outer_index)+'.txt', index=False, sep='\t')    

      # SHAP values
      shap_values = xgb_clf_tuned_2.predict(dtest, ntree_limit=best_iteration, pred_contribs=True)
      print(shap_values)
      print(shap_values.shape)
      list_shap_values.append(shap_values)

      # SHAP interactions
      shap_interactions = xgb_clf_tuned_2.predict(dtest, ntree_limit=best_iteration, pred_interactions=True)
      print(shap_interactions)
      print(shap_interactions.shape)
      list_shap_interactions.append(shap_interactions)


      # copied from apply model end


  
      fscore = xgb_clf_tuned_2.get_score( importance_type='total_gain')
      #print(fscore)
      importances = pd.DataFrame.from_dict(fscore, orient='index',columns=['fscore'])  
      importances.insert(0, 'feature', importances.index)
      print(importances)
      list_importances.append(importances)

      # this retrieves all booster and non-booster parameters
      with open(configFilenamesList[outer_index], 'r') as config_file:
          config = json.load(config_file) # your xgb booster object
      stack = [config]
      params_dict = {}
      while stack:
          obj = stack.pop()
          for k, v in obj.items():
              if k.endswith('_param'):
                  for p_k, p_v in v.items():
                      params_dict[p_k] = p_v
              elif isinstance(v, dict):
                  stack.append(v)
      print(params_dict)
      # retrieve all parameter values from xgb.train in param search dict

      xgb_clf_tuned_scikit = xgb.XGBClassifier(
        #colsample_bytree=0.8, 
        colsample_bytree=float(params_dict['colsample_bytree']),
        subsample=float(params_dict['subsample']),
        gamma=float(params_dict['gamma']),
        reg_alpha=float(params_dict['alpha']), 
        reg_lambda=float(params_dict['lambda']), 
        learning_rate=float(params_dict['eta']),
        max_delta_step=1,
        max_depth=int(params_dict['max_depth']),
        min_child_weight=int(params_dict['min_child_weight']),
        scale_pos_weight=float(params_dict['scale_pos_weight']), 
        n_estimators=int(params_dict['num_trees']),
        random_state = RANDOM_SEED
      )   
      print(xgb_clf_tuned_scikit)

      # boruta shap
      Feature_Selector = BorutaShap(model=xgb_clf_tuned_scikit,
                            importance_measure='shap',
                            classification=True, 
                            percentile=80, pvalue=0.1)
      Feature_Selector.fit(X=X_test, y=y_test, n_trials=100, sample=True,
                 train_or_test = 'train', normalize=True, verbose=True)
      fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(400, 50))
      Feature_Selector.plot(which_features='all')
      plt.savefig(outdir+'/'+'boruta.'+str(outer_index)+'.pdf')
      plt.close()
      Feature_Selector.results_to_csv(filename=outdir+'/'+'feature_importance.'+str(outer_index)+'.txt')
      features_to_remove = pd.DataFrame({"features":Feature_Selector.features_to_remove})
      print(features_to_remove)
      features_to_remove.to_csv(outdir+'/'+'features_to_remove.'+str(outer_index)+'.txt', index=False, sep='\t')




  X.insert(0, "ID", IDdf['ID'])
  X.to_csv(outdir+'/X.'+studyname+'.txt', sep='\t') 
  X = X.drop(columns = ["ID"])
  IDdf.to_csv(outdir+'/ID.'+studyname+'.txt', sep='\t') 

  importance_values = pd.DataFrame(columns = ['feature', 'fscore'])
  for i in range(nfold):
      importance_values = pd.concat([importance_values, list_importances[i]])
  importance_values.sort_values(by=['fscore'],ascending=False,ignore_index=True).to_csv(outdir+'/importance.'+studyname+'.txt', index=False, sep='\t')

  # combining shap values from all folds
  shap_values = np.array(list_shap_values[0])
  for i in range(1, nfold):
      shap_values = np.concatenate((shap_values,np.array(list_shap_values[i])), axis=0)
  shap_values = shap_values[:,:-1]
  shap_pandas = pd.DataFrame(shap_values, columns = X.columns, index=X.index)
  print(shap_values)
  print(shap_values.shape)
  print(shap_pandas)
  shap_pandas.insert(0, "ID", IDdf['ID'])
  shap_pandas.to_csv(outdir+'/shap_values.'+studyname+'.txt', sep='\t')
  shap_pandas = shap_pandas.drop(columns = ['ID'])

  # summary plot
  fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(100, 100))
  shap.summary_plot(shap_pandas.to_numpy(), X, max_display=100)
  plt.savefig(outdir+'/'+'shap.'+studyname+'.shap.pdf')
  fig.clf()
  plt.close(fig)
  fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(100, 100))
  shap.summary_plot(shap_pandas.to_numpy(), X_scaled, max_display=100)
  plt.savefig(outdir+'/'+'shap.'+studyname+'.shap.scaled.pdf')
  fig.clf()
  plt.close(fig)

  # heatmap
  #fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(100, 100))
  #shap.plots.heatmap(shap_pandas.to_numpy())
  #plt.savefig(outdir+'/'+'shap.'+studyname+'.shap.pdf')
  #fig.clf()
  #plt.close(fig)

  # plot dependency for top features 
  shap_order = shap_pandas.abs().mean().sort_values(ascending=[False])
  #for i in range(0,18):
  #  for j in range(0,18):
  for i in range(0,45):
    for j in range(0,25):
      fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(100, 100))
      shap.dependence_plot(shap_order.index[i], shap_values=shap_values, features=X, interaction_index=shap_order.index[j])
      plt.savefig(outdir+'/'+'dependence.'+shap_order.index[i]+'x'+shap_order.index[j]+'.pdf')
      plt.cla()
      fig.clf()
      plt.close(fig)



 
  # interpret by contact
  hic_median = 0.007
  X_hicontact = X.loc[X['hic_contact'] >= hic_median].copy()
  X_lowcontact = X.loc[X['hic_contact'] < hic_median*0.1].copy()
  y_hicontact = y.loc[X_hicontact.index].copy()
  y_hicontact = y_hicontact['Significant']
  print(y_hicontact)
  print(str(sum(y_hicontact))+'/'+str(len(y_hicontact)))
  y_lowcontact = y.loc[X_lowcontact.index].copy()
  y_lowcontact = y_lowcontact['Significant']
  print(y_lowcontact)
  print(str(sum(y_lowcontact))+'/'+str(len(y_lowcontact)))
  shap_hicontact = shap_pandas.loc[(X['hic_contact'] >= hic_median)].copy()
  shap_lowcontact = shap_pandas.loc[(X['hic_contact'] < hic_median*0.1)].copy()
  print(shap_hicontact)
  print(X_hicontact['hic_contact'])
  print(shap_lowcontact)
  print(X_lowcontact['hic_contact'])

  fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(100, 100))
  shap.summary_plot(shap_hicontact.to_numpy(), X_hicontact, max_display=100)
  plt.savefig(outdir+'/'+'shap_hicontact.'+studyname+'.shap.pdf')
  fig.clf()
  plt.close(fig)

  fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(100, 100))
  shap.summary_plot(shap_lowcontact.to_numpy(), X_lowcontact, max_display=100)
  plt.savefig(outdir+'/'+'shap_lowcontact.'+studyname+'.shap.pdf')
  fig.clf()
  plt.close(fig)

  shap_hicontact_pos = shap_hicontact.loc[(y_hicontact==1)].copy()
  shap_lowcontact_pos = shap_lowcontact.loc[(y_lowcontact==1)].copy()
  X_hicontact_pos = X_hicontact.loc[(y_hicontact==1)].copy()
  X_lowcontact_pos = X_lowcontact.loc[(y_lowcontact==1)].copy()
  shap_hicontact_pos.to_csv(outdir+'/shap_hicontact_pos.'+studyname+'.txt', index=False, sep='\t')
  shap_lowcontact_pos.to_csv(outdir+'/shap_lowcontact_pos.'+studyname+'.txt', index=False, sep='\t')

  fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(100, 100))
  shap.summary_plot(shap_hicontact_pos.to_numpy(), X_hicontact_pos, max_display=100)
  plt.savefig(outdir+'/'+'shap_hicontact_pos.'+studyname+'.shap.pdf')
  fig.clf()
  plt.close(fig)

  fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(100, 100))
  shap.summary_plot(shap_lowcontact_pos.to_numpy(), X_lowcontact_pos, max_display=100)
  plt.savefig(outdir+'/'+'shap_lowcontact_pos.'+studyname+'.shap.pdf')
  fig.clf()
  plt.close(fig)

#
#  # interpret by erole
#  e1index = eroles.loc[eroles['erole']==100].index 
#  X_direct = X.loc[e1index.intersection(X.index)].copy()
#  e2plusindex = eroles.loc[eroles['erole']!=100].index 
#  X_indirect = X.loc[e2plusindex.intersection(X.index)].copy()
#  y_direct = y.loc[X_direct.index].copy()
#  y_direct = y_direct['Significant']
#  print(y_direct)
#  print(str(sum(y_direct))+'/'+str(len(y_direct)))
#  y_indirect = y.loc[X_indirect.index].copy()
#  y_indirect = y_indirect['Significant']
#  print(y_indirect)
#  print(str(sum(y_indirect))+'/'+str(len(y_indirect)))
#  shap_direct = shap_pandas.loc[e1index.intersection(shap_pandas.index)].copy()
#  shap_indirect = shap_pandas.loc[e2plusindex.intersection(shap_pandas.index)].copy()
#  print(shap_direct)
#  print(X_direct['hic_contact'])
#  print(shap_indirect)
#  print(X_indirect['hic_contact'])
#
#  fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(100, 100))
#  shap.summary_plot(shap_direct.to_numpy(), X_direct, max_display=100)
#  plt.savefig(outdir+'/'+'shap_direct.'+studyname+'.shap.pdf')
#  plt.close()
#
#  fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(100, 100))
#  shap.summary_plot(shap_indirect.to_numpy(), X_indirect, max_display=100)
#  plt.savefig(outdir+'/'+'shap_indirect.'+studyname+'.shap.pdf')
#  plt.close()
#
#  shap_direct_pos = shap_direct.loc[(y_direct==1)].copy()
#  shap_indirect_pos = shap_indirect.loc[(y_indirect==1)].copy()
#  X_direct_pos = X_direct.loc[(y_direct==1)].copy()
#  X_indirect_pos = X_indirect.loc[(y_indirect==1)].copy()
#  shap_direct_pos.to_csv(outdir+'/shap_direct_pos.'+studyname+'.txt', index=False, sep='\t')
#  shap_indirect_pos.to_csv(outdir+'/shap_indirect_pos.'+studyname+'.txt', index=False, sep='\t')
#
#  fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(100, 100))
#  shap.summary_plot(shap_direct_pos.to_numpy(), X_direct_pos, max_display=100)
#  plt.savefig(outdir+'/'+'shap_direct_pos.'+studyname+'.shap.pdf')
#  plt.close()
#
#  fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(100, 100))
#  shap.summary_plot(shap_indirect_pos.to_numpy(), X_indirect_pos, max_display=100)
#  plt.savefig(outdir+'/'+'shap_indirect_pos.'+studyname+'.shap.pdf')
#  plt.close()
#
#
#
#
  # interaction
  for i in range(nfold):
    shap_interactions = list_shap_interactions[i][:,:-1,:-1]
    print(shap_interactions.shape)
    m,n,r = shap_interactions.shape
    out_arr = np.column_stack((np.repeat(np.arange(m),n),shap_interactions.reshape(m*n,-1)))
    shap_x_pandas = pd.DataFrame(out_arr)
    print(shap_x_pandas)
    shap_x_pandas.to_csv(outdir+'/shap_interactions.'+studyname+'.'+str(i)+'.txt', index=False, sep='\t')

    X_test = pd.read_csv(modeldir +'/'+studyname+'.Xtest.'+str(i)+'.txt', sep='\t', index_col=0)
    fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(100, 100))
    shap.summary_plot(shap_interactions, X.loc[X_test.index])
    plt.savefig(outdir+'/'+'shap_interactions.'+studyname+'.'+str(i)+'.pdf')
    fig.clf()
    plt.close(fig)


print('Total runtime: ' + str(time.time() - tstart))    

