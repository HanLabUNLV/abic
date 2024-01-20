import argparse
import numpy as np
import time, os
import joblib
from  scipy.stats import rankdata as rank
import matplotlib.pyplot as plt
from statistics import mean, stdev
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import preprocessing
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, NMF
from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import precision_recall_curve, auc, average_precision_score, balanced_accuracy_score, make_scorer, f1_score
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
import pandas as pd
import xgboost as xgb
from collections import Counter
from pathlib import Path

RANDOM_SEED = 42

tstart = time.time()
pid = os.getpid()




# python src/apply_model.py  --dir /data8/han_lab/mhan/abcd/data/ --outdir /data8/han_lab/mhan/abcd/run --modelfile /data8/han_lab/mhan/abcd/data/trained_models/mira_data/save38667.xgb.0.json --scalerfile /data8/han_lab/mhan/abcd/run2/38667.scaler.0.gz --features /data8/han_lab/mhan/abcd/run2/38667.Xtest.0.txt --targets /data8/han_lab/mhan/abcd/run2/38667.ytest.0.txt --featurenames /data8/han_lab/mhan/abcd/run2/featurenames38667.xgb.0.txt

# python -i src/apply_model.py --dir /data8/han_lab/mhan/abcd/data/ --outdir /data8/han_lab/mhan/abcd/run2 --modelfile /data8/han_lab/mhan/abcd/data/trained_models/mira_data/save38667.xgb.0.json --scalerfile /data8/han_lab/mhan/abcd/run2/38667.scaler.0.gz --features /data8/han_lab/mhan/abcd/data/Fulco/Fulco2019.CRISPR.ABC.TF.txt --targets /data8/han_lab/mhan/abcd/data/Fulco/Fulco2019.CRISPR.ABC.TF.target.txt --featurenames /data8/han_lab/mhan/abcd/run2/featurenames38667.xgb.0.txt

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--outdir', default='.', help="directory containing edgelist and vertices files")
  parser.add_argument('--chr', default='all', help="chromosome")
  parser.add_argument('--scalerfile', required=True, help="scaler dumped during training")
  parser.add_argument('--modelfile', required=True, help="model saved after training in json file")
  parser.add_argument('--features', required=True, help="feature matrix ")
  parser.add_argument('--targets', help="target truth to compare for evaluation")

  args=parser.parse_args()
  pid = os.getpid()

  chromosome = args.chr
  outdir = args.outdir
  scalerfile = args.scalerfile
  modelfile = args.modelfile
  featurefile = args.features
  targets = args.targets
  prefix = Path(modelfile).stem
  Path(outdir).mkdir(parents=True, exist_ok=True)

  filenamesuffix = ''

  #################################
  #import our data, then format it #
  ##################################

  X_test = pd.read_csv(featurefile, sep='\t', index_col=0)
  X_test = X_test.loc[:,~X_test.columns.str.match("Unnamed")]

  # same feature engineering as training
#####################

  # same feature engineering as training
  ActivityFeatures = X_test[['GeneSymbol', 'TargetGeneExpression', 'H3K27ac.RPKM.quantile.TSS1Kb', 'H3K4me3.RPKM.quantile.TSS1Kb', 'H3K27me3.RPKM.quantile.TSS1Kb']].copy()
  ActivityFeatures.rename(columns={'H3K27ac.RPKM.quantile.TSS1Kb':'H3K27ac.RPKM.quantile_TSS', 'H3K4me3.RPKM.quantile.TSS1Kb':'H3K4me3.RPKM.quantile_TSS', 'H3K27me3.RPKM.quantile.TSS1Kb':'H3K27me3.RPKM.quantile_TSS'}, inplace=True)
  ActivityFeatures = ActivityFeatures.dropna()
  ActivityFeatures['TargetGeneExpression'] = np.log1p(ActivityFeatures['TargetGeneExpression'])
  hicfeatures = X_test[['Enhancer.count.near.TSS', 'mean.contact.to.TSS', 'max.contact.to.TSS', 'total.contact.to.TSS' ]].copy()
  hicfeatures = hicfeatures.dropna()
  TFfeatures = X_test.filter(regex='(_e)|(_TSS)|(NMF)').copy()
  TFfeatures = TFfeatures.dropna()

  # merge features
  features = ActivityFeatures.copy()
  features = pd.merge(features, hicfeatures, left_index=True, right_index=True)
  features = pd.merge(features, TFfeatures, left_index=True, right_index=True)
  data = features.copy()

  ActivityFeatures = data.iloc[:, :ActivityFeatures.shape[1]]
  hicfeatures = data.iloc[:, ActivityFeatures.shape[1]:ActivityFeatures.shape[1]+hicfeatures.shape[1]]
  TFfeatures = data.iloc[:, ActivityFeatures.shape[1]+hicfeatures.shape[1]:ActivityFeatures.shape[1]+hicfeatures.shape[1]+TFfeatures.shape[1]]
  X_test.filter(items = data.index,axis=0).to_csv(outdir+'/applyinput.txt', sep='\t') # save the input file after filtering rows and before filtering columns. 
  X_test = data


  y_test = pd.read_csv(targets, sep='\t', index_col=0)
  y_test = y_test.loc[:,~y_test.columns.str.match("Unnamed")]
  y_test = y_test.filter(items=data.index,axis=0)
  y_test = y_test['Significant'].astype(int)
  print(y_test)
  print(str(sum(y_test))+'/'+str(len(y_test)))



  # load preprocessor 
  scaler = joblib.load(scalerfile)
  featurenames = scaler.get_feature_names_out()
  featurenames = np.insert(featurenames, 0, 'ABC.id', axis=0)
  print(featurenames)
  X_test = X_test.reindex(columns = featurenames)
  # save features
  X_test.to_csv(outdir+'/Xfeatures.'+prefix+'.txt', sep='\t')
  y_test.to_csv(outdir+'/ytarget.'+prefix+'.txt', sep='\t')
  X_test = X_test.drop(columns = ['ABC.id'])
  # transform by scaling
  X_test = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns)
  print(X_test.columns)




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

  assert not X_test.isnull().values.any()

     
 
  dtest = xgb.DMatrix(X_test) 
  y_pred_prob = xgb_clf_tuned_2.predict(dtest, ntree_limit=best_iteration)
  print(y_pred_prob)

  # Data to plot precision - recall curve
  precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob, pos_label = 1)
  #print(precision)
  #print(recall)
  pr = pd.DataFrame({'precision':precision,'recall':recall,'threshold':np.append(thresholds,None)})
  outfile = os.path.basename(targets)
  pr.to_csv(outdir+'/pr_curve.'+prefix+'.txt', sep='\t')
  # find threshold at recall == 0.70
  recallmin = recall-0.70
  ix = np.where(recallmin >= 0, recallmin, np.inf).argmin()
  y_pred = [value>=thresholds[ix] for value in y_pred_prob]
  test_pd = pd.DataFrame({'y_pred':y_pred, 'y_prob':y_pred_prob}, index=y_test.index)
  y_res = pd.merge(y_test, test_pd, left_index=True, right_index=True)
  y_res.to_csv(outdir+'/confusion.'+prefix+'.txt', sep='\t')


  aucpr = auc(recall, precision)
  average_precision = average_precision_score(y_test, y_pred_prob)
  bal_accuracy = balanced_accuracy_score(y_test, y_pred)
  test_f1_score = f1_score(y_test, y_pred)
  metric = {}
  metric['aucpr'] = [aucpr]
  metric['average_precision'] = [average_precision]
  metric['bal_accuracy'] = [bal_accuracy]
  metric['f1_score'] = [test_f1_score]
  metric['best_iteration'] = best_iteration
  evaluation = pd.DataFrame.from_dict(metric)
  evaluation.to_csv(outdir+'/evaluation.'+prefix+'.txt', sep='\t')

  #fnames = data1.loc[:,data1.columns != 'sig'].columns[[int(x[1:]) for x in xgb_clf_tuned_2[:-1].get_feature_names_out()]].tolist()
    #fweights = clf.named_steps[clf_label].coef_.ravel()
  #frank = rank(abs(clf.named_steps[clf_label].coef_.ravel()))
  #for i in range(0,len(fnames)):
  #    feature = fnames[i]
  #    if feature not in feature_ranks:
  #        feature_ranks[feature] = fweights[i]*frank[i]/len(frank)
  #    else:
  #        feature_ranks[feature] += fweights[i]*frank[i]/len(frank)
  #plot pr curve
  #plot_pr_curves([xgb_clf_tuned_2], X_test, y_test, abc_score[test_idx], distance[test_idx], outer_index, 'data/pr_curves_c/xgb/')


#    dump_list = xgb_clf_tuned_2.get_booster().get_dump()
#    num_trees = len(dump_list)
#    fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(100, 100))
#    for i in range(num_trees): 
#      plt.rcParams.update({'font.size':12})
#      plt.rcParams['figure.figsize'] = 80,50
#      plot_tree(xgb_clf_tuned_2, num_trees=i, rankdir='LR')
#      plt.savefig(outdir+'/'+'tree.'+filenamesuffix+'.'+str(pid)+'_'+str(i)+'.pdf')
#      plt.rcParams.update({'font.size':10})
#      plt.cla()
#      if (i > 20): break;
#    plt.close()
# 


  pd.set_option('display.max_columns', None) 
  print(evaluation) 

  fscore = xgb_clf_tuned_2.get_score( importance_type='total_gain')
  #print(fscore)
  importances = pd.DataFrame.from_dict(fscore, orient='index',columns=['fscore'])  
  importances.insert(0, 'feature', importances.index)
  importances.sort_values(by=['fscore'],ascending=False,ignore_index=True).to_csv(outdir+'/total_gain.importance.'+prefix+'.txt', index=False, sep='\t')
  #print(importances)


  #  evaluation.to_csv('data/trained_models/mira_data/'+str(pid)+'.evaluation.txt')
  #save best estimators 
#  for est in best_estimators:
#      joblib.dump(best_estimators[est]['clf'], 'data/trained_models/mira_data/'+str(pid)+'.'+est+'.pkl')
  print('Total runtime: ' + str(time.time() - tstart))    

