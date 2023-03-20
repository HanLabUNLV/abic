import argparse
import time, os
import joblib
os.environ["OMP_NUM_THREADS"] = "8" # export OMP_NUM_THREADS=8
import numpy as np
from  scipy.stats import rankdata as rank
import matplotlib.pyplot as plt
from statistics import mean, stdev
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import preprocessing
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold, cross_val_score
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer 
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA, NMF
from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import precision_recall_curve, auc, average_precision_score, balanced_accuracy_score, make_scorer, f1_score
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
import pandas as pd
import xgboost as xgb
import optuna
from optuna import create_study, logging
from optuna.pruners import MedianPruner
from optuna.integration import XGBoostPruningCallback
from collections import Counter
import glob
import shap
from BorutaShap import BorutaShap
import json


RANDOM_SEED = 42

tstart = time.time()
pid = os.getpid()





#helper class that allows you to iterate over multiple classifiers within the nested for loop
class EstimatorSelectionHelper:
    def __init__(self, models, params, storage=None, dimrs=None, dimr_params=None):
        #if not set(models.keys()).issubset(set(params.keys())):
        #    missing_params = list(set(models.keys()) - set(params.keys()))
        #    raise ValueError("Some estimators are missing parameters: %s" % missing_params)
        self.models = models
        self.params = params
        self.dimrs = dimrs
        self.dimr_params = dimr_params
        self.keys = models.keys()
        self.grid_searches = {}
        self.scores = {}
        self.best_estimator_ = None
        self.best_estimators_ = {}
        self.storage = storage
        self.studies = {}

    def create_studies(self):
        for key in self.keys:
            if self.dimrs is None:
                print("\nCreating Optuna for %s." % key, flush=True)
                #run Optuna search with inner search CV on outer split data 
                pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
                
                # xgb study
                study_name = "parallel."+key
                #optuna.delete_study(study_name=study_name, storage=storage) # if there is existing study remove.
                study = optuna.create_study(study_name=study_name, direction="maximize", storage=storage, pruner=pruner, load_if_exists=True)
                self.studies[key] = study 

    def fit(self, X, y, cv=3, n_jobs=10, verbose=1, scoring=None, refit=False):
        for key in self.keys:
            if self.dimrs is None:
                print("\nRunning Optuna for %s." % key, flush=True)
                #run Optuna search with inner search CV on outer split data 
                # xgb study
                study_name = key
                study = optuna.load_study(study_name=study_name, storage=storage) 
                #study.optimize(Objective(X_split, y_split, key, cv, scoring), n_trials=2000, timeout=600, n_jobs=10)  # will run 4 process to cover 2000 approx trials 

                self.studies[key] = study 
            else:
                print("Running BayesSearchCV for %s." % key)
                for dimr_label in self.dimrs:
                    print("Testing %s dim reduction" % dimr_label)
                    model = self.models[key]
                    params = self.params[key]
                    dimr = self.dimrs[dimr_label]
                    dimr_params = self.dimr_params[dimr_label]
                    pipeline = Pipeline([(dimr_label, dimr), (key,model)])

                    gs_params = {}
                    for i in params:
                        gs_params[key+'__'+i] = params[i]
                    for i in dimr_params:
                        gs_params[dimr_label+'__'+i] = dimr_params[i]

                    gs = BayesSearchCV(pipeline, gs_params, cv=cv, n_jobs=n_jobs,
                        verbose=verbose, scoring=scoring, refit=refit,
                        return_train_score=True)
                    #print(np.argwhere(np.isnan(X)))
                    #print(np.argwhere(np.isinf(X)))
                    #print(np.argwhere(np.isnan(y)))
                    #print(np.argwhere(np.isinf(y)))
                    gs.fit(X,y)
                    self.grid_searches[dimr_label + '_' + key] = gs    

    def score_summary(self, sort_by='mean_score'):
        def row(key, scores, params):
            d = {
                 'estimator': key,
                 'min_score': min(scores),
                 'max_score': max(scores),
                 'mean_score': np.mean(scores),
                 'std_score': np.std(scores),
            }
            return pd.Series({**params,**d})

        rows = []
        for k in self.grid_searches:
            #print(k)
            params = self.grid_searches[k].cv_results_['params']
            scores = []
            if isinstance(self.grid_searches[k].cv, int):
                rng = range(self.grid_searches[k].cv)
            else:
                rng = range(self.grid_searches[k].cv.get_n_splits())
            for i in rng:
                key = "split{}_test_score".format(i)
                r = self.grid_searches[k].cv_results_[key]        
                scores.append(r.reshape(len(params),1))

            all_scores = np.hstack(scores)
            for p, s in zip(params,all_scores):
                rows.append((row(k, s, p)))

        df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)

        columns = ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]


        self.scores = df[columns]
        return df[columns]

    def best_estimator(self, score='mean_score', method='train', X=None, y=None):
        grid_searches = self.grid_searches
        if method=='train':
            scores = self.scores
            if len(scores)==0:
                print('Scores empty, run score_summary()')
                return False
            #id estimator with highest score
            clf0 = scores.sort_values([score]).estimator.to_list()[0]
            return grid_searches[clf0].best_estimator_
        if method=='test':
            test_results = pd.DataFrame(columns=['DimReduction','Classifier','test_bal_accuracy','test_f1','clf_idx'])
            clfs = []
            clfidx = 0
            #choose best estimator from each gridsearch
            #also store in self.best_estimators_
            for gs in grid_searches:
                clf0 = grid_searches[gs].best_estimator_
                #compute test accuracy
                test_acc = balanced_accuracy_score(y, clf0.predict(X))
                test_f1 = f1_score(y, clf0.predict(X))
                if '_' in gs:
                    dr, cl = gs.split('_')
                    classifier = clf0.steps[1]
                else:
                    dr = None
                    cl = gs
                    classifier = clf0.steps[0]
                if hasattr(classifier[1], 'feature_importances_'):
                    importances = classifier[1].feature_importances_ 
                else:
                    importances = None
                self.best_estimators_[gs] = {'clf':clf0, 'test_acc':test_acc, 'test_f1':test_f1, 'importances':importances}
                test_results = test_results.append(pd.DataFrame({'DimReduction':[dr],'Classifier':[cl],'test_bal_accuracy':[test_acc], 'test_f1':[test_f1], 'clf_idx':[clfidx]}), ignore_index=True)
                clfs.append(clf0)
                clfidx += 1
            #choose clf with highest test accuracy
            #clfidx = test_results.loc[test_results['test_bal_accuracy'] == test_results.test_bal_accuracy.max(), 'clf_idx'].values[0]
            clfidx = test_results.loc[test_results['test_f1'] == test_results.test_f1.max(), 'clf_idx'].values[0]
            self.best_estimator_ = clfs[clfidx]
            return(clfs[clfidx])
            
    def best_params(self, score='mean_score'):
        if self.best_estimator is None:
            scores = self.scores
            if len(scores)==0:
                print('Scores empty, run score_summary()')
                return False
            #id estimator with highest score
            clf0 = self.scores.sort_values([score]).estimator.to_list()[0]
            return self.grid_searches[clf0].best_params_
        else:
            best_pipeline = self.best_estimator_
            #print(best_pipeline)
            steps = [x[0] for x in best_pipeline.get_params()['steps']]
            best_params = best_pipeline.get_params()
            if len(steps) == 1:
              clf_name = steps[0]
              clf_param_names = [i for i in self.params[clf_name]]
              clf_param_vals = [best_params[clf_name+'__'+i] for i in clf_param_names]
              pcols = [clf_name+'__'+i for i in clf_param_names]
              pcols.extend(['DimReduction','Classifier'])
              #out = pd.DataFrame(columns=pcols)
              outdir = {'DimReduction':['None'], 'Classifier':[clf_name]}
              print(clf_param_vals)
              for i in range(len(clf_param_names)):
                  outdir[clf_name+ '__' +clf_param_names[i]] = [clf_param_vals[i]]
            else:
              clf_name = steps[1]
              dimr_name = steps[0]
              clf_param_names = [i for i in self.params[clf_name]]
              dr_param_names = [i for i in self.dimr_params[dimr_name]]
              clf_param_vals = [best_params[clf_name+'__'+i] for i in clf_param_names]
              dr_param_vals = [best_params[dimr_name+'__'+i] for i in dr_param_names]
              pcols = [clf_name+'__'+i for i in clf_param_names]
              pcols.extend([dimr_name+'__'+i for i in dr_param_names])
              pcols.extend(['DimReduction','Classifier'])
              #out = pd.DataFrame(columns=pcols)
              outdir = {'DimReduction':[dimr_name], 'Classifier':[clf_name]}
              print(clf_param_vals)
              for i in range(len(clf_param_names)):
                  outdir[clf_name+ '__' +clf_param_names[i]] = [clf_param_vals[i]]
              for i in range(len(dr_param_names)):
                  outdir[dimr_name+ '__' +dr_param_names[i]] = [dr_param_vals[i]]
            return pd.DataFrame.from_dict(outdir)
            #print(param_names)

#logfile
#out = open('../run.RF/logs/cross_validation.log','w')
def plot_pr_curves(temp_estimators, X, y, abc_score, distance, out_idx, outdir):
    colors = ['purple','red','blue','orange','cyan','green','pink','yellow','blueviolet']
    fig, ax = plt.subplots()

    for i in temp_estimators:
        pipe = temp_estimators[i]['clf']
        #reduced = pipe.named_steps.SelectKBest.transform(X)
        if 'RandomForest' in i:
            label = 'Random Forest'
            ypred = pipe.named_steps.RandomForestClassifier.predict_proba(X)
            precision, recall, thresholds = precision_recall_curve(y, ypred[:,1])
            ax.plot(recall, precision, color=colors[1], label=label)

        if 'xgb' in i:
            label = 'XGBoost'
            ypred = pipe.named_steps['xgb'].predict_proba(X)
            precision, recall, thresholds = precision_recall_curve(y, ypred[:,1])
            ax.plot(recall, precision, color=colors[5], label=label)

        if 'LogisticRegression' in i:
            label = 'Logistic Regression'
            ypred = pipe.named_steps.LogisticRegression.predict_proba(X)
            precision, recall, thresholds = precision_recall_curve(y, ypred[:,1])
            ax.plot(recall, precision, color=colors[4], label=label)

    #add abc and distance
    precision, recall, thresholds = precision_recall_curve(y, distance)
    ax.plot(recall, precision, color='black', label='Lin Distance')

    precision, recall, thresholds = precision_recall_curve(y, abc_score)
    ax.plot(recall, precision, color='brown', label='ABC')

    #add axis labels to plot
    ax.set_title('Precision-Recall Curve')
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    ax.legend(loc='upper right')

    plt.savefig(outdir+'pr_curve_'+str(out_idx)+'.png')


#might need to add an option to save the figure witha specific name
def plot_coefficients(classifier, feature_names, top_features=20):
    coef = classifier.coef_.ravel()
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # create plot
    plt.figure(figsize=(15, 5))
    colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(0.3, 0.3 +2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
    plt.tight_layout()
    plt.savefig('../run.RF/data/top_'+str(top_features)+'_features.png')


#X, y = load_digits(return_X_y=True)
#print(np.isnan(y).any())
#####################################
# define the classifiers and params #
#####################################

#in this script we use diff params for bayescv

models = {
    'xgb': xgb.XGBClassifier( 
      objective= 'binary:logistic',
      nthread=4,
    ),
    'rf':RandomForestClassifier(class_weight='balanced'),
    #'lr':LogisticRegression(solver='liblinear', class_weight="balanced"),
}

params = {
    'xgb':{
      'n_estimators' : Integer(20, 100, 'uniform'),
      'max_depth' : Integer(1, 5, 'uniform'),
      'min_child_weight' : Real(0.1, 10, 'log-uniform'),
      'colsample_bytree' : Real(0.8, 1, 'uniform'),
      'subsample' : Real(0.5, 1, 'uniform'),
      'gamma': (1e-9, 0.1, 'log-uniform')
    },   
    'rf':{'n_estimators': Integer(100,300),'min_samples_leaf':Integer(1,20),
      'max_depth': Integer(5, 12),
      'min_samples_split': Integer(2, 10)
    },
    #'lr':{'C':Real(1e-6,1e5, prior='log-uniform')},
}

dim_reductions = {
#    'NMF':NMF(max_iter=300),
    'SelectKBest':SelectKBest(chi2),
#    'PCA':PCA(iterated_power=100),
}

dimr_params = {
#    'SelectKBest':{'k':Integer(1,50)},
#    'PCA':{'n_components':Integer(1,50)},
#    'NMF':{'n_components':Integer(1,50)},
    'SelectKBest':['passthrough'],
}


class Objective:
  def __init__(self, X, y, classifier, cv, scoring):
    # Hold this implementation specific arguments as the fields of the class.
    self.X = X 
    self.y = y
    self.classifier = classifier
    self.cv = cv
    self.scoring = scoring

  def __call__(self, trial):
    # Calculate an objective value by using the extra arguments.
    counter = Counter(self.y)
    estimate = counter[0] / counter[1]
    cls_weight = (self.y.shape[0] - np.sum(self.y)) / np.sum(self.y)

    dtrain = xgb.DMatrix(self.X, label=self.y)

    param = {}
    if self.classifier == "xgb":
      param = { 
          "verbosity": 0,
          "random_state" : RANDOM_SEED,
          "objective": "binary:logistic",
          # use exact for small featuresset.
          "tree_method": "hist",
          # n_estimator
          "num_boost_round": trial.suggest_int("num_boost_round", 50, 400),
          # defines booster
          "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
          # maximum depth of the tree, signifies complexity of the tree.
          "max_depth": trial.suggest_int("max_depth", 2, 10),
          # minimum child weight, larger the term more conservative the tree.
          "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
          # learning rate
          "eta": trial.suggest_float("eta", 1e-8, 0.3, log=True),
          # sampling ratio for training features.
          "subsample": trial.suggest_float("subsample", 0.4, 0.8),
          # sampling according to each tree.
          "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1),
          # L2 regularization weight.
          "lambda": trial.suggest_float("lambda", 1e-9, 0.01, log=True),
          # L1 regularization weight.
          "alpha": trial.suggest_float("alpha", 1e-9, 0.2, log=True),
          # defines how selective algorithm is.
          "gamma": trial.suggest_float("gamma", 1e-9, 1.0, log=True),
          "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
          "scale_pos_weight": np.sqrt(cls_weight),
          "eval_metric" : 'map',        #map: mean average precision aucpr: auc for precision recall
          "max_delta_step" : 1,
      }
      if param["booster"] == "dart":
          param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
          param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
          param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
          param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)
    elif self.classifier == "rf":
      param = { 
          "verbosity": 0,
          "random_state" : RANDOM_SEED,
          "objective": "binary:logistic",
          # use exact for small featuresset.
          "tree_method": "hist",
          # num_parallel_tree
          "num_parallel_tree": trial.suggest_int("num_parallel_tree", 50, 300),
          # maximum depth of the tree, signifies complexity of the tree.
          "max_depth": trial.suggest_int("max_depth", 2, 10),
          # minimum child weight, larger the term more conservative the tree.
          "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
          # learning rate
          "eta": trial.suggest_float("eta", 1e-8, 0.3, log=True),
          # sampling ratio for training features.
          "subsample": trial.suggest_float("subsample", 0.4, 0.8),
          # sampling by node
          "colsample_bynode": trial.suggest_float("colsample_bynode", 0.5, 0.99),
          "scale_pos_weight": np.sqrt(cls_weight),
          "eval_metric" : 'map',
          "max_delta_step" : 1,
      }
      # booster is set to "gbtree"
      param['booster'] = "gbtree"
      # num_boost_round(n_estimator) is set to 1 to make it RF instead of boosting. 
      param['num_boost_round'] = 1

    # set up cross-validation
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "test-map")
    cv_results = xgb.cv(param, dtrain, nfold=3, stratified=True, callbacks=[pruning_callback])

    # Save cross-validation results.
    cv_results.to_csv(outdir+'/'+'cv.'+filenamesuffix+'.'+str(pid)+'.'+str(trial.number)+'.txt', index=False, sep='\t')

    mean_map = cv_results["test-map-mean"].values[-1]
    return mean_map



# python -i src/runshap.py --modeldir run.newtrain/run.newtrain10 --studyname newtrain10 --outdir run.applymodel/newtrain10/
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--modeldir', required=True, help="directory containing data and model files")
  parser.add_argument('--studyname', required=True, help="studyname prefix for data files") 
  parser.add_argument('--outdir', default='.', help="directory to save shap results")
  parser.add_argument('--chr', default='all', help="chromosome")

  args=parser.parse_args()
  pid = os.getpid()

  nfold = 4
  modeldir = args.modeldir
  studyname = args.studyname
  outdir = args.outdir
  chromosome = args.chr
  scalerfile = ''
  modelfile = ''
  features = ''
  targets = ''
  featurenames = ''

  filenamesuffix = ''

  #################################
  #import our data, then format it #
  ##################################

#  eroles = pd.read_csv('data/Gasperini/Gasperini2019.at_scale.erole.txt', sep="\t", index_col=0)
#  print(eroles)
  outer_index = 0
  modelFilenamesList = glob.glob(modeldir+'/'+studyname+'.save*.json')
  configFilenamesList = glob.glob(modeldir+'/'+studyname+'.config*.json')

  list_importances = []
  list_shap_values = []
  list_shap_interactions = []
  X_test = pd.read_csv(modeldir +'/'+studyname+'.Xtest.0.txt', sep='\t', index_col=0)
  X_test = X_test.drop(columns = ['ABC.id'])
  print(X_test)
  X = pd.DataFrame(columns = X_test.columns)
  print(X)
  ABCid = pd.DataFrame(columns = ['ABC.id'])
  y = pd.DataFrame(columns = ['Significant'])

  for outer_index in range(nfold):
      modelfile = modelFilenamesList[outer_index]
      print(modelfile)

      X_test = pd.read_csv(modeldir +'/'+studyname+'.Xtest.'+str(outer_index)+'.txt', sep='\t', index_col=0)
      X_test = X_test.drop(columns = ['ABC.id'])
      y_test = pd.read_csv(modeldir +'/'+studyname+'.ytest.'+str(outer_index)+'.txt', sep='\t', index_col=0)
      ABCid_test = pd.read_csv(modeldir +'/'+studyname+'.ABCidtest.'+str(outer_index)+'.txt', sep='\t', index_col=0)
      y = pd.concat([y, y_test])
      y_test = y_test['Significant']
      print(y_test)
      print(str(sum(y_test))+'/'+str(len(y_test)))

      dtestfilename = modeldir +'/'+'dtest.'+str(outer_index)+'.data'
      dtest = xgb.DMatrix(dtestfilename)

      # load model
      xgb_clf_tuned_2 = xgb.Booster()
      xgb_clf_tuned_2.load_model(modelfile)    
      best_iteration = xgb_clf_tuned_2.best_iteration
      lst_vars_in_model = xgb_clf_tuned_2.feature_names
      featurenames = pd.DataFrame({"features":lst_vars_in_model})
      print(featurenames)

      print(X_test)
      # load featurenames we need
      X_test = X_test.reindex(columns = featurenames['features'])
      print(X_test)
      X = pd.concat([X, X_test])
      ABCid = pd.concat([ABCid, ABCid_test])
      # load preprocessor 
      #scaler = joblib.load(scalerfile)
      #cols = X_test.columns
      #X_test = pd.DataFrame(scaler.transform(X_test), columns = cols)



      #y_pred_prob = xgb_clf_tuned_2.predict(dtest, ntree_limit=best_iteration)
      #print(y_pred_prob)
      #y_pred = [round(value) for value in y_pred_prob]
  
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
        #lambda=params_dict['lambda'], 
        learning_rate=float(0.01),
        max_delta_step=1,
        #max_depth=params_dict['max_depth'],
        max_depth=4,
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


  X.insert(0, "ABC.id", ABCid['ABC.id'])
  X.to_csv(outdir+'/X.'+studyname+'.txt', sep='\t') 
  X = X.drop(columns = ['ABC.id'])
  ABCid.to_csv(outdir+'/ABCid.'+studyname+'.txt', sep='\t') 

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
  shap_pandas.insert(0, "ABC.id", ABCid['ABC.id'])
  shap_pandas.to_csv(outdir+'/shap_values.'+studyname+'.txt', sep='\t')
  shap_pandas = shap_pandas.drop(columns = ['ABC.id'])

  # summary plot
  fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(100, 100))
  shap.summary_plot(shap_pandas.to_numpy(), X, max_display=100)
  plt.savefig(outdir+'/'+'shap.'+studyname+'.shap.pdf')
  fig.clf()
  plt.close(fig)


  # plot dependency for top features 
  shap_order = shap_pandas.abs().mean().sort_values(ascending=[False])
  for i in range(0,20):
    for j in range(0,20):
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

