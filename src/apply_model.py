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
    cv_results.to_csv(outdir+'/'+'cv.'+filenamesuffix+'.'+str(pid)+'.'+str(trial.number)+'.txt', index=False)

    mean_map = cv_results["test-map-mean"].values[-1]
    return mean_map



# python -i src/apply_model.py  --dir /data8/han_lab/mhan/abcd/data/ --outdir /data8/han_lab/mhan/abcd/run --modelfile /data8/han_lab/mhan/abcd/data/trained_models/mira_data/save35212.xgb.0.json --features /data8/han_lab/mhan/abcd/run2/35212.Xtest.0.txt --targets /data8/han_lab/mhan/abcd/run2/35212.ytest.0.txt --featurenames /data8/han_lab/mhan/abcd/run2/featurenames35212.xgb.0.txt 
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--dir', required=True, help="directory containing edgelist and vertices files")
  parser.add_argument('--outdir', default='.', help="directory containing edgelist and vertices files")
  parser.add_argument('--chr', default='all', help="chromosome")
  parser.add_argument('--modelfile', required=True, help="model saved after training in json file")
  parser.add_argument('--features', required=True, help="feature matrix ")
  parser.add_argument('--targets', help="target truth to compare for evaluation")
  parser.add_argument('--featurenames', required=True, help="feature names saved after training in csv file")

  args=parser.parse_args()
  pid = os.getpid()

  base_directory = args.dir
  chromosome = args.chr
  outdir = args.outdir
  modelfile = args.modelfile
  features = args.features
  targets = args.targets
  featurenames = args.featurenames

  filenamesuffix = ''

  #################################
  #import our data, then format it #
  ##################################

  X_test = pd.read_csv(features)
  X_test = X_test.loc[:,~X_test.columns.str.match("Unnamed")]
  y_test = pd.read_csv(targets)
  y_test = y_test.loc[:,~y_test.columns.str.match("Unnamed")]

  # load model
  xgb_clf_tuned_2 = xgb.Booster()
  xgb_clf_tuned_2.load_model(modelfile)    
  featurenames_in_training = pd.read_csv(featurenames,sep='\t')
  best_ntree_limit = xgb_clf_tuned_2.best_ntree_limit

  #for be in temp_estimators:
  #    acc = temp_estimators[be]['test_f1']
  #    if be not in best_estimators:
  #        best_estimators[be] = temp_estimators[be]
  #    elif best_estimators[be]['test_f1'] < acc:
  #        best_estimators[be] = temp_estimators[be]
  #    importances = temp_estimators[be]['importances']
  #    if importances is not None:
  #        pd.DataFrame(data=importances, index=feature_labels).to_csv('data/trained_models/mira_data/'+str(pid)+'.importance.'+be+'.'+str(evaluation.shape[0])+'.txt')
           
  #return the best performing model on test data
  
  dtest = xgb.DMatrix(X_test) 
  y_pred_prob = xgb_clf_tuned_2.predict(dtest)
  y_pred = [round(value) for value in y_pred_prob]
  # Data to plot precision - recall curve
  precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob, pos_label = 1)
  print(precision)
  print(recall)
  aucpr = auc(recall, precision)
  average_precision = average_precision_score(y_test, y_pred_prob)
  bal_accuracy = balanced_accuracy_score(y_test, y_pred)
  test_f1_score = f1_score(y_test, y_pred)
  metric = {}
  metric['aucpr'] = [aucpr]
  metric['average_precision'] = [average_precision]
  metric['bal_accuracy'] = [bal_accuracy]
  metric['f1_score'] = [test_f1_score]
  evaluation = pd.DataFrame.from_dict(metric)
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


  pd.set_option('display.max_columns', None) 
  print(evaluation) 
#  evaluation.to_csv('data/trained_models/mira_data/'+str(pid)+'.evaluation.txt')
  #save best estimators 
#  for est in best_estimators:
#      joblib.dump(best_estimators[est]['clf'], 'data/trained_models/mira_data/'+str(pid)+'.'+est+'.pkl')
  print('Total runtime: ' + str(time.time() - tstart))    

