import argparse
import time, os
import joblib
from importlib import reload
os.environ["OMP_NUM_THREADS"] = "2" # export OMP_NUM_THREADS=2
import numpy as np
from  scipy.stats import rankdata as rank
import matplotlib.pyplot as plt
from statistics import mean, stdev
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import preprocessing
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold, StratifiedGroupKFold, cross_val_score
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



def set_num_threads(num):
  os.environ["OMP_NUM_THREADS"] = "2" # export OMP_NUM_THREADS=2
  #os.environ["OPENBLAS_NUM_THREADS"] = "2" # export OPENBLAS_NUM_THREADS=2 
  #os.environ["MKL_NUM_THREADS"] = "2" # export MKL_NUM_THREADS=2
  #os.environ["VECLIB_MAXIMUM_THREADS"] = "2" # export VECLIB_MAXIMUM_THREADS=2
  #os.environ["NUMEXPR_NUM_THREADS"] = "2" # export NUMEXPR_NUM_THREADS=2
  


#helper class that allows you to iterate over multiple classifiers within the nested for loop
class EstimatorSelectionHelper:
    def __init__(self, models, storage=None):
        #if not set(models.keys()).issubset(set(params.keys())):
        #    missing_params = list(set(models.keys()) - set(params.keys()))
        #    raise ValueError("Some estimators are missing parameters: %s" % missing_params)
        self.models = models
        self.grid_searches = {}
        self.scores = {}
        self.best_estimator_ = None
        self.best_estimators_ = {}
        self.storage = storage
        self.studies = {}
        for model in self.models: 
            self.studies[model] = {}
        self.current_model = {}
        self.current_idx = {}


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

#models = ['xgb', 'rf']
models = ['xgb']
#in this script we use diff params for bayescv
#
#models = {
#    'xgb': xgb.XGBClassifier( 
#      objective= 'binary:logistic',
#      nthread=4,
#    ),
#    'rf':RandomForestClassifier(class_weight='balanced'),
#    #'lr':LogisticRegression(solver='liblinear', class_weight="balanced"),
#}
#
#params = {
#    'xgb':{
#      'n_estimators' : Integer(20, 100, 'uniform'),
#      'max_depth' : Integer(1, 5, 'uniform'),
#      'min_child_weight' : Real(0.1, 10, 'log-uniform'),
#      'colsample_bytree' : Real(0.8, 1, 'uniform'),
#      'subsample' : Real(0.5, 1, 'uniform'),
#      'gamma': (1e-30, 0.00001, 'log-uniform')
#    },   
#    'rf':{'n_estimators': Integer(100,300),'min_samples_leaf':Integer(1,20),
#      'max_depth': Integer(5, 12),
#      'min_samples_split': Integer(2, 10)
#    },
#    #'lr':{'C':Real(1e-6,1e5, prior='log-uniform')},
#}
##
#dim_reductions = {
##    'NMF':NMF(max_iter=300),
#    'SelectKBest':SelectKBest(chi2),
##    'PCA':PCA(iterated_power=100),
#}
#
#dimr_params = {
##    'SelectKBest':{'k':Integer(1,50)},
##    'PCA':{'n_components':Integer(1,50)},
##    'NMF':{'n_components':Integer(1,50)},
#    'SelectKBest':['passthrough'],
#}
#

def RandomGroupKFold_split(groups, n, seed=None):  # noqa: N802
    """
    Random analogous of sklearn.model_selection.GroupKFold.split.

    :return: list of (train, test) indices
    """
    groups = pd.Series(groups)
    ix = np.arange(len(groups))
    unique = np.unique(groups)
    np.random.RandomState(seed).shuffle(unique)
    result = []
    for split in np.array_split(unique, n):
        mask = groups.isin(split)
        train, test = ix[~mask], ix[mask]
        result.append((train, test))

    return result

class Objective:
  def __init__(self, X, y, classifier, custom_fold, study_name_prefix, scoring = 'map', cls_weight = 100):
    # Hold this implementation specific arguments as the fields of the class.
    self.X = X 
    self.y = y
    #self.dtrain = dtrain
    self.classifier = classifier
    self.custom_fold = custom_fold
    self.scoring = scoring
    self.cls_weight = cls_weight
    self.study_name_prefix = study_name_prefix

  def __call__(self, trial):
    # Calculate an objective value by using the extra arguments.

    #if (self.dtrain == None): 
    #    self.dtrain = xgb.DMatrix(self.X, label=self.y)

    param = {}
    if self.classifier == "xgb":
      param = { 
          "verbosity": 0,
          "random_state" : RANDOM_SEED,
          "objective": "binary:logistic",
          # use exact for small featuresset.
          "tree_method": "auto",
          # n_estimator
          "num_boost_round": trial.suggest_int("num_boost_round", 100, 3000),
          # defines booster
          "booster": trial.suggest_categorical("booster", ["gbtree"]),
          #"booster": trial.suggest_categorical("booster", ["dart"]),
          # maximum depth of the tree, signifies complexity of the tree.
          #"max_depth": trial.suggest_int("min_child_weight", 3, 4),
          "max_depth": 3,
          # minimum child weight, larger the term more conservative the tree.
          "min_child_weight": trial.suggest_int("min_child_weight", 10, 20),
          # learning rate
          #"eta": trial.suggest_float("eta", 1e-8, 0.3, log=True),
          "eta": 0.01,
          # sampling ratio for training features.
          "subsample": trial.suggest_float("subsample", 0.5, 0.8),
          # sampling according to each tree.
          "colsample_bytree": trial.suggest_float("colsample_bytree", 0.70, 0.90),
          # L2 regularization weight.
          #"lambda": trial.suggest_float("lambda", 1e-9, 0.01, log=True),
          # L1 regularization weight.
          #"alpha": trial.suggest_float("alpha", 1e-9, 0.2, log=True),
          # defines how selective algorithm is.
          "gamma": trial.suggest_float("gamma", 10, 20),
          #"grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
          "scale_pos_weight": np.sqrt(self.cls_weight),
          "eval_metric" : 'map',        #map: mean average precision aucpr: auc for precision recall
          "max_delta_step" : 1,
      }
      if param["booster"] == "dart":
          #param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
          #param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
          param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 0.5, log=True)
          param["skip_drop"] = trial.suggest_float("skip_drop", 0.5, 1, log=True)
    elif self.classifier == "rf":
      param = { 
          "verbosity": 0,
          "random_state" : RANDOM_SEED,
          "objective": "binary:logistic",
          # use exact for small featuresset.
          "tree_method": "auto",
          # num_parallel_tree
          "num_parallel_tree": trial.suggest_int("num_parallel_tree", 50, 300),
          # maximum depth of the tree, signifies complexity of the tree.
          "max_depth": trial.suggest_int("max_depth", 2, 10),
          # minimum child weight, larger the term more conservative the tree.
          "min_child_weight": trial.suggest_int("min_child_weight", 10, 20),
          # learning rate
          #"eta": trial.suggest_float("eta", 1e-8, 0.3, log=True),
          "eta": 0.05,
          # sampling ratio for training features.
          "subsample": trial.suggest_float("subsample", 0.4, 0.8),
          # sampling by node
          "colsample_bynode": trial.suggest_float("colsample_bynode", 0.5, 0.99),
          "scale_pos_weight": np.sqrt(self.cls_weight),
          "eval_metric" : 'map',
          "max_delta_step" : 1,
      }
      # booster is set to "gbtree"
      param['booster'] = "gbtree"
      # num_boost_round(n_estimator) is set to 1 to make it RF instead of boosting. 
      param['num_boost_round'] = 1

    # set up cross-validation
    idx = 0
    cv_scores = np.empty(len(self.custom_fold))
    for (train_idx, test_idx) in self.custom_fold:
        X_train, X_test = self.X.iloc[train_idx], self.X.iloc[test_idx]
        y_train, y_test = self.y[train_idx], self.y[test_idx]
        print(X_train)
        print(X_test)
        print(y_train)
        print(y_test)
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        evals_result = {}
        pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-map")
        if idx == 0:
            xgb_clf_cv = xgb.train(params=param, dtrain=dtrain, 
                              num_boost_round=param['num_boost_round'],
                              evals=[(dtrain, "train"),(dtest, "validation")],
                              early_stopping_rounds=300,
                              evals_result=evals_result,
                              callbacks=[pruning_callback]
                              )
        else:
            xgb_clf_cv = xgb.train(params=param, dtrain=dtrain, 
                              num_boost_round=param['num_boost_round'],
                              evals=[(dtrain, "train"),(dtest, "validation")],
                              early_stopping_rounds=300,
                              evals_result=evals_result,
                              )

        print('')
        print('Access params through a loop:')
        for p_name, p_vals in param.items():
            print('- {}'.format(p_name))
            print('      - {}'.format(p_vals))
        print('')
 
        print('')
        print('Access metrics through a loop:')
        for e_name, e_mtrs in evals_result.items():
            print('- {}'.format(e_name))
            for e_mtr_name, e_mtr_vals in e_mtrs.items():
                print('   - {}'.format(e_mtr_name))
                print('      - {}'.format(e_mtr_vals))
        print('')
        cv_scores[idx] = evals_result['validation']['map'][-1]
        # Save cross-validation results.
        pd.DataFrame.from_dict(evals_result).to_csv(outdir+'/'+self.study_name_prefix+'.cv.'+filenamesuffix+'.'+str(pid)+'.'+str(idx)+'.txt', index=False, sep='\t')

        idx=idx+1

    #pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "test-map")
    #cv_results = xgb.cv(param, self.dtrain, folds=self.custom_fold, early_stopping_rounds=100, callbacks=[pruning_callback])
    
    #trial.set_user_attr("n_estimators", len())
    print(param['scale_pos_weight'])
    print(np.sqrt(self.cls_weight))
    trial.set_user_attr("scale_pos_weight", np.sqrt(self.cls_weight))

    best_iteration = xgb_clf_cv.best_iteration
    print('best_iteration: ' + str(best_iteration))
    trial.set_user_attr("best_iteration", best_iteration)

    #mean_map = cv_results["test-map-mean"].values[-1]
    print(cv_scores)
    mean_map = np.mean(cv_scores)
    print(mean_map)
    return mean_map


# class outer folds
class OuterFolds:
    def __init__(self, foldsplit, nfold, groups, storage, study_name_prefix, scoring):
      # Hold this implementation specific arguments as the fields of the class.
      self.study_name_prefix=study_name_prefix
      self.outer_split = foldsplit
      self.nfold = nfold
      self.groups = groups
      self.storage = storage
      self.scoring = scoring
      self.outer_results = pd.DataFrame()
      #self.X_splits = {}
      #self.y_splits = {}
      #self.group_splits = {}
      #self.dtrains = {}
      #self.dtrainfilenames = {}
      #self.X_tests = {}
      #self.y_tests = {}
      #self.group_tests = {}
      #self.dtests = {}
      #self.dtestfilenames = {}

    # Set up outer folds for testing 
    def create_outer_fold(self, X, y):
        #######################
        # nested cv structure #
        #######################

        outer_index = 0
        #for split in self.outer_split.split(X,y):
        for split in self.outer_split.split(X,y,self.groups):
            #get indices for outersplit
            train_idx, test_idx = split

            #outer split data
            X_split = X.iloc[train_idx, :].copy()
            y_split = y.iloc[train_idx].copy()
            group_train = self.groups.iloc[train_idx]
            X_split.to_csv(outdir +'/'+self.study_name_prefix+'.Xsplit.'+str(outer_index)+'.txt', sep='\t')
            y_split.to_csv(outdir +'/'+self.study_name_prefix+'.ysplit.'+str(outer_index)+'.txt', sep='\t')
            group_train.to_csv(outdir +'/'+self.study_name_prefix+'.groupsplit.'+str(outer_index)+'.txt', sep='\t')
            #self.X_splits[outer_index] = X_split
            #self.y_splits[outer_index] = y_split
            #self.group_splits[outer_index] = group_train

            cols = X_split.columns
            scaler = preprocessing.MinMaxScaler()
            scaler.fit(X_split)
            X_split = pd.DataFrame(scaler.transform(X_split), columns = cols)
            dtrain = xgb.DMatrix(X_split, label=y_split)
            dtrainfilename = outdir +'/'+'dtrain.'+str(outer_index)+'.data'
            dtrain.save_binary(dtrainfilename)
            #self.dtrains[outer_index] = dtrain
            #self.dtrainfilenames[outer_index] = dtrainfilename
            joblib.dump(scaler, outdir +'/'+'scaler.'+str(outer_index)+'.gz')

            X_test = X.iloc[test_idx,:].copy()
            y_test = y.iloc[test_idx].copy()
            group_test = self.groups.iloc[test_idx]
            X_test.to_csv(outdir +'/'+self.study_name_prefix+'.Xtest.'+str(outer_index)+'.txt', sep='\t')
            y_test.to_csv(outdir +'/'+self.study_name_prefix+'.ytest.'+str(outer_index)+'.txt', sep='\t')
            group_test.to_csv(outdir +'/'+self.study_name_prefix+'.grouptest.'+str(outer_index)+'.txt', sep='\t')
            #self.X_tests[outer_index] = X_test
            #self.y_tests[outer_index] = y_test
            #self.group_tests[outer_index] = group_test
            X_test = pd.DataFrame(scaler.transform(X_test), columns = cols)
            dtest = xgb.DMatrix(X_test) 
            dtestfilename = outdir +'/'+'dtest.'+str(outer_index)+'.data'
            dtest.save_binary(dtestfilename)
            #self.dtests[outer_index] = dtest
            #self.dtestfilenames[outer_index] = dtestfilename

            storage = optuna.storages.RDBStorage(url="postgresql://mhan@localhost:"+str(postgres_port)+"/example")
            for model in models: 
                print("\nCreating Optuna for %s outer fold %d." % (model, outer_index), flush=True)
                pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
                # xgb study
                study_name = self.study_name_prefix+model+"."+str(outer_index)
                #optuna.delete_study(study_name=study_name, storage=storage) # if there is existing study remove.
                study = optuna.create_study(study_name=study_name, direction="maximize", storage=storage, pruner=pruner, load_if_exists=True)
            outer_index += 1


    def optimize_hyperparams(self, model, outer_index, n_inner_fold=4, scoring='map'):
        #outer_index = 0
        #for classifier, folds in self.helper.studies.items():
            #print(classifier)
            #print(folds)
            #for outer_index,study_name in folds.items():
            #    print(outer_index)
            #    print(study_name) 
        study_name = self.study_name_prefix+model+"."+str(outer_index)
        study = optuna.load_study(study_name=study_name, storage=self.storage) 
        print("Loaded study  %s with  %d trials." % (study_name, len(study.trials)))
        X_split = pd.read_csv(outdir +'/'+self.study_name_prefix+'.Xsplit.'+str(outer_index)+'.txt', sep='\t', index_col=0).reset_index(drop=True)
        print(X_split)
        y_split = pd.read_csv(outdir +'/'+self.study_name_prefix+'.ysplit.'+str(outer_index)+'.txt', sep='\t', index_col=0).reset_index(drop=True)
        y_split = y_split['Significant']
        print(y_split)
        counter = Counter(y_split)
        estimate = counter[0] / counter[1]
        cls_weight = (y_split.shape[0] - np.sum(y_split)) / np.sum(y_split)
        dtrainfilename = outdir +'/'+'dtrain.'+str(outer_index)+'.data'
        dtrain = xgb.DMatrix(dtrainfilename)
        group_split = pd.read_csv(outdir +'/'+self.study_name_prefix+'.groupsplit.'+str(outer_index)+'.txt', sep='\t', index_col=0).reset_index(drop=True)
        group_split = group_split['group']
        
        #run Optuna search with inner search CV on outer split data 
        inner_splits = StratifiedGroupKFold(n_splits=n_inner_fold, shuffle=True, random_state=RANDOM_SEED)
        custom_fold = []  #list of (train, test) indices
        for split in inner_splits.split(X_split,y_split,group_split):
            train_idx, test_idx = split
            custom_fold.append((train_idx, test_idx))
        study.optimize(Objective(X_split, y_split, model, custom_fold, self.study_name_prefix+str(outer_index), scoring, cls_weight), n_trials=2000, timeout=600, n_jobs=1)  # will run 4 process to cover 2000 approx trials 
       
    # test and summarize outer fold results based on best hyperparms
    def test_results(self):
        outer_index = 0
        for classifier in models:
            for outer_index in range(self.nfold):
                print(classifier)
                print(outer_index)
                study_name = self.study_name_prefix+classifier+"."+str(outer_index)
                print(study_name) 
                study = optuna.load_study(study_name=study_name, storage=self.storage) 
                print("Number of finished trials for  %s: %d." % (study_name, len(study.trials)))
                if (len(study.trials) == 0):
                    break;
                print("Best trial:")
                trial = study.best_trial

                print("  Value: {}".format(trial.value))
                print("  best_iteration: {}".format(trial.user_attrs['best_iteration']))
                print("  Params: ")
                for key, value in trial.params.items():
                    print("    {}: {}".format(key, value))
                #print("  Number of estimators: {}".format(trial.user_attrs["n_estimators"]))
                print("  scale_pos_weight: {}".format(trial.user_attrs["scale_pos_weight"]))

                params = trial.params
                #print("params[scale_pos_weight]")
                #print(params['scale_pos_weight'])

                #y_split = pd.read_csv(outdir +'/'+'ysplit.'+str(outer_index)+'.txt', sep='\t', index_col=0)
                #y_split = y_split['Significant']
                #counter = Counter(y_split)
                #estimate = counter[0] / counter[1]
                #cls_weight = (y_split.shape[0] - np.sum(y_split)) / np.sum(y_split)
                #params['scale_pos_weight'] = np.sqrt(cls_weight)
                params['scale_pos_weight'] = trial.user_attrs["scale_pos_weight"]
                params['objective'] = "binary:logistic" 
                if (classifier == 'rf'):
                    params['num_boost_round'] = 1
                dtrainfilename = outdir +'/'+'dtrain.'+str(outer_index)+'.data'
                dtrain = xgb.DMatrix(dtrainfilename)
                xgb_clf_tuned_2 = xgb.train(params=params, dtrain=dtrain, num_boost_round=trial.user_attrs['best_iteration'])
                #xgb_clf_tuned_2 = xgb.train(params=params, dtrain=dtrain, num_boost_round=params['num_boost_round'])
                lst_vars_in_model = xgb_clf_tuned_2.feature_names
                print(lst_vars_in_model)
                featurenames = pd.DataFrame({"features":lst_vars_in_model})
                featurenames.to_csv(outdir+'/'+self.study_name_prefix+'.featurenames'+str(pid)+'.'+classifier+'.'+str(outer_index)+'.txt', index=False, sep='\t')
                best_iteration = xgb_clf_tuned_2.best_iteration
                print("new best_iteration: "+str(best_iteration))
                xgb_clf_tuned_2.save_model('data/trained_models/mira_data/save'+str(pid)+'.'+classifier+'.'+str(outer_index)+'.json')
                xgb_clf_tuned_2.dump_model('data/trained_models/mira_data/dump'+str(pid)+'.'+classifier+'.'+str(outer_index)+'.json')
                
                X_test = pd.read_csv(outdir +'/'+self.study_name_prefix+'.Xtest.'+str(outer_index)+'.txt', sep='\t', index_col=0)
                y_test = pd.read_csv(outdir +'/'+self.study_name_prefix+'.ytest.'+str(outer_index)+'.txt', sep='\t', index_col=0)
                y_test = y_test['Significant']
                print(y_test)
                dtestfilename = outdir +'/'+'dtest.'+str(outer_index)+'.data'
                dtest = xgb.DMatrix(dtestfilename)
                y_pred_prob = xgb_clf_tuned_2.predict(dtest, ntree_limit=best_iteration)
                print(y_pred_prob)
                y_pred = [round(value) for value in y_pred_prob]
                test_pd = pd.DataFrame(y_pred, columns=['pred'], index=y_test.index)
                y_res = pd.merge(test_pd, y_test, left_index=True, right_index=True)
                #res = pd.merge(y_res, pd.DataFrame(X_test, columns=X_test_columns, index=X_test_index), left_index=True, right_index=True)
                res = pd.merge(y_res, X_test, left_index=True, right_index=True)
                res.to_csv(outdir+'/'+self.study_name_prefix+'.confusion.'+filenamesuffix+'.'+str(pid)+'.'+classifier+'.'+str(outer_index)+'.txt', index=False, sep='\t')
                # Data to plot precision - recall curve
                precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob, pos_label = 1)
                print(precision)
                print(recall)
                aucpr = auc(recall, precision)
                average_precision = average_precision_score(y_test, y_pred_prob)
                bal_accuracy = balanced_accuracy_score(y_test, y_pred)
                test_f1_score = f1_score(y_test, y_pred)
                params['aucpr'] = [aucpr]
                params['average_precision'] = [average_precision]
                params['bal_accuracy'] = [bal_accuracy]
                params['f1_score'] = [test_f1_score]
                params['classifier'] = classifier
                params['best_iteration'] = best_iteration
                self.outer_results = pd.concat([self.outer_results,pd.DataFrame.from_dict(params)])
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
        print(self.outer_results) 
        self.outer_results.to_csv('data/trained_models/mira_data/'+str(pid)+'.outer_results.txt', sep='\t')


    # next func





# python src/mira_cross_val_bayescv.eroles.xgb.optuna.py --dir /data8/han_lab/mhan/abcd/data/ --outdir /data8/han_lab/mhan/abcd/run.groupcv --port 44803
# python -i src/mira_cross_val_bayescv.eroles.xgb.optuna.py --dir /data8/han_lab/mhan/abcd/data/ --outdir /data8/han_lab/mhan/abcd/run.groupcv --port 44803 --model 'rf' --outerfold 2
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--dir', required=True, help="directory containing edgelist and vertices files")
  parser.add_argument('--outdir', default='.', help="directory containing edgelist and vertices files")
  parser.add_argument('--chr', default='all', help="chromosome")
  parser.add_argument("--port", required=True, help="postgres port for storage")
  parser.add_argument("--studyname", required=True, help="study name prefix")
  parser.add_argument("--opt", action='store_true', help="parallel optimize") # if on, add parallel optimizers only
  parser.add_argument("--model", default='all', help="choose one of xgb, rf, lr only when optimizing")
  parser.add_argument("--outerfold", default='all', help="choose one of each outer fold only when optimizing")
  parser.add_argument("--test", action='store_true', help="gather test results based on tuned model") # if on, gather test results only

  args=parser.parse_args()
  pid = os.getpid()

  base_directory = args.dir
  chromosome = args.chr
  outdir = args.outdir
  postgres_port = args.port
  study_name_prefix = args.studyname+'.'
  optimize_only = args.opt
  classifier = args.model
  outerfold = args.outerfold
  test_only = args.test

  filenamesuffix = ''

  #################################
  #import our data, then format it #
  ##################################

  data2 = pd.read_csv('/data8/han_lab/mhan/abcd/data/Gasperini/Gasperini2019.at_scale.ABC.TF.erole.grouped.train.txt',sep='\t', header=0)
  data2 = data2.loc[:,~data2.columns.str.match("Unnamed")]
  #data2 = pd.read_csv('/data8/han_lab/mhan/abcd/data/Gasperini/Gasperini2019.at_scale.ABC.TF.eindirect.txt',sep='\t', header=0)
  #data2 = data2.loc[data2['e1']==1,]
  #data2 = data2.loc[(data2['e2']==1) | (data2['e3']==1),]
  #data2['distance'] = data2.apply(lambda row: np.absolute(row.start_x - row.TargetGeneTSS), axis=1)

  features_gasperini = data2
  ActivityFeatures = features_gasperini[['normalized_h3K27ac', 'normalized_dhs', 'activity_base_x', 'TargetGeneExpression', 'TargetGenePromoterActivityQuantile', 'TargetGeneIsExpressed', 'distance']].copy()
  ActivityFeatures = ActivityFeatures.dropna()
  #hicfeatures = features_gasperini[['hic_contact', 'ABC.Score.Numerator', 'ABC.Score']].copy()
  hicfeatures = features_gasperini[['hic_contact','ABC.Score']].copy()
  hicfeatures = hicfeatures.dropna()
  hicfeatures['hic_contact'] = np.log1p(hicfeatures['hic_contact'])
  TFfeatures = features_gasperini.filter(regex='(_e)|(_TSS)').copy()
  TFfeatures = TFfeatures.dropna()
  #cobindingfeatures = features_gasperini.filter(regex='(_co)').copy()
  #cobindingfeatures = cobindingfeatures.dropna()
  crisprfeatures = features_gasperini[['EffectSize', 'Significant', 'pValue' ]].copy()
  crisprfeatures = crisprfeatures.dropna()
  groupfeatures = features_gasperini[['group']].copy()

  features = ActivityFeatures.copy()
  features = pd.merge(features, hicfeatures, left_index=True, right_index=True)
  features = pd.merge(features, TFfeatures, left_index=True, right_index=True)
  features = pd.merge(features, crisprfeatures, left_index=True, right_index=True)
  data = pd.merge(features, groupfeatures, left_index=True, right_index=True)
  ActivityFeatures = data.iloc[:, :ActivityFeatures.shape[1]]
  hicfeatures = data.iloc[:, ActivityFeatures.shape[1]:ActivityFeatures.shape[1]+hicfeatures.shape[1]]
  TFfeatures = data.iloc[:, ActivityFeatures.shape[1]+hicfeatures.shape[1]:ActivityFeatures.shape[1]+hicfeatures.shape[1]+TFfeatures.shape[1]]
  #crisprfeatures = data.iloc[:, -3:]
  crisprfeatures = data[['EffectSize', 'Significant', 'pValue' ]]
  groupfeatures = data[['group']]
  f1 = set(list(features.columns))
  #features = data.iloc[:, :data.shape[1]-3]
  features = data.iloc[:, :data.shape[1]-crisprfeatures.shape[1]-groupfeatures.shape[1]]
  f2 = set(list(features.columns))
  target = crisprfeatures['Significant'].astype(int)
  groups = groupfeatures

  abc_score = features['ABC.Score'].values
  distance = features['distance'].values

  features.drop(columns=['ABC.Score'], axis=1, inplace=True)
  feature_labels = list(features.columns)

  X = features
  y = target

  # create outer fold object
  nfold = 5
  #outer_split = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=RANDOM_SEED)
  outer_split = StratifiedGroupKFold(n_splits=nfold, shuffle=True, random_state=RANDOM_SEED)
  storage = optuna.storages.RDBStorage(url="postgresql://mhan@localhost:"+str(postgres_port)+"/example")
  outerFolds = OuterFolds(outer_split, nfold, groups, storage, study_name_prefix, '')
  if (optimize_only == True): 
      outerFolds.optimize_hyperparams(classifier, outerfold)
  elif (test_only == True):
      outerFolds.test_results()
  else:
      outerFolds.create_outer_fold(X, y)
  storage.remove_session()


  print('Total runtime: ' + str(time.time() - tstart))    

