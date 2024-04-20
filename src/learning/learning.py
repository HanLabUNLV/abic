import argparse
import time, os
import joblib
from pathlib import Path
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.metrics import precision_recall_curve, auc, average_precision_score, balanced_accuracy_score, f1_score
import pandas as pd
import xgboost as xgb
import optuna
from optuna import create_study, logging
from optuna.pruners import MedianPruner
from optuna.integration import XGBoostPruningCallback
from collections import Counter
from BorutaShap import BorutaShap
import seaborn as sns
import matplotlib.pyplot as plt

RANDOM_SEED = 42

tstart = time.time()
pid = os.getpid()



def set_num_threads(num):
  os.environ["OMP_NUM_THREADS"] = "2" # export OMP_NUM_THREADS=2
  #os.environ["OPENBLAS_NUM_THREADS"] = "2" # export OPENBLAS_NUM_THREADS=2 
  #os.environ["MKL_NUM_THREADS"] = "2" # export MKL_NUM_THREADS=2
  #os.environ["VECLIB_MAXIMUM_THREADS"] = "2" # export VECLIB_MAXIMUM_THREADS=2
  #os.environ["NUMEXPR_NUM_THREADS"] = "2" # export NUMEXPR_NUM_THREADS=2
  








#models = ['xgb', 'rf']
models = ['xgb']





class Objective:
  def __init__(self, X, y, model, params, custom_fold, study_name_prefix, scoring = 'map', cls_weight = ''):
    # Hold this implementation specific arguments as the fields of the class.
    self.X = X 
    self.y = y
    #self.dtrain = dtrain
    self.model = model
    self.params = params
    self.custom_fold = custom_fold
    self.scoring = scoring
    self.cls_weight = cls_weight
    self.study_name_prefix = study_name_prefix


  def __call__(self, trial):
    # Calculate an objective value by using the extra arguments.

    param = {}
    param_all = {
      "verbosity": 0,
      "random_state" : RANDOM_SEED,
      "objective": "binary:logistic",
      # use exact for small featuresset.
      "tree_method": "auto",
      # n_estimator
      "num_boost_round": trial.suggest_int("num_boost_round", 50, 300),
      # defines booster
      "booster": trial.suggest_categorical("booster", ["gbtree"]),
      #"booster": trial.suggest_categorical("booster", ["dart"]),
      # maximum depth of the tree, signifies complexity of the tree.
      #"max_depth": trial.suggest_int("max_depth", 3, 4),
      "max_depth": 3,
      # minimum child weight, larger the term more conservative the tree.
      "min_child_weight": trial.suggest_int("min_child_weight", 14, 20),
      # learning rate
      #"eta": trial.suggest_float("eta", 1e-8, 0.3, log=True),
      "eta": 0.03,
      # sampling ratio for training features.
      #"subsample": 0.6,
      "subsample": trial.suggest_float("subsample", 0.3, 0.7),
      # sampling according to each tree.
      #"colsample_bytree": 0.7,
      "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.8),
      # L2 regularization weight.
      "lambda": trial.suggest_float("lambda", 2, 3, log=True),
      # L1 regularization weight.
      "alpha": trial.suggest_float("alpha", 1e-4, 0.1, log=True),
      # defines how selective algorithm is.
      "gamma": trial.suggest_float("gamma", 12, 17),
      #"grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
      "scale_pos_weight": self.cls_weight,
      "eval_metric" : 'map',        #map: mean average precision aucpr: auc for precision recall
      "max_delta_step" : 1,
    }

    param_atleast1 = {
      "verbosity": 0,
      "random_state" : RANDOM_SEED,
      "objective": "binary:logistic",
      # use exact for small featuresset.
      "tree_method": "auto",
      # n_estimator
      "num_boost_round": trial.suggest_int("num_boost_round", 50, 300),
      # defines booster
      "booster": trial.suggest_categorical("booster", ["gbtree"]),
      #"booster": trial.suggest_categorical("booster", ["dart"]),
      # maximum depth of the tree, signifies complexity of the tree.
      "max_depth": 3,
      #"max_depth": trial.suggest_int("max_depth", 3, 4),
      # minimum child weight, larger the term more conservative the tree.
      #"min_child_weight": 6,
      "min_child_weight": trial.suggest_int("min_child_weight", 4, 8),
      # learning rate
      #"eta": trial.suggest_float("eta", 1e-8, 0.3, log=True),
      "eta": 0.05,
      # sampling ratio for training features.
      #"subsample": 0.5,
      "subsample": trial.suggest_float("subsample", 0.4, 0.6),
      # sampling according to each tree.
      #"colsample_bytree": 0.5,
      "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.6),
      # L2 regularization weight.
      #"lambda": 2,
      "lambda": trial.suggest_float("lambda", 2, 3, log=True),
      # L1 regularization weight.
      "alpha": trial.suggest_float("alpha", 1e-9, 0.2, log=True),
      # defines how selective algorithm is.
      "gamma": trial.suggest_float("gamma", 13, 20),
      #"grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
      "scale_pos_weight": self.cls_weight,
      "eval_metric" : 'map',        #map: mean average precision aucpr: auc for precision recall
      "max_delta_step" : 1,
    }

    param_gene = {
    }

    param_e1 = {
      "verbosity": 0,
      "random_state" : RANDOM_SEED,
      "objective": "binary:logistic",
      # use exact for small featuresset.
      "tree_method": "auto",
      # n_estimator
      "num_boost_round": trial.suggest_int("num_boost_round", 50, 300),
      # defines booster
      "booster": trial.suggest_categorical("booster", ["gbtree"]),
      #"booster": trial.suggest_categorical("booster", ["dart"]),
      # maximum depth of the tree, signifies complexity of the tree.
      "max_depth": 3,
      #"max_depth": trial.suggest_int("max_depth", 3, 4),
      # minimum child weight, larger the term more conservative the tree.
      #"min_child_weight": 6,
      "min_child_weight": trial.suggest_int("min_child_weight", 10, 25),
      # learning rate
      #"eta": trial.suggest_float("eta", 1e-8, 0.3, log=True),
      "eta": 0.03,
      # sampling ratio for training features.
      #"subsample": 0.5,
      "subsample": trial.suggest_float("subsample", 0.4, 0.8),
      # sampling according to each tree.
      #"colsample_bytree": 0.5,
      "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.8),
      # L2 regularization weight.
      #"lambda": 2,
      "lambda": trial.suggest_float("lambda", 1, 3, log=True),
      # L1 regularization weight.
      "alpha": trial.suggest_float("alpha", 1e-9, 0.2, log=True),
      # defines how selective algorithm is.
      "gamma": trial.suggest_float("gamma", 10, 20),
      #"grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
      "scale_pos_weight": self.cls_weight,
      "eval_metric" : 'map',        #map: mean average precision aucpr: auc for precision recall
      "max_delta_step" : 1,
    }


    param_e2 = {
      "verbosity": 0,
      "random_state" : RANDOM_SEED,
      "objective": "binary:logistic",
      # use exact for small featuresset.
      "tree_method": "auto",
      # n_estimator
      "num_boost_round": trial.suggest_int("num_boost_round", 10, 500),
      # defines booster
      "booster": trial.suggest_categorical("booster", ["gbtree"]),
      #"booster": trial.suggest_categorical("booster", ["dart"]),
      # maximum depth of the tree, signifies complexity of the tree.
      "max_depth": 3,
      #"max_depth": trial.suggest_int("max_depth", 2,3),
      # minimum child weight, larger the term more conservative the tree.
      #"min_child_weight": 6,
      "min_child_weight": trial.suggest_int("min_child_weight", 10, 20),
      # learning rate
      #"eta": trial.suggest_float("eta", 1e-8, 0.3, log=True),
      "eta": 0.03,
      # sampling ratio for training features.
      #"subsample": 0.5,
      "subsample": trial.suggest_float("subsample", 0.6, 0.9),
      # sampling according to each tree.
      #"colsample_bytree": 0.5,
      "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 0.9),
      # L2 regularization weight.
      #"lambda": 2,
      "lambda": trial.suggest_float("lambda", 1, 5, log=True),
      # L1 regularization weight.
      "alpha": trial.suggest_float("alpha", 1e-9, 0.5, log=True),
      # defines how selective algorithm is.
      "gamma": trial.suggest_float("gamma", 8, 20),
      #"grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
      "scale_pos_weight": self.cls_weight,
      "eval_metric" : 'map',        #map: mean average precision aucpr: auc for precision recall
      "max_delta_step" : 1,
    }


    param_reduced = {
      "verbosity": 0,
      "random_state" : RANDOM_SEED,
      "objective": "binary:logistic",
      # use exact for small featuresset.
      "tree_method": "auto",
      # n_estimator
      "num_boost_round": trial.suggest_int("num_boost_round", 10, 150),
      # defines booster
      "booster": trial.suggest_categorical("booster", ["gbtree"]),
      #"booster": trial.suggest_categorical("booster", ["dart"]),
      # maximum depth of the tree, signifies complexity of the tree.
      #"max_depth": trial.suggest_int("max_depth", 3, 4),
      "max_depth": 3,
      # minimum child weight, larger the term more conservative the tree.
      "min_child_weight": trial.suggest_int("min_child_weight", 10, 20),
      # learning rate
      #"eta": trial.suggest_float("eta", 1e-8, 0.3, log=True),
      "eta": 0.03,
      # sampling ratio for training features.
      #"subsample": 0.6,
      "subsample": trial.suggest_float("subsample", 0.4, 0.8),
      # sampling according to each tree.
      #"colsample_bytree": 0.7,
      "colsample_bytree": trial.suggest_float("colsample_bytree", 0.45, 0.8),
      # L2 regularization weight.
      "lambda": trial.suggest_float("lambda", 2, 3, log=True),
      # L1 regularization weight.
      "alpha": trial.suggest_float("alpha", 1e-4, 0.1, log=True),
      # defines how selective algorithm is.
      "gamma": trial.suggest_float("gamma", 10, 17),
      #"grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
      "scale_pos_weight": self.cls_weight,
      "eval_metric" : 'map',        #map: mean average precision aucpr: auc for precision recall
      "max_delta_step" : 1,
    }


    print("self.params:{}".format(self.params))

    if self.params == "xgb.all":
      param = param_all 
    elif self.params == "xgb.atleast1":
      param = param_atleast1
    elif self.params == "xgb.gene":
      param = param_gene
    elif self.params == "xgb.e1":
      param = param_e1
    elif self.params == "xgb.e2":
      param = param_e2
    elif self.params == "xgb.reduced":
      param = param_reduced
    else:
      print("model params not defined")
      quit() 
      
    if param["booster"] == "dart":
      #param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
      #param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
      param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 0.5, log=True)
      param["skip_drop"] = trial.suggest_float("skip_drop", 0.5, 1, log=True)

    print("suggested num_boost_round:{}".format(param['num_boost_round']))

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

        # xgb train with evals
        evals_result = {}
        pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-map")
        print("used num_boost_round:{}".format(param['num_boost_round']))
        if idx == 0:
            xgb_clf_cv = xgb.train(params=param, dtrain=dtrain, 
                              num_boost_round=param['num_boost_round'],
                              evals=[(dtrain, "train"),(dtest, "validation")],
                              early_stopping_rounds=50,
                              evals_result=evals_result,
                              callbacks=[pruning_callback]
                              )
        else:
            xgb_clf_cv = xgb.train(params=param, dtrain=dtrain, 
                              num_boost_round=param['num_boost_round'],
                              evals=[(dtrain, "train"),(dtest, "validation")],
                              early_stopping_rounds=50,
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
    print(self.cls_weight)
    trial.set_user_attr("scale_pos_weight", self.cls_weight)
    trial.set_user_attr("eta", param['eta'])
    trial.set_user_attr("max_depth", param['max_depth'])

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
    def __init__(self, foldsplit, nfold, storage, study_name_prefix, outdir, scoring):
      # Hold this implementation specific arguments as the fields of the class.
      self.outer_split = foldsplit
      self.nfold = nfold
      self.storage = storage
      self.study_name_prefix=study_name_prefix
      self.outdir=outdir
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
    def create_outer_fold(self, X, y, groups):
        #######################
        # nested cv structure #
        #######################

        self.groups = groups
        outer_index = 0
        #for split in self.outer_split.split(X,y):
        for split in self.outer_split.split(X,y,self.groups):
            #get indices for outersplit
            train_idx, test_idx = split

            #outer split data
            X_split = X.iloc[train_idx, :].copy()
            y_split = y.iloc[train_idx].copy()
            group_train = self.groups.iloc[train_idx]
            X_split.to_csv(self.outdir +'/'+self.study_name_prefix+'.Xsplit.'+str(outer_index)+'.txt', sep='\t')
            y_split.to_csv(self.outdir +'/'+self.study_name_prefix+'.ysplit.'+str(outer_index)+'.txt', sep='\t')
            group_train.to_csv(self.outdir +'/'+self.study_name_prefix+'.groupsplit.'+str(outer_index)+'.txt', sep='\t')
            #self.X_splits[outer_index] = X_split
            #self.y_splits[outer_index] = y_split
            #self.group_splits[outer_index] = group_train
            ID_split = X_split['ABC.id']
            ID_split.to_csv(self.outdir +'/'+self.study_name_prefix+'.IDsplit.'+str(outer_index)+'.txt', sep='\t')
            X_split = X_split.drop(columns = ['ABC.id'])

            X_test = X.iloc[test_idx,:].copy()
            y_test = y.iloc[test_idx].copy()
            group_test = self.groups.iloc[test_idx]
            X_test.to_csv(self.outdir +'/'+self.study_name_prefix+'.Xtest.'+str(outer_index)+'.txt', sep='\t')
            y_test.to_csv(self.outdir +'/'+self.study_name_prefix+'.ytest.'+str(outer_index)+'.txt', sep='\t')
            group_test.to_csv(self.outdir +'/'+self.study_name_prefix+'.grouptest.'+str(outer_index)+'.txt', sep='\t')
            #self.X_tests[outer_index] = X_test
            #self.y_tests[outer_index] = y_test
            #self.group_tests[outer_index] = group_test
            ID_test = X_test['ABC.id']
            ID_test.to_csv(self.outdir +'/'+self.study_name_prefix+'.IDtest.'+str(outer_index)+'.txt', sep='\t')
            X_test = X_test.drop(columns = ['ABC.id'])

            outer_index=outer_index+1


    # Set up outer folds for testing 
    def create_studies(self, study_name_prefix, nfold):
        n_train_iter=100
        for outer_index in range(nfold):
            #get indices for outersplit
            storage = optuna.storages.RDBStorage(url="postgresql://mhan@localhost:"+str(postgres_port)+"/example")
            for model in models: 
                print("\nCreating Optuna for %s outer fold %d." % (model, outer_index), flush=True)
                pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource=n_train_iter, reduction_factor=3)
                #pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
                # xgb study
                study_name = study_name_prefix+'.'+model+"."+str(outer_index)
                #optuna.delete_study(study_name=study_name, storage=storage) # if there is existing study remove.
                study = optuna.create_study(study_name=study_name, direction="maximize", storage=storage, pruner=pruner, load_if_exists=True)




    def preprocess(self, X, y, group, scaler_dump):
        X = X.drop(columns = ['ABC.id'])
        cols = X.columns
        if os.path.exists(scaler_dump):
            scaler = joblib.load(scaler_dump)
        else: 
            scaler = preprocessing.MinMaxScaler()
            scaler.fit(X)
            joblib.dump(scaler, scaler_dump)
        X = pd.DataFrame(scaler.transform(X), columns = cols)
        y = y['Significant']
        if group is not None:
            group = group['group']
        return X, y, group
 


    def optimize_hyperparams(self, model, params, outer_index, n_inner_fold=4, scoring='map'):
        #outer_index = 0
        #for classifier, folds in self.helper.studies.items():
            #print(classifier)
            #print(folds)
            #for outer_index,study_name in folds.items():
            #    print(outer_index)
            #    print(study_name) 
        study_name = self.study_name_prefix+'.'+model+"."+str(outer_index)
        study = optuna.load_study(study_name=study_name, storage=self.storage) 
        print("Loaded study  %s with  %d trials." % (study_name, len(study.trials)))
        X_split = pd.read_csv(self.outdir +'/'+self.study_name_prefix+'.Xsplit.'+str(outer_index)+'.txt', sep='\t', index_col=0).reset_index(drop=True)
        y_split = pd.read_csv(self.outdir +'/'+self.study_name_prefix+'.ysplit.'+str(outer_index)+'.txt', sep='\t', index_col=0).reset_index(drop=True)
        group_split = pd.read_csv(self.outdir +'/'+self.study_name_prefix+'.groupsplit.'+str(outer_index)+'.txt', sep='\t', index_col=0).reset_index(drop=True)
        scaler_dump = self.outdir +'/'+self.study_name_prefix+'.scaler.'+str(outer_index)+'.gz'

        X_split, y_split, group_split = self.preprocess(X_split, y_split, group_split, scaler_dump=scaler_dump)
        counter = Counter(y_split)
        estimate = counter[0] / counter[1]
        cls_weight = np.sqrt((y_split.shape[0] - np.sum(y_split)) / np.sum(y_split))
        
        #run Optuna search with inner search CV on outer split data 
        #inner_splits = GroupKFold(n_splits=n_inner_fold, shuffle=True, random_state=RANDOM_SEED)
        inner_splits = StratifiedGroupKFold(n_splits=n_inner_fold, shuffle=False, random_state=None)
        custom_fold = []  #list of (train, test) indices
        for split in inner_splits.split(X_split,y_split,group_split):
            train_idx, test_idx = split
            custom_fold.append((train_idx, test_idx))
        study.optimize(Objective(X_split, y_split, model, params, custom_fold, study_name, scoring, cls_weight), n_trials=2000, timeout=600, n_jobs=1)  # will run  process to cover 2000 approx trials 
     

    def feature_selection(self, model, outer_index):
        study_name = self.study_name_prefix+'.'+model+"."+str(outer_index)
        study = optuna.load_study(study_name=study_name, storage=self.storage) 
        print("Number of finished trials for  %s: %d." % (study_name, len(study.trials)))
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
        params['scale_pos_weight'] = trial.user_attrs["scale_pos_weight"]
        params['eta'] = trial.user_attrs["eta"]
        params['max_depth']=trial.user_attrs['max_depth']
        params['objective'] = "binary:logistic"

        if (classifier == 'rf'):
            params['num_boost_round'] = 1
        X_train = pd.read_csv(self.outdir +'/'+self.study_name_prefix+'.Xsplit.'+str(outer_index)+'.txt', sep='\t', index_col=0)
        y_train = pd.read_csv(self.outdir +'/'+self.study_name_prefix+'.ysplit.'+str(outer_index)+'.txt', sep='\t', index_col=0)
        scaler_dump = self.outdir +'/'+self.study_name_prefix+'.scaler.'+str(outer_index)+'.gz'
        X_train, y_train, group_train =  self.preprocess(X_train, y_train, group=None, scaler_dump=scaler_dump)
        dtrain = xgb.DMatrix(X_train)

        #dtrainfilename = outdir +'/'+'dtrain.'+str(outer_index)+'.data'
        #dtrain = xgb.DMatrix(dtrainfilename)

        print("  Params: ")
        for key, value in params.items():
            print("    {}: {}".format(key, value))
        xgb_clf_tuned_2 = xgb.XGBClassifier(
          #colsample_bytree=0.8, 
          colsample_bytree=params['colsample_bytree'], 
          subsample=params['subsample'], 
          gamma=params['gamma'], 
          #lambda=params['lambda'], 
          eta=params['eta'],
          max_depth=params['max_depth'],
          max_delta_step=1, 
          #max_depth=params['max_depth'],
          min_child_weight=params['min_child_weight'],
          scale_pos_weight=params['scale_pos_weight'], 
          n_estimators=trial.user_attrs['best_iteration']
        )
        print(xgb_clf_tuned_2)
        # feature selection after a round of hyperparam optimization
        Feature_Selector = BorutaShap(model=xgb_clf_tuned_2,
                              importance_measure='shap',
                              classification=True, 
                              percentile=80, pvalue=0.1)
        Feature_Selector.fit(X=X_train, y=y_train, n_trials=50, sample=True,
                   train_or_test = 'train', normalize=True, verbose=True)
        fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(400, 50))
        Feature_Selector.plot(which_features='all')
        plt.savefig(self.outdir+'/'+'boruta.'+self.study_name_prefix+'.'+str(outer_index)+'.pdf')
        plt.close()
        Feature_Selector.results_to_csv(filename=self.outdir+'/'+self.study_name_prefix+'.feature_importance.'+str(outer_index)+'.txt')
        features_to_keep = pd.DataFrame({"features":Feature_Selector.Subset().columns.values})
        features_to_keep.to_csv(self.outdir+'/'+self.study_name_prefix+'.features_to_keep.'+str(outer_index)+'.txt', index=False, sep='\t')
        #features_to_remove = pd.DataFrame({"features":Feature_Selector.features_to_remove})
        #print(features_to_remove)
        #features_to_remove.to_csv(self.outdir+'/'+self.study_name_prefix+'.features_to_remove.'+str(outer_index)+'.txt', index=False, sep='\t')




    def drop_features(self):
        new_study_name_prefix = self.study_name_prefix+'.2pass'
        featurenames = np.loadtxt(self.outdir+'/'+self.study_name_prefix+'.featurelabels.txt', skiprows=1, dtype='str')
        print(featurenames)
        features_to_keep = []
        for outer_index in range(self.nfold):
            features_to_keep_fold = np.loadtxt(self.outdir+'/'+self.study_name_prefix+'.features_to_keep.'+str(outer_index)+'.txt', skiprows=1, dtype='str')
            print(features_to_keep_fold)
            features_to_keep = np.union1d(features_to_keep, features_to_keep_fold)
        print(features_to_keep)
        features_to_keep = np.append(features_to_keep, ['ABC.id'])
        features_to_drop = np.setdiff1d(featurenames, features_to_keep)
        #features_to_drop = np.append(features_to_drop, ['ABC.Score.Numerator', "ABC.Score.mean"])
        print(features_to_drop)
        np.savetxt(self.outdir+'/'+new_study_name_prefix+'.features_kept.txt', np.transpose([features_to_keep]), fmt="%s")

        os.symlink(os.path.abspath(self.outdir +'/'+self.study_name_prefix+'.learninginput.txt'), self.outdir +'/'+new_study_name_prefix+'.learninginput.txt')
        for outer_index in range(self.nfold):
            src_path = self.outdir +'/'+self.study_name_prefix+'.Xsplit.'+str(outer_index)+'.txt'
            #X_split = pd.read_csv(src_path, sep='\t', index_col=0).reset_index(drop=True)
            X_split = pd.read_csv(src_path, sep='\t', index_col=0)
            X_split = X_split.drop(columns = features_to_drop)
            X_split.to_csv(self.outdir +'/'+new_study_name_prefix+'.Xsplit.'+str(outer_index)+'.txt', sep='\t')
            X_split = X_split.drop(columns = ['ABC.id'])

            os.symlink(os.path.abspath(self.outdir +'/'+self.study_name_prefix+'.ysplit.'+str(outer_index)+'.txt'), self.outdir +'/'+new_study_name_prefix+'.ysplit.'+str(outer_index)+'.txt')
            os.symlink(os.path.abspath(self.outdir +'/'+self.study_name_prefix+'.groupsplit.'+str(outer_index)+'.txt'), self.outdir +'/'+new_study_name_prefix+'.groupsplit.'+str(outer_index)+'.txt')
            os.symlink(os.path.abspath(self.outdir +'/'+self.study_name_prefix+'.IDsplit.'+str(outer_index)+'.txt'), self.outdir +'/'+new_study_name_prefix+'.IDsplit.'+str(outer_index)+'.txt')
       
            src_path = self.outdir +'/'+self.study_name_prefix+'.Xtest.'+str(outer_index)+'.txt'
            #X_test = pd.read_csv(src_path, sep='\t', index_col=0).reset_index(drop=True)
            X_test = pd.read_csv(src_path, sep='\t', index_col=0)
            X_test = X_test.drop(columns = features_to_drop)
            X_test.to_csv(self.outdir +'/'+new_study_name_prefix+'.Xtest.'+str(outer_index)+'.txt', sep='\t')
            X_test = X_test.drop(columns = ['ABC.id'])
            os.symlink(os.path.abspath(self.outdir +'/'+self.study_name_prefix+'.ytest.'+str(outer_index)+'.txt'), self.outdir +'/'+new_study_name_prefix+'.ytest.'+str(outer_index)+'.txt')
            os.symlink(os.path.abspath(self.outdir +'/'+self.study_name_prefix+'.grouptest.'+str(outer_index)+'.txt'), self.outdir +'/'+new_study_name_prefix+'.grouptest.'+str(outer_index)+'.txt')
            os.symlink(os.path.abspath(self.outdir +'/'+self.study_name_prefix+'.IDtest.'+str(outer_index)+'.txt'), self.outdir +'/'+new_study_name_prefix+'.IDtest.'+str(outer_index)+'.txt')
 


 
    # test and summarize outer fold results based on best hyperparms
    def test_results(self):
        for classifier in models:
            for outer_index in range(self.nfold):
                print(classifier)
                print(outer_index)
                study_name = self.study_name_prefix+'.'+classifier+"."+str(outer_index)
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
                #params['scale_pos_weight'] = cls_weight
                params['scale_pos_weight'] = trial.user_attrs["scale_pos_weight"]
                params['objective'] = "binary:logistic" 
                params['eta'] = trial.user_attrs['eta']
                params['max_delta_step'] = 1
                params['max_depth'] = trial.user_attrs['max_depth']
                params['best_trial_Value'] = trial.value

                if (classifier == 'rf'):
                    params['num_boost_round'] = 1



                X_train = pd.read_csv(self.outdir +'/'+self.study_name_prefix+'.Xsplit.'+str(outer_index)+'.txt', sep='\t', index_col=0)
                y_train = pd.read_csv(self.outdir +'/'+self.study_name_prefix+'.ysplit.'+str(outer_index)+'.txt', sep='\t', index_col=0)
                scaler_dump = self.outdir +'/'+self.study_name_prefix+'.scaler.'+str(outer_index)+'.gz'
                X_train, y_train, group_train = self.preprocess(X_train, y_train, group=None, scaler_dump=scaler_dump) 
                dtrain = xgb.DMatrix(X_train, label=y_train)
                xgb_clf_tuned_2 = xgb.train(params=params, dtrain=dtrain, num_boost_round=trial.user_attrs['best_iteration'])
                lst_vars_in_model = xgb_clf_tuned_2.feature_names
                print(lst_vars_in_model)
                featurenames = pd.DataFrame({"features":lst_vars_in_model})
                featurenames.to_csv(self.outdir+'/'+self.study_name_prefix+'.featurenames'+'.'+str(outer_index)+'.txt', index=False, sep='\t')
                best_iteration = trial.user_attrs['best_iteration']
                print("new best_iteration: "+str(xgb_clf_tuned_2.best_iteration))
                print("old best_iteration: "+str(best_iteration))
                xgb_clf_tuned_2.save_model(self.outdir+'/'+self.study_name_prefix+'.save'+'.'+str(outer_index)+'.json')
                xgb_clf_tuned_2.dump_model(self.outdir+'/'+self.study_name_prefix+'.dump'+'.'+str(outer_index)+'.json')
                config_str = xgb_clf_tuned_2.save_config()
                with open(self.outdir+'/'+self.study_name_prefix+'.config'+'.'+str(outer_index)+'.json', "w") as config_file: 
                    config_file.write(config_str)
                print(config_str)
                
                X_test = pd.read_csv(self.outdir +'/'+self.study_name_prefix+'.Xtest.'+str(outer_index)+'.txt', sep='\t', index_col=0)
                y_test = pd.read_csv(self.outdir +'/'+self.study_name_prefix+'.ytest.'+str(outer_index)+'.txt', sep='\t', index_col=0)
                X_test, y_test, group_test = self.preprocess(X_test, y_test, group=None, scaler_dump=scaler_dump) 
                dtest = xgb.DMatrix(X_test, label=y_test)
                y_pred_prob = xgb_clf_tuned_2.predict(dtest, ntree_limit=best_iteration)
                print(y_pred_prob)
                # Data to plot precision - recall curve
                precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob, pos_label = 1)
                print(precision)
                print(recall)
                pr = pd.DataFrame({'precision':precision,'recall':recall,'thresholds':np.append(thresholds,None)})
                pr.to_csv(self.outdir+'/'+self.study_name_prefix+'.pr_curve.'+classifier+'.'+str(outer_index)+'.txt', sep='\t')
                # find threshold for confusion matrix at recall == 0.70
                recallmin = recall-0.70
                ix = np.where(recallmin >= 0, recallmin, np.inf).argmin()
                y_pred = [value>=thresholds[ix] for value in y_pred_prob]
                #y_pred = [round(value) for value in y_pred_prob]
                test_pd = pd.DataFrame({'y_pred':y_pred, 'y_prob':y_pred_prob}, index=y_test.index)
                y_res = pd.merge(y_test, test_pd, left_index=True, right_index=True)
                #res = pd.merge(y_res, X_test, left_index=True, right_index=True)
                y_res.to_csv(self.outdir+'/'+self.study_name_prefix+'.confusion.'+classifier+'.'+str(outer_index)+'.txt', sep='\t')

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
        self.outer_results.to_csv(outdir+'/'+self.study_name_prefix+'.outer_results.txt', sep='\t')


    # next func






# python src/mira_cross_val_bayescv.eroles.xgb.optuna.py --dir /data8/han_lab/mhan/abcd/data/ --outdir /data8/han_lab/mhan/abcd/run.groupcv --port 44803
# python -i src/mira_cross_val_bayescv.eroles.xgb.optuna.py --dir /data8/han_lab/mhan/abcd/data/ --outdir /data8/han_lab/mhan/abcd/run.groupcv --port 44803 --model 'rf' --outerfold 2
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--dir', required=False, help="directory containing edgelist and vertices files")
  parser.add_argument('--infile', required=False, help="infile name")
  parser.add_argument('--outdir', default='.', help="directory containing edgelist and vertices files")
  parser.add_argument('--chr', default='all', help="chromosome")
  parser.add_argument("--port", required=True, help="postgres port for storage")
  parser.add_argument("--studyname", required=True, help="study name prefix")
  parser.add_argument("--init", action='store_true', help="create outer folds") # if on, create outer folds and optuna studies 
  parser.add_argument("--init_reduced", action='store_true', help="create outer folds") # if on, create outer folds and optuna studies for a reduced model without TF features 
  parser.add_argument("--init2pass", action='store_true', help="create outer folds") # if on, create optuna studies for 2nd pass hyperparam optimization after feature selection 
  parser.add_argument("--opt", action='store_true', help="parallel optimize") # if on, add parallel optimizers 
  parser.add_argument("--fs", action='store_true', help="featureselection") # if on, feature selection 
  parser.add_argument("--dropfeatures", action='store_true', help="drop features") # if on, drop features except selected 
  parser.add_argument("--test", action='store_true', help="gather test results based on tuned model") # if on, gather test results 
  parser.add_argument("--model", default='all', help="choose one of xgb, rf, lr only when optimizing")
  parser.add_argument("--params", default='all', help="choose one of parameter sets only when optimizing")
  parser.add_argument("--outerfold", default='all', help="choose one of each outer fold only when optimizing")
  parser.add_argument("--e1", default=False, action='store_true', help="use only e1 pairs")
  parser.add_argument("--e1minus", default=False, action='store_true', help="use only e0e1 pairs")
  parser.add_argument("--e2plus", default=False, action='store_true', help="use only e2plus pairs")
  parser.add_argument("--e3plus", default=False, action='store_true', help="use only e3plus pairs")

  args=parser.parse_args()
  pid = os.getpid()

  base_directory = args.dir
  infile = args.infile
  chromosome = args.chr
  outdir = args.outdir
  postgres_port = args.port
  study_name_prefix = args.studyname
  run_init = args.init
  run_init_reduced = args.init_reduced
  run_init2pass = args.init2pass
  run_optimize = args.opt
  run_feature_selection = args.fs
  run_drop_features = args.dropfeatures
  classifier = args.model
  modelparams = args.params
  outerfold = args.outerfold
  run_test = args.test

  filenamesuffix = ''
  subdir = ""
  if ((args.e1 + args.e1minus + args.e2plus + args.e3plus)>1):
      print ("--e1, --e1minus, --e2plus and --e3plus are mutually exclusive. please use only one option")
      quit()
  elif args.e1:
      subdir = "e1"
  elif args.e1minus:
      subdir = "e1minus"
  elif args.e2plus:
      subdir = "e2plus"
  elif args.e3plus:
      subdir = "e3plus"
  outdir = outdir+"/"+subdir
  Path(outdir).mkdir(parents=True, exist_ok=True)

  # create outer fold object
  nfold = 4
  outer_split = StratifiedGroupKFold(n_splits=nfold, shuffle=False, random_state=None)
  storage = optuna.storages.RDBStorage(url="postgresql://mhan@localhost:"+str(postgres_port)+"/example")
  outerFolds = OuterFolds(outer_split, nfold, storage, study_name_prefix, outdir, '')
  if (run_init == True | run_init_reduced == True):
      #################################
      #import our data, then format it #
      ##################################

      if infile is None:
        print( "--infile is required to run the --init process")
        quit()
      if base_directory is None:
        print( "--base_directory is required to run the --init process")
        quit()
      data2 = pd.read_csv(base_directory+'/'+infile,sep='\t', header=0, index_col=0)
      data2 = data2.loc[:,~data2.columns.str.match("Unnamed")]
      if (args.e1):
        data2 = data2.loc[data2['e1']==1,]
      elif (args.e1minus):
        data2 = data2.loc[(data2['e0']==1) | (data2['e1']==1),]
      elif (args.e2plus):
        data2 = data2.loc[(data2['e0']!=1) & (data2['e1']!=1),]
      elif (args.e3plus):
        data2 = data2.loc[(data2['e0']!=1) & (data2['e1']!=1) & (data2['e2']!=1),]
      print(data2)


      features_gasperini = data2
      ActivityFeatures = features_gasperini[['ABC.id', 'normalized_h3K27ac', 'normalized_h3K4me3', 'normalized_h3K27me3', 'normalized_dhs', 'TargetGeneExpression', 'distance', 'H3K27ac.RPKM.quantile.TSS1Kb', 'H3K4me3.RPKM.quantile.TSS1Kb', 'H3K27me3.RPKM.quantile.TSS1Kb']].copy()
      ActivityFeatures.rename(columns={'normalized_h3K27ac':'normalized_h3K27ac_e', 'normalized_h3K4me3':'normalized_h3K4me3_e', 'normalized_h3K27me3':'normalized_h3K27me3_e','H3K27ac.RPKM.quantile.TSS1Kb':'H3K27ac.RPKM.quantile_TSS', 'H3K4me3.RPKM.quantile.TSS1Kb':'H3K4me3.RPKM.quantile_TSS', 'H3K27me3.RPKM.quantile.TSS1Kb':'H3K27me3.RPKM.quantile_TSS'}, inplace=True)
      ActivityFeatures = ActivityFeatures.dropna()
      ActivityFeatures['TargetGeneExpression'] = np.log1p(ActivityFeatures['TargetGeneExpression'])
      hicfeatures = features_gasperini[['hic_contact', 'Enhancer.count.near.TSS', 'zscore.contact.to.TSS', 'diff.from.max.contact.to.TSS', 'remaining.enhancers.contact.to.TSS', 'TSS.count.near.enhancer', 'zscore.contact.from.enhancer', 'diff.from.max.contact.from.enhancer', 'remaining.TSS.contact.from.enhancer' ]].copy()
      hicfeatures = hicfeatures.dropna()
      if not run_init_reduced:
        TFfeatures = features_gasperini.filter(regex='(_e)|(_TSS)|(NMF)').copy()
        TFfeatures = TFfeatures.dropna()
      crisprfeatures = features_gasperini[['EffectSize', 'Significant', 'pValue' ]].copy()
      crisprfeatures = crisprfeatures.dropna()
      groupfeatures = features_gasperini[['group']].copy()

      features = ActivityFeatures.copy()
      features = pd.merge(features, hicfeatures, left_index=True, right_index=True)
      if not run_init_reduced:
        features = pd.merge(features, TFfeatures, left_index=True, right_index=True)
      features = pd.merge(features, crisprfeatures, left_index=True, right_index=True)
      data = pd.merge(features, groupfeatures, left_index=True, right_index=True)
      ActivityFeatures = data.iloc[:, :ActivityFeatures.shape[1]]
      hicfeatures = data.iloc[:, ActivityFeatures.shape[1]:ActivityFeatures.shape[1]+hicfeatures.shape[1]]
      if not run_init_reduced:
        TFfeatures = data.iloc[:, ActivityFeatures.shape[1]+hicfeatures.shape[1]:ActivityFeatures.shape[1]+hicfeatures.shape[1]+TFfeatures.shape[1]]
      crisprfeatures = data[['EffectSize', 'Significant', 'pValue' ]]
      groupfeatures = data[['group']]
      f1 = set(list(features.columns))
      features = data.iloc[:, :data.shape[1]-crisprfeatures.shape[1]-groupfeatures.shape[1]]
      f2 = set(list(features.columns))
      target = crisprfeatures['Significant'].astype(int)
      groups = groupfeatures


      featurelabels = pd.DataFrame({"features":features.columns})
      featurelabels.to_csv(outdir+'/'+study_name_prefix+'.featurelabels.txt', index=False, sep='\t')
      features_gasperini.filter(items = data.index,axis=0).to_csv(outdir+'/'+study_name_prefix+'.learninginput.txt', sep='\t')

      X = features
      y = target

      outerFolds.create_outer_fold(X, y, groups)
      outerFolds.create_studies(study_name_prefix, nfold)
  elif (run_init2pass == True):
      outerFolds.create_studies(study_name_prefix+'.2pass', nfold)
  elif (run_optimize == True): 
      outerFolds.optimize_hyperparams(classifier, modelparams, outerfold)
  elif (run_feature_selection == True): 
      outerFolds.feature_selection(classifier, outerfold)
  elif (run_drop_features == True):
      outerFolds.drop_features()
  elif (run_test == True):
      outerFolds.test_results()
  storage.remove_session()


  print('Total runtime: ' + str(time.time() - tstart))    

