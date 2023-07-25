import numpy as np
import time
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
from sklearn.metrics import precision_recall_curve, balanced_accuracy_score, make_scorer
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
import pandas as pd

tstart = time.time()

#helper class that allows you to iterate over multiple classifiers within the nested for loop
class EstimatorSelectionHelper:
    def __init__(self, models, params, dimrs=None, dimr_params=None):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError("Some estimators are missing parameters: %s" % missing_params)
        self.models = models
        self.params = params
        self.dimrs = dimrs
        self.dimr_params = dimr_params
        self.keys = models.keys()
        self.grid_searches = {}
        self.scores = {}
        self.best_estimator_ = None
        self.best_estimators_ = {}

    def fit(self, X, y, cv=3, n_jobs=3, verbose=1, scoring=None, refit=False):
        for key in self.keys:
            if self.dimrs is None:
                print("Running GridSearchCV for %s." % key)
                model = self.models[key]
                params = self.params[key]
                gs = GridSearchCV(model, params, cv=cv, n_jobs=n_jobs,
                              verbose=verbose, scoring=scoring, refit=refit,
                              return_train_score=True)
                gs.fit(X,y)
                self.grid_searches[key] = gs    
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

    def best_estimator(self, score='max_score', method='train', X=None, y=None):
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
            test_results = pd.DataFrame(columns=['DimReduction','Classifier','test_bal_accuracy','clf_idx'])
            clfs = []
            clfidx = 0
            #choose best estimator from each gridsearch
            #also store in self.best_estimators_
            for gs in grid_searches:
                clf0 = grid_searches[gs].best_estimator_
                #compute test accuracy
                test_acc = balanced_accuracy_score(y, clf0.predict(X))
                dr, cl = gs.split('_')
                self.best_estimators_[gs] = {'clf':clf0, 'bal_acc':test_acc}
                test_results = test_results.append(pd.DataFrame({'DimReduction':[dr],'Classifier':[cl],'test_bal_accuracy':[test_acc], 'clf_idx':[clfidx]}), ignore_index=True)
                clfs.append(clf0)
                clfidx += 1
            #choose clf with highest test accuracy
            clfidx = test_results.loc[test_results['test_bal_accuracy'] == test_results.test_bal_accuracy.max(), 'clf_idx'].values[0]
            self.best_estimator_ = clfs[clfidx]
            return(clfs[clfidx])
            
    def best_params(self, score='max_score'):
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
out = open('logs/cross_validation.log','w')

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
    plt.savefig('data/top_'+str(top_features)+'_features.png')


#X, y = load_digits(return_X_y=True)

##################################
#import our data, then format it #
##################################

data2 = pd.read_csv('data/full_feature_matrix.promPOL.subset2.tsv',sep='\t', header=0)

#data2['source'] = 'gasperini'
#data1['source'] = 'fulco'
#for some reason MCM5 didnt run on the new dataset, so remove that col
#data = pd.concat([data1,data2],ignore_index=True)
#normalize all non binary variables
normalizer = preprocessing.MinMaxScaler()

data2['activity'] = normalizer.fit_transform(data2[["activity"]].values)
data2['contact'] = normalizer.fit_transform(data2[["contact"]].values)
data2['abc_score'] = normalizer.fit_transform(data2[["abc_score"]].values)
data2['degree'] = normalizer.fit_transform(data2[["degree"]].values)
data2['effect_size'] = normalizer.fit_transform(data2[["effect_size"]].values)
data2['sig'] = data2.sig.astype(int)

#normalize promoter rnapol signal
for i in ['POLR2A_promoter', 'POLR2AphosphoS2_promoter', 'POLR2AphosphoS5_promoter', 'POLR2B_promoter', 'POLR2G_promoter', 'POLR3A_promoter', 'POLR3G_promoter']:
    data2[i] = normalizer.fit_transform(data2[[i]].values)


#add genomic distance param
data2['midpoint'] = (data2['stop']-data2['start'])/2
data2['distance'] = abs(data2['midpoint'] - data2['tss'])
data2.drop(columns=['midpoint'],inplace=True)
data2['distance'] = normalizer.fit_transform(data2[["distance"]].values)

#remove contact cols
#data2.drop(columns=['contact','degree'],inplace=True)


#activity histograms for data1 v 2
#plt.hist(data1['activity'], density=True, bins=50)
#plt.xlabel('activity normalized')
#plt.savefig('data/ds1_activity_histogram.png')

#plt.hist(data2['activity'], density=True, bins=50)
#plt.xlabel('activity normalized')
#plt.savefig('data/ds2_activity_histogram.png')

#code roles as binary
data2['e1'] = 0
data2['e2'] = 0
data2['e3'] = 0

data2.loc[data2['role']=='E1','e1'] = 1
data2.loc[data2['role']=='E2','e2'] = 1
data2.loc[data2['role']=='E3','e3'] = 1

data2.drop(labels=['chr','start','stop','tss','gene','role','abc_score','effect_size'], axis=1, inplace=True)
data2.dropna(inplace=True)

pos_data2 = data2.loc[data2['sig']==1,]
neg_data2 = data2.loc[data2['sig']==0,]

data2 = pd.concat([pos_data2, neg_data2]).copy()
data2.replace([np.inf, -np.inf], np.nan)
data2.dropna(inplace=True)

X2 = data2.loc[:,data2.columns != 'sig'].to_numpy()
y2 = data2.loc[:,data2.columns == 'sig'].to_numpy().T[0]

#X=np.concatenate((X1,X2), axis=0)
#y=np.concatenate((y1, y2), axis=0)
#until I add effect_size and degree to data1, train on data2
X=X2
y=y2

#####################################
# define the classifiers and params #
#####################################

#in this script we use diff params for bayescv

models = {
    'RandomForestClassifier':RandomForestClassifier(),
    'SVC': SVC(max_iter=1500000),
    'LogisticRegression':LogisticRegression(max_iter=1500000,solver='liblinear'),
    }

params = {
    'RandomForestClassifier':{'n_estimators': Integer(20,400),'ccp_alpha':Real(0,5),'min_samples_leaf':Integer(1,10)},
    'SVC': {'C':Real(1e-6,1e6, prior='log-uniform'),'gamma': Real(1e-6, 1e+1, prior='log-uniform'),'kernel': Categorical(['linear', 'sigmoid', 'rbf'])},
    'LogisticRegression':{'C':Real(1e-6,1e5, prior='log-uniform')},
    }

dim_reductions = {
    'SelectKBest':SelectKBest(chi2),
    'PCA':PCA(iterated_power=100),
   }

dimr_params = {
    'SelectKBest':{'k':Integer(1,50)},
    'PCA':{'n_components':Integer(1,50)},
}
#######################
# nested cv structure #
#######################
test_sz = 0.2
inner_split = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
outer_split = StratifiedKFold(n_splits=3, shuffle=True, random_state=2)
outer_results = pd.DataFrame()
outer_index = 0
best_estimators = {}
for split in outer_split.split(X,y):
    #get indices for outersplit
    train_idx, test_idx = split

    #outer split data
    X_split = X[train_idx, :]
    y_split = y[train_idx]
    
    #grid search outer split data with inner search CV
    #init helper
    helper = EstimatorSelectionHelper(models, params, dimrs=dim_reductions, dimr_params=dimr_params)
    #helper fit on inner cv
    helper.fit(X_split, y_split, cv=inner_split, scoring=make_scorer(balanced_accuracy_score), n_jobs=10, refit=True)
    #get best performing models 
    helper.score_summary(sort_by='max_score')
    #helper performs the inner gridsearch by itself, but by using the best_estimator(method='test') command, we can run the outer gridsearch using test data
    clf = helper.best_estimator(method='test',X=X[test_idx,:], y=y[test_idx])
    best_params = helper.best_params()
    #helper also stores the best estimator of each combination of dimr and clf, so store the most accurate ones    
    temp_estimators = helper.best_estimators_
    for be in temp_estimators:
        acc = temp_estimators[be]['bal_acc']
        if be not in best_estimators:
            best_estimators[be] = temp_estimators[be]
        elif best_estimators[be]['bal_acc'] < acc:
            best_estimators[be] = temp_estimators[be]
             
    #return the best performing model on test data
    bal_accuracy = balanced_accuracy_score(y[test_idx], clf.predict(X[test_idx,:]))
    best_params['bal_accuracy'] = [bal_accuracy]
    outer_results = pd.concat([outer_results,best_params])

    #fnames = data1.loc[:,data1.columns != 'sig'].columns[[int(x[1:]) for x in clf[:-1].get_feature_names_out()]].tolist()
    #fweights = clf.named_steps[clf_label].coef_.ravel()
    #frank = rank(abs(clf.named_steps[clf_label].coef_.ravel()))
    #for i in range(0,len(fnames)):
    #    feature = fnames[i]
    #    if feature not in feature_ranks:
    #        feature_ranks[feature] = fweights[i]*frank[i]/len(frank)
    #    else:
    #        feature_ranks[feature] += fweights[i]*frank[i]/len(frank)
pd.set_option('display.max_columns', None) 
print(outer_results) 
print(best_estimators)
#save best estimators 
for est in best_estimators:
    joblib.dump(best_estimators[est]['clf'], 'data/trained_models/contact_models/'+est+'.pkl')
print('Total runtime: ' + str(time.time() - tstart))    
exit()
