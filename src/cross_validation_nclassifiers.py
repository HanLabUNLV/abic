import numpy as np
import joblib
from  scipy.stats import rankdata as rank
import matplotlib.pyplot as plt
from statistics import mean, stdev
from sklearn import preprocessing
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA, NMF
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import precision_recall_curve, balanced_accuracy_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
import pandas as pd


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
                print("Running GridSearchCV for %s." % key)
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

                    gs = GridSearchCV(pipeline, gs_params, cv=cv, n_jobs=n_jobs,
                        verbose=verbose, scoring=scoring, refit=refit,
                        return_train_score=True)
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

data1 = pd.read_csv('data/full_feature_matrix.coboundp.dataset1.tsv',sep='\t', header=0)
data2 = pd.read_csv('data/full_feature_matrix.coboundp.merged.validated.tsv',sep='\t', header=0)

#data1['source'] = 'gasperini'
#data2['source'] = 'fulco'
#for some reason MCM5 didnt run on the new dataset, so remove that col
data1 = data1.loc[:,data1.columns != 'MCM5']
data1 = data1.loc[:,data1.columns != 'MCM5_p_cobound']
#data = pd.concat([data1,data2],ignore_index=True)
#normalize all non binary variables
normalizer = preprocessing.MinMaxScaler()
data1['activity'] = normalizer.fit_transform(data1[["activity"]].values)
data1['contact'] = normalizer.fit_transform(data1[["contact"]].values)
data1['abc_score'] = normalizer.fit_transform(data1[["abc_score"]].values)


data2['activity'] = normalizer.fit_transform(data2[["activity"]].values)
data2['contact'] = normalizer.fit_transform(data2[["contact"]].values)
data2['abc_score'] = normalizer.fit_transform(data2[["abc_score"]].values)

#activity histograms for data1 v 2
#plt.hist(data1['activity'], density=True, bins=50)
#plt.xlabel('activity normalized')
#plt.savefig('data/ds1_activity_histogram.png')

#plt.hist(data2['activity'], density=True, bins=50)
#plt.xlabel('activity normalized')
#plt.savefig('data/ds2_activity_histogram.png')

#code roles as binary
data1['e1'] = 0
data1['e2'] = 0
data1['e3'] = 0

data2['e1'] = 0
data2['e2'] = 0
data2['e3'] = 0

data1.loc[data1['role']=='E1','e1'] = 1
data1.loc[data1['role']=='E2','e2'] = 1
data1.loc[data1['role']=='E3','e3'] = 1

data2.loc[data2['role']=='E1','e1'] = 1
data2.loc[data2['role']=='E2','e2'] = 1
data2.loc[data2['role']=='E3','e3'] = 1

data1.drop(labels=['chr','start','stop','tss','classification','gene','role','abc_score'], axis=1, inplace=True)
data1.dropna(inplace=True)

data2.drop(labels=['chr','start','stop','tss','classification','gene','role','abc_score'], axis=1, inplace=True)
data2.dropna(inplace=True)

pos_data1 = data1.loc[data1['sig']==1,]
neg_data1 = data1.loc[data1['sig']==0,].sample(200)

pos_data2 = data2.loc[data2['sig']==1,]
neg_data2 = data2.loc[data2['sig']==0,].sample(8000)

data1 = pd.concat([pos_data1, neg_data1])

data2 = pd.concat([pos_data2, neg_data2])

X1 = data1.loc[:,data1.columns != 'sig'].to_numpy()
y1 = data1.loc[:,data1.columns == 'sig'].to_numpy().T[0]

X2 = data2.loc[:,data2.columns != 'sig'].to_numpy()
y2 = data2.loc[:,data2.columns == 'sig'].to_numpy().T[0]

X=np.concatenate((X1,X2), axis=0)
y=np.concatenate((y1, y2), axis=0)


#####################################
# define the classifiers and params #
#####################################
models = {
    'RandomForestClassifier':RandomForestClassifier(),
    'LinearSVC': LinearSVC(max_iter=30000),
    'LogisticRegression':LogisticRegression(max_iter=30000),
    }

params = {
    'RandomForestClassifier':{'n_estimators':[20, 50, 100]},
    'LinearSVC': {'C':[0.01, 0.1, 1,10]},
    'LogisticRegression':{'C':[0.01, 0.1, 1,10]},
    }

dim_reductions = {
    'SelectKBest':SelectKBest(chi2),
    'PCA':PCA(iterated_power=100),
    }

dimr_params = {
    'SelectKBest':{'k':[1,2,3,4,5,10,20,35,50,100,150] },
    'PCA':{'n_components':[1,2,3,4,5,10]},
}
#######################
# nested cv structure #
#######################
test_sz = 0.2
nfeats = [1,2,3,4,5,10,20,35,50,100,150]
inner_split = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
outer_split = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)
inner_model = SelectKBest(chi2)
pipeline = Pipeline([('kbest', inner_model), ('svc',LinearSVC(max_iter=10000))])
inner_gridsearch = GridSearchCV(pipeline,{'kbest__k': nfeats},cv=inner_split, n_jobs=5, return_train_score=True)
results = pd.DataFrame(columns=['k', 'inner_partition','outer_partition', 'train', 'test'])
outer_results = pd.DataFrame()
outer_index = 0
feature_ranks = {}
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
    helper.fit(X_split, y_split, cv=inner_split, scoring='accuracy', n_jobs=10, refit=True)
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
    
print(outer_results) 
print(best_estimators)
#save best estimators 
for est in best_estimators:
    joblib.dump(best_estimators[est]['clf'], 'data/trained_models/'+est+'.pkl')
    
exit()
print(feature_ranks)
wrank = open('data/weighted_feature_rank.tsv','w')
for feature in feature_ranks:
    wrank.write(feature + '\t' + str(feature_ranks[feature]) + '\n')
results.to_csv('data/CV_kfeat_accuracy.tsv',index=False, sep='\t')
exit()
#choose best performing model

cmax = 0
clf = ''
for i in range(0, len(outer_results)):
    if outer_results[i][1]['kbest__k']==20:
        cmax=outer_results[i][2]
        clf=outer_results[i][0]
        k=outer_results[i][1]

#print([int(x[1:]) for x in clf[:-1].get_feature_names_out()])        
#plot_coefficients(clf['svc'], data1.loc[:,data1.columns != 'sig'].columns[[int(x[1:]) for x in clf[:-1].get_feature_names_out()]], 20)

y_score = clf.decision_function(X[test_idx,:])#predict_proba(X_test_selected)[:, 1]
y_test = y[test_idx]
precision, recall, thresholds = precision_recall_curve(y_test, y_score)
fig, ax = plt.subplots()
ax.plot(recall, precision, color='blue')
ax.set_title('Precision-Recall Curve k='+str(k['kbest__k']))
ax.set_ylabel('Precision')
ax.set_xlabel('Recall')
plt.savefig('data/pr_curve.'+str(k['kbest__k'])+'.png')


exit()

results = []


for train_idx, test_idx in skf.split(X,y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    clf = LinearSVC()
    clf.fit(X_train, y_train)
    results.append([clf.score(X_train, y_train), clf.score(X_test, y_test)])
test_result = [x[1] for x in results]
print('List of possible test accuracy:', test_result)
print('\nMaximum Accuracy That can be obtained from this model is:',
      max(test_result)*100, '%')
print('\nMinimum Accuracy:',
      min(test_result)*100, '%')
print('\nOverall Accuracy:',
      mean(test_result)*100, '%')
print('\nStandard Deviation is:', stdev(test_result))

test_result = [x[0] for x in results]
print('List of possible train accuracy:', test_result)
print('\nMaximum Accuracy That can be obtained from this model is:',
      max(test_result)*100, '%')
print('\nMinimum Accuracy:',
      min(test_result)*100, '%')
print('\nOverall Accuracy:',
      mean(test_result)*100, '%')
print('\nStandard Deviation is:', stdev(test_result))

exit()

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=test_sz)

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=test_sz)

X_train = np.concatenate((X1_train, X2_train), axis=0)
X_test = np.concatenate((X1_test, X2_test), axis=0)
y_train = np.concatenate((y1_train, y2_train), axis=0)
y_test = np.concatenate((y1_test, y2_test), axis=0)

#nfeat = 9 #num feat selected
for nfeat in [100]:#[5,10,15,30,50,100,200]:
    select = SelectKBest(chi2, k=nfeat)
    X_train_selected = select.fit_transform(X_train,y_train)
    #print(data.columns[model.get_support(indices=True)])
    X_test_selected = select.transform(X_test)
    x = data1.drop(labels=['sig'], axis=1, inplace=False)
    out.write('k='+str(nfeat)+'\n')
    out.write('Features: ' + ', '.join(x.columns[select.get_support()].tolist())+'\n')
    clf = LinearSVC()
    clf.fit(X_train_selected, y_train)
    out.write('Training accuracy: ' + str(clf.score(X_train_selected, y_train))+'\n')
    out.write('Test accuracy: ' + str(clf.score(X_test_selected, y_test))+'\n')
    out.write('\n\n')

    #generate coef plot
    plot_coefficients(clf, data1.loc[:,data1.columns != 'sig'].columns[select.get_support()], 40)
    #next generate pr curve

    y_score = clf.decision_function(X_test_selected)#predict_proba(X_test_selected)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_score)
    fig, ax = plt.subplots()
    ax.plot(recall, precision, color='blue')
    ax.set_title('Precision-Recall Curve k='+str(nfeat))
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    plt.savefig('data/pr_curve.'+str(nfeat)+'.png')
exit()
mean_scores = np.array(grid.cv_results_["mean_test_score"])
# scores are in the order of param_grid iteration, which is alphabetical
mean_scores = mean_scores.reshape(len(C_OPTIONS), -1, len(N_FEATURES_OPTIONS))
# select score for best C
mean_scores = mean_scores.max(axis=0)
bar_offsets = np.arange(len(N_FEATURES_OPTIONS)) * (len(reducer_labels) + 1) + 0.5

plt.figure()
COLORS = "bgrcmyk"
for i, (label, reducer_scores) in enumerate(zip(reducer_labels, mean_scores)):
    plt.bar(bar_offsets + i, reducer_scores, label=label, color=COLORS[i])

plt.title("Comparing feature reduction techniques")
plt.xlabel("Reduced number of features")
plt.xticks(bar_offsets + len(reducer_labels) / 2, N_FEATURES_OPTIONS)
plt.ylabel("Enhancer-promoter classification F1 score")
plt.ylim((0, 1))
plt.legend(loc="upper left")
plt.savefig('feature_select_compare.png')
