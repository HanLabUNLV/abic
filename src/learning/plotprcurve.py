
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, confusion_matrix, f1_score


def prcurve_from_file(pr_filename, confusion_filename, y_real, y_proba, y_pred, colorname):
    prcurve = pd.read_csv(pr_filename, sep='\t', index_col=None)
    precision = prcurve['precision']
    recall = prcurve['recall']
    confusion = pd.read_csv(confusion_filename, sep='\t', index_col=None)
     
    # Plotting each individual PR Curve
    plt.plot(recall, precision, lw=1, alpha=0.3, color=colorname,
             #label='PR fold %d (AUC = %0.2f)' % (i, average_precision_score(confusion['Significant'], confusion['y_prob']))
            )
    
    y_real.append(confusion['Significant'])
    y_proba.append(confusion['y_prob'])
    y_pred.append(confusion['y_pred'])


def ABC_predict(inputfile, threshold=0.022, inputindex=None):
    ABC_pd = pd.read_csv(inputfile, sep="\t", index_col=0)
    if inputindex is not None:
      ABC_index = pd.read_csv(inputindex, sep="\t", index_col=0).index
      ABC_pd = ABC_pd.filter(items=ABC_index, axis=0)

    ABC_score = ABC_pd['ABC.Score'].reset_index(drop=True)
    distance =  1/np.log((ABC_pd['distance'])+1.1).reset_index(drop=True)
    y = ABC_pd['Significant'].astype(int).reset_index(drop=True)
    ABC_test = pd.concat([y, distance, ABC_score], axis=1)

    ABC_test['y_pred'] = ABC_test['ABC.Score'] > threshold
    ABC_test['y_pred'] = ABC_test['y_pred'].astype(int)
    return ABC_test



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--traindir', action='append',  help="directory containing  pr_curve and confusion from test")
  parser.add_argument('--testdir', action='append',  help="directory containing  pr_curve and confusion from test")
  parser.add_argument('--studyname', action='append',  help="prefix")
  parser.add_argument('--testname',  help="test data label")
  args=parser.parse_args()

  train_dir_list = args.traindir
  test_dir_list = args.testdir
  studyname_list = args.studyname
  testname = args.testname

  plt.rcParams.update({'font.size': 14})

  # plot the train outer CV performance 
  if train_dir_list:
    fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    colors = ['red', 'blue']
    models = ['full', 'noTF']
    
    for train_idx in range(len(train_dir_list)):
      train_dir = train_dir_list[train_idx]
      studyname = studyname_list[train_idx]
      color = colors[train_idx]
      model = models[train_idx]


      i = 0
      y_real_cv = []
      y_proba_cv = []
      y_pred_cv = []

      pr_cv = [
              train_dir+"/"+studyname+".pr_curve.xgb.0.txt",
              train_dir+"/"+studyname+".pr_curve.xgb.1.txt",
              train_dir+"/"+studyname+".pr_curve.xgb.2.txt",
              train_dir+"/"+studyname+".pr_curve.xgb.3.txt", 
              ]
      confusion_cv = [
          train_dir+"/"+studyname+".confusion.xgb.0.txt",
          train_dir+"/"+studyname+".confusion.xgb.1.txt",
          train_dir+"/"+studyname+".confusion.xgb.2.txt",
          train_dir+"/"+studyname+".confusion.xgb.3.txt",
          ]
      train_inputfile  = train_dir+"/"+studyname+".learninginput.txt"
      Xfeatures_cv = [
          train_dir+"/"+studyname+".Xtest.0.txt",
          train_dir+"/"+studyname+".Xtest.1.txt",
          train_dir+"/"+studyname+".Xtest.2.txt",
          train_dir+"/"+studyname+".Xtest.3.txt",
      ]
   
      for i in range(4):
          prcurve_from_file(pr_cv[i], confusion_cv[i], y_real_cv, y_proba_cv, y_pred_cv, color)
          i += 1

      y_real_cv = np.concatenate(y_real_cv)
      y_proba_cv = np.concatenate(y_proba_cv)
      y_pred_cv = np.concatenate(y_pred_cv)
      precision, recall, _ = precision_recall_curve(y_real_cv, y_proba_cv)
      AUCPR=auc(recall, precision)
      plt.plot(recall, precision, color=color,
               label=r'XGB %s (AUC = %0.2f)' % (model, average_precision_score(y_real_cv, y_proba_cv)),
               #label=r'Test(outer fold CV) (AUC = %0.2f)' % (AUCPR),
               lw=2, alpha=.8)
      print('Best XGB %s (CV) confusion matrix:' % (model))
      confmat = confusion_matrix(y_real_cv,y_pred_cv)
      print(confmat)

    ABC_cv = pd.DataFrame()
    for i in range(4):
        ABC_fold = ABC_predict(train_inputfile, inputindex=Xfeatures_cv[i])
        ABC_cv = pd.concat([ABC_cv, ABC_fold]) 

    ABC_cv = ABC_cv[['Significant','y_pred', 'ABC.Score', 'distance']]
    ABC_cv.to_csv(train_dir_list[0]+'/ABC.gasperini.outerCV.default.confusion.txt', sep='\t')

    precision, recall, thresholds = precision_recall_curve(ABC_cv['Significant'], ABC_cv['ABC.Score'])
    AUCPR=auc(recall, precision)
    # convert to f score
    fscore = (2 * precision * recall) / (precision + recall)
    where_nans = np.isnan(fscore)
    fscore[where_nans] = 0
    # locate the index of the largest f score
    ix = np.argmax(fscore)
    print('Best ABC (CV) Threshold=%.3f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))
    ABC_cv = pd.DataFrame()
    for i in range(4):
        ABC_fold = ABC_predict(train_inputfile, threshold=thresholds[ix], inputindex=Xfeatures_cv[i])
        ABC_cv = pd.concat([ABC_cv, ABC_fold]) 
    ABC_cv = ABC_cv[['Significant','y_pred', 'ABC.Score', 'distance']]
    ABC_cv.to_csv(train_dir_list[0]+'/ABC.gasperini.outerCV.best.confusion.txt', sep='\t')
    confmat = confusion_matrix(ABC_cv['Significant'],ABC_cv['y_pred'])
    print(confmat)

    recallmin = recall-0.70
    ix2 = np.where(recallmin >= 0, recallmin, np.inf).argmin()
    print('0.70 recall ABC (CV) Threshold=%.3f, F-Score=%.3f' % (thresholds[ix2], fscore[ix2]))
    ABC_cv = pd.DataFrame()
    for i in range(4):
        ABC_fold = ABC_predict(train_inputfile, threshold=thresholds[ix2], inputindex=Xfeatures_cv[i])
        ABC_cv = pd.concat([ABC_cv, ABC_fold]) 
    ABC_cv = ABC_cv[['Significant','y_pred', 'ABC.Score', 'distance']]
    ABC_cv.to_csv(train_dir_list[0]+'/ABC.gasperini.outerCV.0.7recall.confusion.txt', sep='\t')
    confmat = confusion_matrix(ABC_cv['Significant'],ABC_cv['y_pred'])
    print(confmat)


    ABC_cv = ABC_cv.dropna()
    plt.plot(recall, precision, color='green',
             label=r'ABC (AUC = %0.2f)' % (average_precision_score(ABC_cv['Significant'], ABC_cv['ABC.Score'])),
             #label=r'ABC_score (AUC = %0.2f)' % (AUCPR),
             lw=2, alpha=.8)
    plt.scatter(recall[ix], precision[ix], marker='o', color='black', label='Best ABC Threshold=%.3f' % (thresholds[ix]))
    plt.scatter(recall[ix2], precision[ix2], marker='o', color='green', label='0.7 recall ABC Threshold=%.3f' % (thresholds[ix2]))

    precision, recall, thresholds = precision_recall_curve(ABC_cv['Significant'], ABC_cv['distance'])
    AUCPR=auc(recall, precision)
    plt.plot(recall, precision, color='black',
             label=r'distance (AUC = %0.2f)' % (average_precision_score(ABC_cv['Significant'], ABC_cv['distance'])),
             #label=r'distance (AUC = %0.2f)' % (AUCPR),
             lw=2, alpha=.8)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR curve outer CV')
    plt.legend(loc="upper right")
    plt.show()

    plt.savefig(train_dir_list[0]+'/prcurve.outerCV.pdf')
    plt.close()





  # now let's plot the test set performance 
  if test_dir_list:
    fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    colors = ['red', 'blue']
    models = ['full', 'noTF']
    
    for test_idx in range(len(test_dir_list)):
      test_dir = test_dir_list[test_idx]
      studyname = studyname_list[test_idx]
      color = colors[test_idx]
      model = models[test_idx]

      y_real_test = []
      y_proba_test = []
      y_pred_test = []
      f1 = []

      pr_test = [
              test_dir+"/pr_curve."+studyname+".save.0.txt",
              test_dir+"/pr_curve."+studyname+".save.1.txt",
              test_dir+"/pr_curve."+studyname+".save.2.txt",
              test_dir+"/pr_curve."+studyname+".save.3.txt", 
              ]
      confusion_test = [
          test_dir+"/confusion."+studyname+".save.0.txt",
          test_dir+"/confusion."+studyname+".save.1.txt",
          test_dir+"/confusion."+studyname+".save.2.txt",
          test_dir+"/confusion."+studyname+".save.3.txt",
          ]

      test_inputfile  = test_dir+"/applyinput.txt"
      Xfeatures_test = test_dir+"/Xfeatures."+studyname+".save.0.txt",

      for i in range(4):
          prcurve_from_file(pr_test[i], confusion_test[i], y_real_test, y_proba_test, y_pred_test, color)
          f1.append( f1_score(y_real_test[i], y_pred_test[i]))
          i += 1

      y_real_test_concat = np.concatenate(y_real_test)
      y_proba_test_concat = np.concatenate(y_proba_test)
      y_pred_test_concat = np.concatenate(y_pred_test)
      xgb_best_model_idx = np.argmax(f1)

      precision, recall, _ = precision_recall_curve(y_real_test_concat, y_proba_test_concat)
      AUCPR=auc(recall, precision)
      avgPrecision = average_precision_score(y_real_test_concat, y_proba_test_concat)
      plt.plot(recall, precision, color=color,
               label=r'XGB %s (AUC = %0.2f)' % (model, avgPrecision),
               #label=r'Test: %s (AUC = %0.2f)' % (AUCPR),
               lw=2, alpha=.8)
      print('Best XGB %s (test) confusion matrix:' % (model))
      confmat = confusion_matrix(y_real_test[xgb_best_model_idx],y_pred_test[xgb_best_model_idx])
      print(confmat)


    ABC_test = ABC_predict(test_inputfile)
    ABC_test = ABC_test[['Significant','y_pred', 'ABC.Score', 'distance']]
    ABC_test.to_csv(test_dir_list[0]+'/ABC.test.'+testname+'.default.confusion.txt', sep='\t')

    precision, recall, thresholds = precision_recall_curve(ABC_test['Significant'], ABC_test['ABC.Score'])
    AUCPR=auc(recall, precision)
    avgPrecision = average_precision_score(ABC_test['Significant'], ABC_test['ABC.Score'])
    # convert to f score
    fscore = (2 * precision * recall) / (precision + recall)
    # locate the index of the largest f score
    ix = np.argmax(fscore)
    print('Best ABC (test) Threshold=%.3f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))
    ABC_test = ABC_predict(test_inputfile, threshold=thresholds[ix])
    ABC_test = ABC_test[['Significant','y_pred', 'ABC.Score', 'distance']]
    ABC_test.to_csv(test_dir_list[0]+'/ABC.test.'+testname+'.best.confusion.txt', sep='\t')
    confmat = confusion_matrix(ABC_test['Significant'],ABC_test['y_pred'])
    print(confmat)

    recallmin = recall-0.70
    ix2 = np.where(recallmin >= 0, recallmin, np.inf).argmin()
    print('0.70 recall ABC (test) Threshold=%.3f, F-Score=%.3f' % (thresholds[ix2], fscore[ix2]))
    ABC_test = ABC_predict(test_inputfile, threshold=thresholds[ix2])
    ABC_test = ABC_test[['Significant','y_pred', 'ABC.Score', 'distance']]
    ABC_test.to_csv(test_dir_list[0]+'/ABC.test.'+testname+'.0.7recall.confusion.txt', sep='\t')
    confmat = confusion_matrix(ABC_test['Significant'],ABC_test['y_pred'])
    print(confmat)

    ABC_test = ABC_test.dropna()
    plt.plot(recall, precision, color='green',
             label=r'ABC (AUC = %0.2f)' % (avgPrecision),
             #label=r'ABC_score (AUC = %0.2f)' % (AUCPR),
             lw=2, alpha=.8)
    plt.scatter(recall[ix], precision[ix], marker='o', color='black', label='Best ABC Threshold=%.3f' % (thresholds[ix]))
    plt.scatter(recall[ix2], precision[ix2], marker='o', color='green', label='0.70 recall ABC Threshold=%.3f' % (thresholds[ix2]))

    precision, recall, thresholds = precision_recall_curve(ABC_test['Significant'], ABC_test['distance'])
    AUCPR=auc(recall, precision)
    avgPrecision = average_precision_score(ABC_test['Significant'], ABC_test['distance'])
    plt.plot(recall, precision, color='black',
             label=r'distance (AUC = %0.2f)' % (avgPrecision),
             #label=r'distance (AUC = %0.2f)' % (AUCPR),
             lw=2, alpha=.8)


    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(r'PR curve %s' % (testname))
    plt.legend(loc="upper right")
    plt.show()

    plt.savefig(test_dir_list[0]+'/prcurve.test.'+testname+'.pdf')
    plt.close()





