import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc, average_precision_score, precision_recall_curve


def prcurve_from_file(pr_filename, confusion_filename, y_real, y_proba, colorname):
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


if __name__ == "__main__":

    pr_cv = [
            "run.all/all.2pass.pr_curve.xgb.0.txt",
            "run.all/all.2pass.pr_curve.xgb.1.txt",
            "run.all/all.2pass.pr_curve.xgb.2.txt",
            "run.all/all.2pass.pr_curve.xgb.3.txt", 
            ]
    confusion_cv = [
        "run.all/all.2pass.confusion.xgb.0.txt",
        "run.all/all.2pass.confusion.xgb.1.txt",
        "run.all/all.2pass.confusion.xgb.2.txt",
        "run.all/all.2pass.confusion.xgb.3.txt",
        ]
    pr_test = [
            "apply.all/Gasperini//pr_curve.all.2pass.save.0.txt",
            "apply.all/Gasperini//pr_curve.all.2pass.save.1.txt",
            "apply.all/Gasperini//pr_curve.all.2pass.save.2.txt",
            "apply.all/Gasperini//pr_curve.all.2pass.save.3.txt", 
            ]
    confusion_test = [
        "apply.all/Gasperini//confusion.all.2pass.save.0.txt",
        "apply.all/Gasperini//confusion.all.2pass.save.1.txt",
        "apply.all/Gasperini//confusion.all.2pass.save.2.txt",
        "apply.all/Gasperini//confusion.all.2pass.save.3.txt",
        ]

    i = 0
    y_real_cv = []
    y_proba_cv = []
    y_real_test = []
    y_proba_test = []

    fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    for i in range(4):
        prcurve_from_file(pr_cv[i], confusion_cv[i], y_real_cv, y_proba_cv, 'blue')
        i += 1
 
    for i in range(4):
        prcurve_from_file(pr_test[i], confusion_test[i], y_real_test, y_proba_test, 'red')
        i += 1

    y_real_cv = np.concatenate(y_real_cv)
    y_proba_cv = np.concatenate(y_proba_cv)
    precision, recall, _ = precision_recall_curve(y_real_cv, y_proba_cv)
    AUCPR=auc(recall, precision)
    plt.plot(recall, precision, color='blue',
    #         label=r'Test(outer fold CV) (AUC = %0.2f)' % (average_precision_score(y_real_cv, y_proba_cv)),
             label=r'Test(outer fold CV) (AUC = %0.2f)' % (AUCPR),
             lw=2, alpha=.8)

    y_real_test = np.concatenate(y_real_test)
    y_proba_test = np.concatenate(y_proba_test)

    precision, recall, _ = precision_recall_curve(y_real_test, y_proba_test)
    AUCPR=auc(recall, precision)
    plt.plot(recall, precision, color='red',
             #label=r'Test(chr 5,10,15,20) (AUC = %0.2f)' % (average_precision_score(y_real_test, y_proba_test)),
             label=r'Test(chr 5,10,15,20) (AUC = %0.2f)' % (AUCPR),
             lw=2, alpha=.8)



    #ABC_pd = pd.read_csv('data/Gasperini/Gasperini2019.at_scale.ABC.TF.cobinding.txt', sep="\t", index_col=None)
    #ABC_pd = pd.read_csv('data/Gasperini/Gasperini2019.at_scale.ABC.TF.cobinding.erole.grouped.allsig.txt', sep="\t", index_col=None)
    ABC_pd = pd.read_csv('data/Gasperini/Gasperini2019.at_scale.ABC.TF.NMF.erole.grouped.test.txt', sep="\t", index_col=None)
    ABC_score = ABC_pd['ABC.Score'] 
    distance = 1/np.log(ABC_pd['distance'])
    y = ABC_pd['Significant'].astype(int)
    ABC_test = pd.concat([y, distance, ABC_score], axis=1)

    ABC_test['y_pred'] = ABC_test['ABC.Score'] > 0.022
    ABC_test['y_pred'] = ABC_test['y_pred'].astype(int)
    y_res = ABC_test[['Significant','y_pred']]
    y_res.to_csv('ABC.confusion.txt', index=False, sep='\t')

    ABC_test = ABC_test.dropna()
    precision, recall, thresholds = precision_recall_curve(ABC_test['Significant'], ABC_test['ABC.Score'])
    AUCPR=auc(recall, precision)
    plt.plot(recall, precision, color='green',
             #label=r'ABC_score (AUC = %0.2f)' % (average_precision_score(ABC_test['Significant'], ABC_test['ABC.Score'])),
             label=r'ABC_score (AUC = %0.2f)' % (AUCPR),
             lw=2, alpha=.8)

    precision, recall, thresholds = precision_recall_curve(ABC_test['Significant'], ABC_test['distance'])
    AUCPR=auc(recall, precision)
    plt.plot(recall, precision, color='black',
             #label=r'distance (AUC = %0.2f)' % (average_precision_score(ABC_test['Significant'], ABC_test['distance'])),
             label=r'distance (AUC = %0.2f)' % (AUCPR),
             lw=2, alpha=.8)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR curve')
    plt.legend(loc="upper right")
    plt.show()

    plt.savefig('prcurve.pdf')
    plt.close()

