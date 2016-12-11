#!/usr/bin/env python
"""
cross_validaiton.py

This script is for doing cross validation on logistic regression and other prediction
models with data
"""

import argparse
import logging
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from itertools import cycle
from scipy import interp
import read_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-in', '--excel_file', help='Input excel file with data')
    parser.add_argument('-sheet', '--sheetname', default='Individualized Data', help="Name of sheet in excel file")
    args = parser.parse_args()

    logging.basicConfig(format='[%(funcName)s] - %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    excel_file = args.excel_file
    if args.excel_file is None:
        excel_file = "/Users/jonpdeaton/Google Drive/school/BIOE 141A/Heat_Stroke_Prediction/Literature_Data.xlsx"

    df = pd.read_excel(excel_file, sheetname=args.sheetname)
    read_data.complete_missing_data(df)
    neg_df = read_data.get_negative_data()

    # For testing
    #df = read_data.get_test_data()

    classifier = linear_model.LogisticRegression(C=1e5)
    X = np.array(df.drop('outcome', axis=1))
    y = np.array(df.outcome)
    scores = cross_val_score(classifier, X, y, cv=5)
    print(scores)

    logger.info("Cross Validating...")
    predicted = cross_val_predict(classifier, X, y, cv=5)
    print("Prediction Accuracy: %f" % metrics.accuracy_score(df.outcome, predicted))
    print("Scoring accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
    lw = 2

    N_fold = 6
    fold = 0
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    cv = StratifiedKFold(n_splits=N_fold, shuffle=True)
    for (train, test), color in zip(cv.split(X, y), colors):
        probas = classifier.fit(X[train], y[train]).predict_proba(X[test])
        fpr, tpr, thresholds = roc_curve(y[test], probas[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=lw, color=color, label='ROC fold %d (area = %0.2f)' % (fold, roc_auc))
        fold += 1

    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k', label='Chance')
    mean_tpr /= cv.get_n_splits(X, y)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--', label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == '__main__':
    main()