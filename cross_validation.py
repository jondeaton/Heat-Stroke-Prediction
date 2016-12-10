#!/usr/bin/env python

import string
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

def get_fake_data(N=100, num_fts=20):
    positive_data = np.ones((N, 1 + num_fts))
    negative_data = np.zeros((N, 1 + num_fts))
    for i in range(1, 1 + num_fts):
        positive_data[:, i] = i + pow(i+1, 2) * np.random.random(N)
        negative_data[:, i] = 1.2 * (i + pow(i+1, 2) * np.random.random(N))
    columns = ['outcome'] + list(string.ascii_lowercase[:num_fts])
    data = np.vstack((positive_data, negative_data))
    df = pd.DataFrame(data, columns=columns)
    return df

def get_negative_data():
    # todo
    pass

def complete_missing_data(df):
    pass



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
    complete_missing_data(df)
    neg_df = get_negative_data()

    # For testing
    df = get_fake_data()

    predictor = linear_model.LogisticRegression(C=1e5)
    predictor.fit(df.drop('outcome', axis=1), df.outcome)


    scores = cross_val_score(predictor, df.drop('outcome', axis=1), df.outcome, cv=5)
    print(scores)

    logger.info("Cross Validating...")
    predicted = cross_val_predict(predictor, df.drop('outcome', axis=1), df.outcome, cv=5)
    print("Prediction Accuracy: %f" % metrics.accuracy_score(df.outcome, predicted))
    print("Scoring accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    roc_auc = auc(fpr, tpr)


    plt.plot(fpr, tpr, lw=2, label='ROC Curve(area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='Chance')
    plt.show()

if __name__ == '__main__':
    main()