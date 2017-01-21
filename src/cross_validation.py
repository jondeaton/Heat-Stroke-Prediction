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
import warnings
import os

logging.basicConfig(format='[%(funcName)s] - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class CrossValidator(object):

    def __init__(self):
        self.outcome_field = "Heat stroke"
        self.output_directory = os.path.dirname(os.path.abspath(__file__))
        self.roc_filename = "roc_curve.svg"
        self.N_fold = 6
        self.df = None
        self.cv = StratifiedKFold(n_splits=self.N_fold, shuffle=True)

    def perform_cross_validation(self):
        """
        This function performs cross validation on the this instances data frame
        :return: None
        """
        classifier = linear_model.LogisticRegression(C=1e5)
        y = np.array(self.df[self.outcome_field])
        X = np.array(self.df.drop(self.outcome_field, axis=1))

        logger.info("Cross Validating...")

        scores = cross_val_score(classifier, X, y, cv=self.cv)
        predicted = cross_val_predict(classifier, X, y, cv=self.cv)

        print("Scores: %s" % scores)
        print("Prediction Accuracy: %f" % metrics.accuracy_score(self.df[self.outcome_field], predicted))
        print("Scoring accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

        colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
        lw = 2

        fold = 0
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)

        for (train, test), color in zip(self.cv.split(X, y), colors):
            probas = classifier.fit(X[train], y[train]).predict_proba(X[test])
            fpr, tpr, thresholds = roc_curve(y[test], probas[:, 1])
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=lw, color=color, label='ROC fold %d (area = %0.2f)' % (fold, roc_auc))
            fold += 1

        plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k', label='Chance')
        mean_tpr /= self.cv.get_n_splits(X, y)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--', label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        try:
            output_file = os.path.join(self.output_directory, self.roc_filename)
        except:
            output_file = self.roc_filename
        plt.savefig(output_file)

def main():

    script_description = "This script performs cross validation on heat stroke prediction algorithms"
    parser = argparse.ArgumentParser(description=script_description)

    input_group = parser.add_argument_group("Inputs")
    input_group.add_argument('-in', '--input', help='Input spreadsheet with case data')
    input_group.add_argument('-sheet', '--sheetname', default='Individualized Data', help="Name of sheet in excel file")

    output_group = parser.add_argument_group("Outputs")
    output_group.add_argument("-out", "--output", help="Output")

    options_group = parser.add_argument_group("Opitons")
    options_group.add_argument("-f", "--fake",action="store_true", help="Use fake data")

    console_options_group = parser.add_argument_group("Console Options")
    console_options_group.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    console_options_group.add_argument('--debug', action='store_true', help='Debug console')

    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
        logging.basicConfig(format='[%(asctime)s][%(levelname)s][%(funcName)s] - %(message)s')
    elif args.verbose:
        warnings.filterwarnings('ignore')
        logger.setLevel(logging.INFO)
        logging.basicConfig(format='[%(asctime)s][%(levelname)s][%(funcName)s] - %(message)s')
    else:
        warnings.filterwarnings('ignore')
        logger.setLevel(logging.WARNING)
        logging.basicConfig(format='[log][%(levelname)s] - %(message)s')

    reader = read_data.HeatStrokeDataFiller()
    reader.use_fake_data = args.fake
    reader.read_and_filter_data()
    logger.info("Read data into DataFrame.")

    cross_validator = CrossValidator()
    cross_validator.df = reader.df

    cross_validator.perform_cross_validation()

if __name__ == '__main__':
    main()