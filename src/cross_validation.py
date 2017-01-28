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
import sklearn.model_selection
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from itertools import cycle
from scipy import interp
import read_data
import warnings
import os
import copy

logging.basicConfig(format='[%(funcName)s] - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class CrossValidator(object):

    def __init__(self):
        self.outcome_field = "Heat stroke"
        self.output_directory = os.path.dirname(os.path.abspath(__file__))
        self.plot_filename = "performancs_curves.svg"
        self.N_fold = 6
        self.df = None
        self.cv = StratifiedKFold(n_splits=self.N_fold, shuffle=True)
        self.roc_colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
        self.use_all_fields = False
        self.fields_used = ['Patient temperature', 'Heat Index (HI)', 'Relative Humidity', 'Environmental temperature (C)']

        self.classifier = linear_model.LogisticRegression(C=1e5)

    def CV_overall(self)

    def CV_sensitivity_specificity(self):



    def CV_precision_recall(self):
        # Precision Recall Curve
            precision[which_fold], recall[which_fold], threaholds = precision_recall_curve(y[test], probas[:, 1])
            average_precision[which_fold] = average_precision_score(y_test[:, i], y_score[:, i])

            axarr[1].plot(precision, rcall, lw=lw, color=color, label="Fold: %d" % which_fold)

        # setup plot details
        colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
        lw = 2
        # Binarize the output
        y = label_binarize(y, classes=[0, 1, 2])
        n_classes = y.shape[1]
        precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(), y_score.ravel())
        average_precision["micro"] = average_precision_score(y_test, y_score, average="micro")



        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=random_state)

        precision = dict()
        recall = dict()
        average_precision = dict()



    def perform_cross_validation(self):
        """
        This function performs cross validation on the this instances data frame
        :return: None
        """

        
        
        y = np.array(self.df[self.outcome_field])
        X = self.df.drop(self.outcome_field, axis=1)
        if not self.use_all_fields:
            X  = X[self.fields_used]
        X = np.array(X)

        logger.info("Cross Validating...")
        scores = sklearn.model_selection.cross_val_score(classifier, X, y, cv=self.cv)
        predicted = sklearn.model_selection.cross_val_predict(classifier, X, y, cv=self.cv)

        print("Scores: %s" % scores)
        print("Prediction Accuracy: %f" % metrics.accuracy_score(self.df[self.outcome_field], predicted))
        print("Scoring accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

        logger.info("Cross validating for ROC curves...")
        lw = 2
        which_fold = 0
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)

        f, axarr = plt.subplots(2, sharey=True)
        axarr[0].set_title('Perfromance')

        for (train, test), color in zip(self.cv.split(X, y), self.roc_colors):
            fitted = classifier.fit(X[train], y[train])
            probas = fitted.predict_proba(X[test])
            
            # ROC Curve
            fpr, tpr, thresholds = roc_curve(y[test], probas[:, 1])
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=lw, color=color, label='ROC fold %d (area = %0.2f)' % (which_fold, roc_auc))

            which_fold += 1



        plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k', label='Chance')
        mean_tpr /= self.cv.get_n_splits(X, y)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--', label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.grid(True)
        hfont = {'fontname':'Helvetica'}
        plt.xlabel('False Positive Rate', fontsize=12, **hfont)
        plt.ylabel('True Positive Rate', fontsize=12, **hfont)
        plt.title('%d-Fold Cross Validation ROC' % self.N_fold, fontsize=12, **hfont)
        plt.legend(fontsize=8, loc='lower right', title="Legend",  fancybox=True)
        try:
            output_file = os.path.join(self.output_directory, self.roc_filename)
        except:
            output_file = self.roc_filename
        plt.savefig(output_file)
        logger.info("ROC saved: %s", os.path.basename(output_file))

def main():

    script_description = "This script performs cross validation on heat stroke prediction algorithms"
    parser = argparse.ArgumentParser(description=script_description)

    input_group = parser.add_argument_group("Inputs")
    input_group.add_argument('-in', '--input', help='Input spreadsheet with case data')
    input_group.add_argument('-sheet', '--sheetname', default='Individualized Data', help="Name of sheet in excel file")

    output_group = parser.add_argument_group("Outputs")
    output_group.add_argument("-out", "--output", help="Output")

    options_group = parser.add_argument_group("Opitons")
    options_group.add_argument("-f", "--fake", action="store_true", help="Use fake data")
    options_group.add_argument('-p', '--prefiltered', action="store_true", help="Use pre-filtered data")
    options_group.add_argument('-all', "--all-fields", dest="all_fields", action="store_true", help="Use all fields")

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

    if args.prefiltered:
        logger.info("Using prefiltered data from: %s" % os.path.basename(reader.filled_output_file))
        reader.read_prefiltered_data()
    else:
        reader.read_and_filter_data()
    logger.info("Read data into DataFrame.")

    if not reader.use_fake_data and not args.prefiltered:
        logger.info("Saving data to file: %s" % os.path.basename(reader.filled_output_file))
        reader.write_data()

    cross_validator = CrossValidator()
    cross_validator.df = copy.deepcopy(reader.df)
    cross_validator.use_all_fields = args.all_fields

    cross_validator.perform_cross_validation()

if __name__ == '__main__':
    main()