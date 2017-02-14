#!/usr/bin/env python
"""
cross_validaiton.py

This script is for doing cross validation on logistic regression and other prediction
models with data
"""

import os
import copy
import argparse
import logging
import warnings
import coloredlogs
from termcolor import colored

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn.model_selection
from sklearn import metrics
from sklearn import linear_model
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from sklearn import svm

from itertools import cycle
from scipy import interp
from scipy.cluster.vq import whiten

import read_data

coloredlogs.install(level='INFO')
logging.basicConfig(format='[%(levelname)s][%(funcName)s] - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class CrossValidator(object):

    def __init__(self):
        self.outcome_field = "Heat stroke"
        self.output_directory = os.path.dirname(os.path.abspath(__file__))
        self.roc_filename = "roc_curve.svg"
        self.prc_filename = "prc_plot.svg"
        self.margins_filename = "margins.svg"
        self.metrics_filename = "cv_metrics.csv"
        self.roc_colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
        
        self.show_plots = False

        self.N_fold = 6
        self.cv = StratifiedKFold(n_splits=self.N_fold, shuffle=True)
        self.use_svm = False
        self.classifier = None

        self.df = None
        self.X = None
        self.y = None

        self.whiten_data = False

        self.use_all_fields = False
        self.fields_used = ['Patient temperature', 'Heat Index (HI)', 'Relative Humidity', 'Environmental temperature (C)', 'Heart / Pulse rate (b/min)']

        self.all_fields = ['Exertional (1) vs classic (0)', 'Dehydration', 'Strenuous exercise', 'Environmental temperature (C)',
        'Relative Humidity', 'Barometric Pressure', 'Heat Index (HI)', 'Time of day', 'Time of year (month)', 'Exposure to sun', 'Sex', 'Age',
        'BMI', 'Weight (kg)', 'Nationality', 'Cardiovascular disease history', 'Sickle Cell Trait (SCT)', 'Patient temperature', 'Rectal temperature (deg C)', 
        'Daily Ingested Water (L)', 'Sweating', 'Skin color (flushed/normal=1, pale=0.5, cyatonic=0)', 'Hot/dry skin',
        'Heart / Pulse rate (b/min)', 'Systolic BP', 'Diastolic BP']

    def set_classifier(self):
        if self.use_svm:
            self.classifier = svm.SVC(kernel='linear', C=1, probability=True)
        else:
            self.classifier = linear_model.LogisticRegression(C=1e5)

    def make_margins_plot(self, feature1='Environmental temperature (C)', feature2='Relative Humidity'):
        # This plot makes a seperating hyperplane
        if self.use_all_fields:
            feature_list = self.all_fields
        else:
            feature_list = self.fields_used
        
        dim0 = feature_list.index(feature1)
        dim1 = feature_list.index(feature2)

        # Fit the SVM classifier
        self.classifier.fit(self.X[:, [dim0, dim1]], self.y)

        w = self.classifier.coef_[0]
        a = -w[0] / w[1]
        x_min = min(self.X[:, dim0])
        x_max = max(self.X[:, dim0])
        xx = np.linspace(x_min - 0.1 * (x_max - x_min), x_max + 0.1 * (x_max - x_min))
        yy = a * xx - (self.classifier.intercept_[0]) / w[1]
        margin = 1 / np.sqrt(np.sum(self.classifier.coef_ ** 2))
        yy_down = yy + a * margin
        yy_up = yy - a * margin

        plt.figure()
        x_points = self.classifier.support_vectors_[:, 0]
        y_points = self.classifier.support_vectors_[:, 1]

        # Plotting the patient data
        plt.scatter(self.X[:, dim0][self.y == 0], self.X[:, dim1][self.y == 0], c="#1482e2", label="No heat stroke")
        plt.scatter(self.X[:, dim0][self.y == 1], self.X[:, dim1][self.y == 1], c="#d3430a", label="Heat stroke")

        # Plot the SVM hyperplane, the points, and the nearest vectors to the plane
        plt.plot(xx, yy, 'k-', label="SVM Hyperplane")
        plt.plot(xx, yy_down, 'k--')
        plt.plot(xx, yy_up, 'k--')

        y_min = min(self.X[:, dim1])
        y_max = max(self.X[:, dim1])
        plt.xlim([x_min - 0.1 * (x_max - x_min), x_max + 0.1 * (x_max - x_min)])
        plt.ylim([y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min)])
        plt.title("SVM Margins Plot")

        plt.xlabel(feature_list[dim0])
        plt.ylabel(feature_list[dim1])
        plt.legend()
        plt.grid(True)
        plt.savefig(self.margins_filename)
        if self.show_plots:
            plt.show()

    def CV_all(self):

        self.y = np.array(self.df[self.outcome_field])
        self.X = self.df.drop(self.outcome_field, axis=1)
        if self.use_all_fields:
            self.X  = self.X[self.all_fields]
        else:
            try:
                # Using ony a subset of fields
                self.X  = self.X[self.fields_used]
            except KeyError:
                # This error would happen if we were using fake
                # data without the same feature names
                pass

        if self.whiten_data:
            self.X = whiten(self.X)
        else:
            self.X = np.array(self.X)

        self.classifier.fit(self.X, self.y)
        logger.info("Coefficients: %s" % self.classifier.coef_[0])

        if self.use_svm:
            logger.info("Making margins plot...")
            self.make_margins_plot()

        logger.info("Cross Validating...")
        self.CV_metrics()

        logger.info("Cross validating for sensitivity/specificity...")
        self.CV_sensitivity_specificity()

        logger.info("Cross validating for precision/recall...")
        self.CV_precision_recall()

    def CV_sensitivity_specificity(self):
        # ROC curve generation
        lw = 2
        which_fold = 0
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)

        plt.figure()
        for (train, test), color in zip(self.cv.split(self.X, self.y), self.roc_colors):
            fitted = self.classifier.fit(self.X[train], self.y[train])
            probas = fitted.predict_proba(self.X[test])
            # ROC Curve
            fpr, tpr, thresholds = roc_curve(self.y[test], probas[:, 1])
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=lw, color=color, label='ROC fold %d (area = %0.2f)' % (which_fold, roc_auc))
            which_fold += 1

        plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k', label='Chance')
        mean_tpr /= self.cv.get_n_splits(self.X, self.y)
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
        if self.show_plots:
            plt.show()

    def CV_precision_recall(self):
        # Precision Recall Curve
        # X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(self.X, self.y, test_size=.5)

        precision = dict()
        recall = dict()
        average_precision = dict()
        lw = 2
        which_fold = 0
        plt.figure()
        for (train, test), color in zip(self.cv.split(self.X, self.y), self.roc_colors):
        # for which_fold in range(self.N_fold):
            fitted = self.classifier.fit(self.X[train], self.y[train])
            probas = fitted.predict_proba(self.X[test])

            precision[which_fold], recall[which_fold], threaholds = precision_recall_curve(self.y[test], probas[:, 1])
            average_precision[which_fold] = sklearn.metrics.average_precision_score(self.y[test], probas[:, 1])
            plt.plot(precision[which_fold], recall[which_fold], lw=lw, color=color, label="Fold: %d" % which_fold)
            which_fold += 1

        # Setup plot details
        # # Binarize the output
        # precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(), y_score.ravel())
        # average_precision["micro"] = average_precision_score(y_test, y_score, average="micro")

        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.grid(True)
        hfont = {'fontname':'Helvetica'}
        plt.xlabel('Recall', fontsize=12, **hfont)
        plt.ylabel('Precision', fontsize=12, **hfont)
        plt.title('%d-Fold Cross Validation Precision Recall' % self.N_fold, fontsize=12, **hfont)
        plt.legend(fontsize=8, loc='lower left', title="Legend",  fancybox=True)
        plt.savefig(self.prc_filename)
        if self.show_plots:
            plt.show()

    def CV_metrics(self):
        """
        This function performs cross validation on the this instances data frame
        :return: None
        """

        df = pd.DataFrame()

        self.scores = sklearn.model_selection.cross_val_score(self.classifier, self.X, self.y, cv=self.cv)
        y_pred = sklearn.model_selection.cross_val_predict(self.classifier, self.X, self.y, cv=self.cv)
        self.accuracy = metrics.accuracy_score(self.df[self.outcome_field], y_pred)
        self.fscore = metrics.f1_score(self.df[self.outcome_field], y_pred)
        fpr, tpr, thresholds = metrics.roc_curve(self.y, y_pred, pos_label=2)

        FPR = fpr[np.argmin(np.abs(thresholds))]

        y_true = self.df[self.outcome_field]

        df['Accuracy'] = [self.scores.mean()]
        df['F-Score'] = [self.fscore]
        df['AUC'] = metrics.roc_auc_score(y_true, y_pred)
        df['Sensitivity'] = [1 - FPR]
        df['Recall'] = metrics.recall_score(y_true, y_pred)
        df['Precision'] = metrics.precision_score(y_true, y_pred)
        df['Hamming Loss'] = metrics.hamming_loss(y_true, y_pred)
        df['MCC'] = metrics.matthews_corrcoef(y_true, y_pred)

        for field in df.columns:
            logger.info(colored("%s: %f" % (field, df[field]), 'yellow'))

        df['Classifier'] = str(self.classifier)
        df.to_csv(self.metrics_filename)

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
    options_group.add_argument('-N', '--num-negative', dest='num_negative', type=int, default=500, required=False, help="Number of negative data points")
    options_group.add_argument('-svm', '--svm', action="store_true", help="Use Support Vector Machine")
    options_group.add_argument('-show', "--show-plots", dest="show_plots", action="store_true", help="Display plots while running")

    console_options_group = parser.add_argument_group("Console Options")
    console_options_group.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    console_options_group.add_argument('--debug', action='store_true', help='Debug console')

    args = parser.parse_args()
    
    if args.debug:
        logging.basicConfig(format='[%(asctime)s][%(levelname)s][%(funcName)s] - %(message)s')
        coloredlogs.install(level='DEBUG')
    elif args.verbose:
        warnings.filterwarnings('ignore')
        logging.basicConfig(format='[%(asctime)s][%(levelname)s][%(funcName)s] - %(message)s')
        coloredlogs.install(level='INFO')
    else:
        warnings.filterwarnings('ignore')
        logging.basicConfig(format='[log][%(levelname)s] - %(message)s')
        coloredlogs.install(level='WARNING')

    reader = read_data.HeatStrokeDataFiller()
    reader.use_fake_data = args.fake
    reader.num_negative = args.num_negative

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
    cross_validator.show_plots = args.show_plots
    cross_validator.use_svm = args.svm
    cross_validator.set_classifier()
    cross_validator.CV_all()

if __name__ == '__main__':
    main()