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
from sklearn.model_selection import StratifiedKFold
from itertools import cycle
from scipy import interp

def get_test_data(N=200, num_fts=20):
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
    cols = pd.Index(['Case', 'Source (Paper)', 'Case ID', 'Died / recovered',
                     'Time to death (hours)', 'Heat stroke', 'Mental state', 'Complications',
                     'Eertional (1) vs classic (0)', 'Dehydration', 'Strenuous exercise',
                     'Geographical location', 'Environmental temperature (C)',
                     'Humidity 8am', 'Humidity noon', 'Humidity 8pm', 'Barometric Pressure',
                     'Heat Index (HI)', 'Time of day', 'Time of year (month)',
                     'Exposure to sun', 'Sex', 'Age', 'Weight', 'BMI', 'Nationality',
                     'Cardiovascular disease history', 'Sickle Cell Trait (SCT)',
                     'Duration of abnormal temperature', 'Patient temperature',
                     'Rectal temperature', 'Respiratory rate', 'Daily Ingested Water (L)',
                     'Sweating', 'Skin rash',
                     'Skin color (flushed/normal=1, pale=0.5, cyatonic=0)', 'Hot/dry skin',
                     'Heart / Pulse rate (b/min)', '(mean) Arterial blood pressure (mmHg)',
                     'Systolic BP', 'Diastolic BP', 'Arterial oxygen saturation',
                     'Total serum protein (gm/100ml)', 'Serum albumin',
                     'Non-protein nitrogen', 'Blood Glucose (mg/100ml)', 'Serum sodium',
                     'Serum potassium', 'Serum chloride', 'femoral arterial oxygen content',
                     'femoral arterial CO2 content', 'femoral venous oxygen content',
                     'femoral venous CO2 content', 'Hemoglobin', 'Red blood cell count',
                     'Hematocrit', 'Hyperpotassemia', 'White blood cell count',
                     'Alkaline phosphatase', 'Platelets', 'Diarrhea', 'Bronchospasm', 'AST',
                     'ALT', 'CPK', 'Pulmonary congestion', 'Muscle tone',
                     'Initial treatment', 'Time of cooling'], dtype='object')

    defaults = {'Case': 1, "Source (Paper)": "None", 'Case ID': 1, 'Died / recovered': 0,
                'Time to death (hours)': 0, 'Heat stroke': 0, 'Mental state':0 , 'Complications': 0,
                    'Eertional (1) vs classic (0)': 0, 'Dehydration':0 , 'Strenuous exercise':0,
                     'Geographical location': "None", 'Environmental temperature (C)': 90,
                     'Humidity 8am': 0.1, 'Humidity noon':0.1, 'Humidity 8pm':0.1, 'Barometric Pressure':24,
                     'Heat Index (HI)': 0, 'Time of day', 'Time of year (month)',
                     'Exposure to sun', 'Sex', 'Age', 'Weight', 'BMI', 'Nationality',
                     'Cardiovascular disease history', 'Sickle Cell Trait (SCT)',
                     'Duration of abnormal temperature', 'Patient temperature',
                     'Rectal temperature', 'Respiratory rate', 'Daily Ingested Water (L)',
                     'Sweating', 'Skin rash',
                     'Skin color (flushed/normal=1, pale=0.5, cyatonic=0)', 'Hot/dry skin',
                     'Heart / Pulse rate (b/min)', '(mean) Arterial blood pressure (mmHg)',
                     'Systolic BP', 'Diastolic BP', 'Arterial oxygen saturation',
                     'Total serum protein (gm/100ml)', 'Serum albumin',
                     'Non-protein nitrogen', 'Blood Glucose (mg/100ml)', 'Serum sodium',
                     'Serum potassium', 'Serum chloride', 'femoral arterial oxygen content',
                     'femoral arterial CO2 content', 'femoral venous oxygen content',
                     'femoral venous CO2 content', 'Hemoglobin', 'Red blood cell count',
                     'Hematocrit', 'Hyperpotassemia', 'White blood cell count',
                     'Alkaline phosphatase', 'Platelets', 'Diarrhea', 'Bronchospasm', 'AST',
                     'ALT', 'CPK', 'Pulmonary congestion', 'Muscle tone',
                     'Initial treatment', 'Time of cooling'}




def trim(df):
    features = pd.Index(['Heat stroke', 'Eertional (1) vs classic (0)', 'Dehydration', 'Strenuous exercise'
                        , 'Environmental temperature (C)', 'Humidity 8am', 'Humidity noon', 'Humidity 8pm', 'Barometric Pressure',
                         'Heat Index (HI)', 'Time of day', 'Time of year (month)',
                         'Exposure to sun', 'Sex', 'Age', 'Weight', 'BMI', 'Nationality',
                         'Cardiovascular disease history', 'Sickle Cell Trait (SCT)',
                         'Duration of abnormal temperature', 'Patient temperature',
                         'Rectal temperature', 'Respiratory rate', 'Daily Ingested Water (L)',
                         'Sweating', 'Skin rash',
                         'Skin color (flushed/normal=1, pale=0.5, cyatonic=0)', 'Hot/dry skin',
                         'Heart / Pulse rate (b/min)', '(mean) Arterial blood pressure (mmHg)',
                         'Systolic BP', 'Diastolic BP', 'Arterial oxygen saturation',
                         'Total serum protein (gm/100ml)', 'Serum albumin',
                         'Non-protein nitrogen', 'Blood Glucose (mg/100ml)', 'Serum sodium',
                         'Serum potassium', 'Serum chloride', 'femoral arterial oxygen content'],
                        dtype='object')
    return df[features]

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
    #df = get_test_data()


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