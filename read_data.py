#!/usr/bin/env python
"""
read_data.py

This is a module used for reading and completing data
"""

import string
import numpy as np
import pandas as pd

def get_filtered_data():
    """
    This function reads data from an excel file and ompleted missing data
    :return: A pandas data frame containing positive and negative datapoints
    """
    excel_file = "/Users/jonpdeaton/Google Drive/school/BIOE 141A/Heat_Stroke_Prediction/Literature_Data.xlsx"
    df = pd.read_excel(excel_file)
    negative_df = get_negative_data()
    df = pd.concat((df, negative_df))
    complete_missing_data(df)
    return df

def get_test_data(N=200, num_fts=20):
    """
    This function is for getting fake data for testing
    :param N: Number of positive and negative data poitns
    :param num_fts: Number of features
    :return: a pandas dataframe with the fake data
    """
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


if __name__ == "__main__":
    pass