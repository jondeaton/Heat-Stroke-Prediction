#!/usr/bin/env python
"""
read_data.py

This is a module used for reading and completing data
"""

import os
import argparse
import string
import logging
import numpy as np
import pandas as pd
import warnings


logging.basicConfig(format='[%(levelname)s][%(funcName)s] - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

features = pd.Index(['Heat stroke', 'Exertional (1) vs classic (0)', 'Dehydration', 'Strenuous exercise',
                     'Environmental temperature (C)', 'Humidity 8am', 'Humidity noon', 'Humidity 8pm',
                        'Barometric Pressure', 'Heat Index (HI)', 'Time of day', 'Time of year (month)',
                         'Exposure to sun', 'Sex', 'Age', 'Weight (kg)', 'BMI', 'Nationality',
                         'Cardiovascular disease history', 'Sickle Cell Trait (SCT)',
                         'Duration of abnormal temperature', 'Patient temperature',
                         'Rectal temperature (deg C)', 'Respiratory rate', 'Daily Ingested Water (L)',
                         'Sweating', 'Skin rash',
                         'Skin color (flushed/normal=1, pale=0.5, cyatonic=0)', 'Hot/dry skin',
                         'Heart / Pulse rate (b/min)', '(mean) Arterial blood pressure (mmHg)',
                         'Systolic BP', 'Diastolic BP', 'Arterial oxygen saturation',
                         'Total serum protein (gm/100ml)', 'Serum albumin',
                         'Non-protein nitrogen', 'Blood Glucose (mg/100ml)', 'Serum sodium',
                         'Serum potassium', 'Serum chloride', 'femoral arterial oxygen content'],
                        dtype='object')

# Physiologically Normal Ranges
defaults = {'Case': 1, "Source (Paper)":0, 'Case ID': 1,
            'Geographical location': "None", 'Environmental temperature (C)': 90,
            'Humidity 8am': 0.1, 'Humidity noon': 0.1, 'Humidity 8pm': 0.1, 'Barometric Pressure': 20,
            'Heat Index (HI)': 0, 'Time of day': 12.00, 'Time of year (month)': 6,
            'Age': 30, 'Weight (kg)': 140, 'BMI': 26.5, 'Nationality':'None',
            'Patient temperature': 37,
            'Rectal temperature (deg C)': 37, 'Respiratory rate': 16, 'Daily Ingested Water (L)': 3.7,
            'Sweating': 0.5, 'Skin color (flushed/normal=1, pale=0.5, cyatonic=0)': 1,
            'Heart / Pulse rate (b/min)': 80, '(mean) Arterial blood pressure (mmHg)': 120,
            'Systolic BP': 120, 'Diastolic BP': 80, 'Arterial oxygen saturation': 95,
            'Total serum protein (gm/100ml)': 70, 'Serum albumin': 40, 'Blood Glucose (mg/100ml)': 150,
            'Serum sodium (mmol/L)': 140, 'Serum potassium (mEq/L)': 4, 'Serum chloride (mEq/L)': 100,
            'Haemoglobin (g/dl)': 14, 'Hematocrit': 42,  'White blood cell count (/mcL)': 5000,
            'Platelets': 300000,  'Initial treatment': "None", 'Temperature cooled to (C)': 37}

# Fields to fill with zero value
fill_with_zero = ['Heat stroke', 'Complications', 'Exertional (1) vs classic (0)', 'Dehydration', 'Strenuous exercise']
fill_with_zero += ['Died (1) / recovered (0)', 'Time to death (hours)', 'Heat stroke', 'Mental state', 'Complications']
fill_with_zero += ['Sex', 'Exposure to sun', 'Cardiovascular disease history','Sickle Cell Trait (SCT)']
fill_with_zero += ['Skin rash', 'Diarrhea', 'Bronchospasm', 'Decrebrate convulsion', 'Hot/dry skin']
fill_with_zero += ['Cerebellar ataxia', 'Myocardial infarction', 'Hepatic failure','Pulmonary congestion']
fill_with_zero += ['Duration of abnormal temperature']

# Fields to fill with the average of others
fill_with_average = ['AST (U/I)', 'ALT (U/I)', 'CPK (U/I)']

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
excel_file = os.path.join(project_dir, "data", "Literature_Data.xlsx")

def read_and_filter_data(excel_file="Literature_Data.xlsx"):
    """
    This function reads data from an excel file and ompleted missing data
    :return: A pandas data frame containing positive and negative datapoints
    """
    df = pd.read_excel(excel_file, sheetname="Individualized Data")
    #negative_df = get_negative_data()
    # print(negative_df)
    # df = pd.concat((df, negative_df))
    fill_missing(df)
    return df

def write_data(data_frame, output_file="filtered_data.xlsx"):
    data_frame.to_csv(output_file)

def read_test_data(N=200, num_fts=20):
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

def get_resting_data(N=30):
    # todo
    pass

def get_exercising_data(N=30):
    # todo
    pass

def get_negative_data():
    # todo
    pass

def fill_missing(df):
    # Filling with default values
    for field in defaults:
        if field not in df.columns:
            logger.warning("(%s) missing from data-frame columns" % field)
            continue
        logger.info("Setting missing in \"%s\" to default: %s" % (field, defaults[field]))
        default_value = defaults[field]

        where = find_where_missing(df, field, find_nan=True, find_str=False)
        how_many_to_fill = np.sum(where)
        df[field].loc[where] = np.array([default_value] * how_many_to_fill)
        write_data(df, "~/Desktop/test.csv")

    # Filling with Zeros
    for field in fill_with_zero:
        if field not in df.columns:
            logger.warning("\"%s\" missing from columns" % field)
            continue
        logger.info("Setting missing in \"%s\" to 0" % field)

        where = find_where_missing(df, field, find_nan=True, find_str=True)
        how_many_to_fill = np.sum(where)
        df[field].loc[where] = np.zeros(how_many_to_fill)
        write_data(df, "~/Desktop/test.csv")

    # Filling in columns with the average from the rest of the column
    for field in fill_with_average:
        if field not in df.columns:
            logger.warning("\"%s\" missing from data-frame columns" % field)
            continue

        where = find_where_missing(df, field, find_nan=True, find_str=True)
        data = df[field][np.invert(where)]
        mean = np.mean(data)
        std = np.std(data)
        if mean == np.nan or std == np.nan:
            mean, std = (0, 0)
        logger.info("Setting missing in \"%s\" with: %.3f +/- %.3f" % (field, mean, std))
        how_many_to_fill = np.sum(where)
        df[field].loc[where] = mean + std * np.random.random(how_many_to_fill)
        write_data(df, "~/Desktop/test.csv")

    write_data(df, "~/Desktop/test.csv")
    exit()

    fields_not_modified = set(df.columns) - set(defaults.keys()) - set(fill_with_average) - set(fill_with_zero)
    logger.info("Fields not modified: %s" % fields_not_modified.__str__())
    return df


def find_where_missing(df, field, find_nan=True, find_str=True):
    # Finding all the places that are strings or Nan
    where = np.zeros(df.shape[0], dtype=bool)
    for i in range(df.shape[0]):
        if find_str and type(df[field][i]) is str:
                where[i] = True
        elif find_nan:
            try:
                where[i] |= np.isnan(df[field][i])
            except TypeError:
                logger.error("Type error for: %s" % df[field][i])
                pass
    return where


def remove_strings(df):
    for field in df.columns:
        where = np.array(list(map(type, df[field]))) == str
        if any(where):
            if field in defaults.keys():
                df[field][where] = defaults[field]
            elif field in fill_with_zero:
                df[field][where] = 0
            elif field in fill_with_average:
                data = df[field][np.invert(where)]
                df[field][where] = np.mean(data) + np.std(data) * np.random.random(len(data))
    return df

def trim(df):
    return df[features]

def main():

    script_description = "This script reads and fills in data from files"
    parser = argparse.ArgumentParser(description=script_description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    console_options_group = parser.add_argument_group("Console Options")
    console_options_group.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    console_options_group.add_argument('--debug', action='store_true', help='Debug console')
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
        logging.basicConfig(format='[%(asctime)s][%(levelname)s][%(funcName)s] - %(message)s')
    elif args.verbose:
        logger.setLevel(logging.INFO)
        logging.basicConfig(format='[%(asctime)s][%(levelname)s][%(funcName)s] - %(message)s')
    else:
        warnings.filterwarnings('ignore')
        logger.setLevel(logging.WARNING)
        logging.basicConfig(format='[log][%(levelname)s] - %(message)s')



    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    excel_file = os.path.join(project_dir, "data", "Literature_Data.xlsx")
    df = read_and_filter_data(excel_file=excel_file)
    write_data(df, os.path.join(project_dir, "data", "filtered_data.csv"))

if __name__ == "__main__":
    main()