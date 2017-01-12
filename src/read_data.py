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


class HeatStrokeDataFiller(object):

    def __init__(self):

        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.excel_file = os.path.join(self.project_dir, "data", "Literature_Data.xlsx")
        self.spreadsheet_name = "Individualized Data"
        self.output_file = os.path.join(self.project_dir, "data", "filled_data.csv")

        self.use_fake_data = False

        self.important_features = pd.Index(['Heat stroke', 'Exertional (1) vs classic (0)', 'Dehydration', 'Strenuous exercise',
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
        # Anything that is a Nan (empty) gets replaced with these
        self.default_map = {'Case': 1, "Source (Paper)": 0, 'Case ID': 1,
            'Geographical location': "None", 'Environmental temperature (C)': 90,
            'Humidity 8am': 0.1, 'Humidity noon': 0.1, 'Humidity 8pm': 0.1, 'Barometric Pressure': 20,
            'Heat Index (HI)': 0, 'Time of day': 12.00, 'Time of year (month)': 6,
            'Age': 30, 'Weight (kg)': 140, 'BMI': 26.5, 'Nationality':"None",
            'Patient temperature': 37, "Sex": "F",
            'Rectal temperature (deg C)': 37, 'Respiratory rate': 16, 'Daily Ingested Water (L)': 3.7,
            'Sweating': 0.5, 'Skin color (flushed/normal=1, pale=0.5, cyatonic=0)': 1,
            'Heart / Pulse rate (b/min)': 80, '(mean) Arterial blood pressure (mmHg)': 120,
            'Systolic BP': 120, 'Diastolic BP': 80, 'Arterial oxygen saturation': 95,
            'Total serum protein (gm/100ml)': 70, 'Serum albumin': 40, 'Blood Glucose (mg/100ml)': 150,
            'Serum sodium (mmol/L)': 140, 'Serum potassium (mEq/L)': 4, 'Serum chloride (mEq/L)': 100,
            'Haemoglobin (g/dl)': 14, 'Hematocrit': 42,  'White blood cell count (/mcL)': 5000,
            'Platelets': 300000,  'Initial treatment': "None", 'Temperature cooled to (C)': 37,}

        # Fields to fill with zero value
        # Anything that isn't a numeric value gets replaces with zero
        self.fields_to_fill_with_zero = {'Heat stroke', 'Complications', 'Exertional (1) vs classic (0)', 'Dehydration', 'Strenuous exercise',
                                         'Died (1) / recovered (0)', 'Time to death (hours)', 'Heat stroke', 'Mental state', 'Complications',
                                         'Exposure to sun', 'Cardiovascular disease history','Sickle Cell Trait (SCT)',
                                         'Skin rash', 'Diarrhea', 'Bronchospasm', 'Decrebrate convulsion', 'Hot/dry skin',
                                         'Cerebellar ataxia', 'Myocardial infarction', 'Hepatic failure','Pulmonary congestion',
                                         'Duration of abnormal temperature'}

        # Fields to fill with the average of others
        # Anything that isn't a float, or int gets replaced with the average of the others
        self.fields_to_fill_with_average = {'AST (U/I)', 'ALT (U/I)', 'CPK (U/I)', "Time of cooling (min)", "Mean cooling time/C (min)"}
        self.temp_fields = ["Environmental temperature (C)", "Patient temperature", "Rectal temperature (deg C)", "Temperature cooled to (C)"]
        self.keyword_map = {"hot": 39, "humid": 1, "hydrated": 5, "low": 0}

    def read_and_filter_data(self):
        """
        This function reads data from an excel file and ompleted missing data
        :return: A pandas data frame containing positive and negative datapoints
        """
        if self.use_fake_data:
            # todo: does this work?
            return HeatStrokeDataFiller.create_fake_test_data()

        self.df = pd.read_excel(self.excel_file, sheetname=self.spreadsheet_name)
        #negative_df = get_negative_data()
        # print(negative_df)
        # df = pd.concat((df, negative_df))
        self.fill_missing()
        self.df.to_csv("~/Desktop/filled.csv")
        self.fix_fields()
        self.df.to_csv("~/Desktop/fixed.csv")

    def fill_missing(self):
        """
        This function fills missing values
        :return:
        """
        df = self.df
        # Filling with default values
        for field in self.default_map:
            if field not in df.columns:
                logger.warning("(%s) missing from data-frame columns" % field)
                continue
            logger.info("Setting missing in \"%s\" to default: %s" % (field, self.default_map[field]))
            default_value = self.default_map[field]

            where = HeatStrokeDataFiller.find_where_missing(df, field, find_nan=True, find_str=False)
            how_many_to_fill = np.sum(where)
            df[field].loc[where] = np.array([default_value] * how_many_to_fill)

        # Filling with Zeros
        for field in self.fields_to_fill_with_zero:
            if field not in df.columns:
                logger.warning("\"%s\" missing from columns" % field)
                continue
            logger.info("Setting missing in \"%s\" to 0" % field)

            where = HeatStrokeDataFiller.find_where_missing(df, field, find_nan=True, find_str=True)
            how_many_to_fill = np.sum(where)
            df[field].loc[where] = np.zeros(how_many_to_fill)

        # Filling in columns with the average from the rest of the column
        for field in self.fields_to_fill_with_average:
            if field not in df.columns:
                logger.warning("\"%s\" missing from data-frame columns" % field)
                continue

            where = HeatStrokeDataFiller.find_where_missing(df, field, find_nan=True, find_str=True)
            data = df[field][np.invert(where)]
            mean = np.mean(data)
            std = np.std(data)
            if mean == np.nan or std == np.nan:
                mean, std = (0, 0)
            logger.info("Setting missing in \"%s\" with: %.3f +/- %.3f" % (field, mean, std))
            how_many_to_fill = np.sum(where)
            df[field].loc[where] = mean + std * np.random.random(how_many_to_fill)

        fields_not_modified = set(df.columns) - set(self.default_map.keys()) - self.fields_to_fill_with_zero - self.fields_to_fill_with_zero
        logger.info("Fields not modified: %s" % fields_not_modified.__str__())
        return df

    def fix_bounded_values(self):
        # fixes values that start with ">" and "<" and turnes them into floats
        for field in self.df.columns:
            for i in range(self.df.shape[0]):
                value = self.df[field][i]
                if type(value) is str:
                    if value.startswith(">") or value.startswith("<"):
                        try:
                            self.df[field].loc[i] = float(value[1:].strip())
                        except:
                            pass

    def fix_temperature_fields(self):
        # Converts temperature fileds to always be celsius
        for temp_field in self.temp_fields:
            where_string = HeatStrokeDataFiller.find_where_string(self.df, temp_field)
            self.df[temp_field][where_string] = 39
            farenheight_valus = self.df[temp_field] > 70
            self.df[temp_field].loc[farenheight_valus] = (self.df[temp_field][farenheight_valus] - 32.0) * 100.0 / (212 - 32)

    def fix_percentage_fields(self):
        self.percentage_fields = {"Humidity 8am", "Humidity noon", "Humidity 8pm"}
        for field in self.percentage_fields:
            for i in range(self.df.shape[0]):
                value = self.df[field][i]
                if value > 1:
                    self.df[field].loc[i] = value / 100

    def fix_keyword_fields(self):
        for field in self.df.columns:
            for i in range(self.df.shape[0]):
                value = self.df[field][i]
                if type(value) is str and value.replace("\"", "") in self.keyword_map:
                    self.df[field].loc[i] = self.keyword_map[value.replace("\"", "")]

    def fix_range_fields(self):
        for field in self.df.columns:
            for i in range(self.df.shape[0]):
                value = self.df[field][i]
                if type(value) is str and "-" in value:
                    try:
                        value = np.mean(map(float, value.split("-")))
                        self.df[field].loc[i] = value
                    except:
                        pass

    def fix_fields(self):
        # Fixed values that are bad
        males = self.df["Sex"] == "M"
        self.df["Sex"] = np.array(males, dtype=int)

        self.fix_bounded_values()
        self.fix_range_fields()
        self.fix_keyword_fields()
        self.fix_temperature_fields()
        self.df.to_csv("~/Desktop/test.csv")
        self.fix_percentage_fields()


    def trim(self):
        """
        Returns the same data frame but only with the specified columns
        :param df: A pandas data frame
        :return: Another pandas dataframe but only with the important columns
        """
        return self.df[self.important_features]

    @staticmethod
    def find_where_missing(df, field, find_nan=True, find_str=True):
        # Finding all the places that are strings or Nan
        where = np.zeros(df.shape[0], dtype=bool)
        for i in range(df.shape[0]):
            if find_str and type(df[field][i]) is str:
                    where[i] = True
            if find_nan and type(df[field][i]) is not str:
                try:
                    where[i] |= np.isnan(df[field][i])
                except TypeError:
                    logger.error("Type error (%s) for: %s" % (type(df[field][i]), df[field][i]))
                    pass
        return where

    @staticmethod
    def find_where_string(df, field):
        where = np.zeros(df.shape[0], dtype=bool)
        for i in range(df.shape[0]):
            if type(df[field][i]) is str:
                    where[i] = True
        return where

    def write_data(self):
        # For saving the data frame to file
        self.df.to_csv(self.output_file)

    @staticmethod
    def create_fake_test_data(N=200, num_fts=20):
        """
        This function is for getting FAKE data for testing
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

    @staticmethod
    def get_resting_data(N=30):
        # todo
        pass
    @staticmethod
    def get_exercising_data(N=30):
        # todo
        pass
    @staticmethod
    def get_negative_data():
        # todo
        pass

    def remove_strings(self):
        for field in self.df.columns:
            where = np.array(list(map(type, self.df[field]))) == str
            if any(where):
                if field in self.default_map.keys():
                    self.df[field][where] = self.default_map[field]
                elif field in self.fields_to_fill_with_zero:
                    self.df[field].loc[where] = 0
                elif field in self.fields_to_fill_with_average:
                    data = self.df[field][np.invert(where)]
                    self.df[field].loc[where] = np.mean(data) + np.std(data) * np.random.random(len(data))


def main():

    script_description = "This script reads and fills in data from files"
    parser = argparse.ArgumentParser(description=script_description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    input_group = parser.add_argument_group("Inputs")
    input_group.add_argument('-in', '--input', required=False,  help='Input Excel file')

    output_group = parser.add_argument_group("Outputs")
    output_group.add_argument('-out', '--output', required=False, help='Output filtered CSV file')

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


    data_filter = HeatStrokeDataFiller()
    if args.input is not None:
        data_filter.excel_file = args.input
    if args.output is not None:
        data_filter.output_file = args.output

    data_filter.read_and_filter_data()
    data_filter.write_data()

if __name__ == "__main__":
    main()