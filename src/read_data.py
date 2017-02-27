#!/usr/bin/env python
"""
read_data.py

This is a script that is used for transforming data gathered from case studies of heat stroke
into a foramt that can be used in a logistic regression model

This script takes as input an excel file that contains data, fills in missing values with
physiologally normal values, makes negative cases using physiologically normal ranges,
and then saves the resulting data to file for later use.
"""

import os
import string
import logging
import numpy as np
import pandas as pd
import warnings
import datetime
import copy
import coloredlogs

coloredlogs.install(level='INFO')
logger = logging.getLogger(__name__)

__author__ = "Jon Deaton"
__email__ = "jdeaton@stanford.edu"

class HeatStrokeDataFiller(object):

    # Physiologically Normal Ranges
    # Anything that is a Nan (empty) gets replaced with these
    default_map = {'Case': 1, "Source (Paper)": 0, 'Case ID': 1,
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
        'Platelets': 300000,  'Initial treatment': "None", 'Temperature cooled to (C)': 37}

    # Fields to fill with zero value
    # Anything that isn't a numeric value gets replaces with zero
    fields_to_fill_with_zero = {'Heat stroke', 'Complications', 'Exertional (1) vs classic (0)', 'Dehydration', 'Strenuous exercise',
                                     'Died (1) / recovered (0)', 'Time to death (hours)', 'Heat stroke', 'Mental state', 'Complications',
                                     'Exposure to sun', 'Cardiovascular disease history','Sickle Cell Trait (SCT)',
                                     'Skin rash', 'Diarrhea', 'Bronchospasm', 'Decrebrate convulsion', 'Hot/dry skin',
                                     'Cerebellar ataxia', 'Myocardial infarction', 'Hepatic failure','Pulmonary congestion',
                                     'Duration of abnormal temperature'}

    # Fields to fill with the average of others
    # Anything that isn't a float, or int gets replaced with the average of the others
    fields_to_fill_with_average = {'AST (U/I)', 'ALT (U/I)', 'CPK (U/I)', "Time of cooling (min)", "Mean cooling time/C (min)"}
    temp_fields = ["Environmental temperature (C)", "Patient temperature", "Rectal temperature (deg C)", "Temperature cooled to (C)"]
    keyword_map = {"hot": 39, "humid": 1, "hydrated": 5, "low": 0}

    # Each negative default has a distribution from which N points are drawn
    # TODO: UPDATE THESE TO BE MORE REALISTIC
    negative_default = {}
    negative_default['Heat stroke'] = lambda N: np.zeros(N)
    negative_default['Exertional (1) vs classic (0)'] = lambda N: np.zeros(N)
    negative_default['Dehydration'] = lambda N: np.zeros(N)
    negative_default['Strenuous exercise'] = lambda N: 0.9 * np.random.random(N)
    negative_default['Environmental temperature (C)'] = lambda N: 30 + 5 * np.random.normal(size=N)
    negative_default['Relative Humidity'] = lambda N: 0.1 + 0.1 * np.random.normal(size=N)
    negative_default['Barometric Pressure'] = lambda N: (29.97 * np.ones(N))
    negative_default['Heat Index (HI)'] = lambda N: 80 + 15 * np.random.normal(size=N)
    negative_default['Time of day'] = lambda N: 9 + 8 * np.random.random(N)
    negative_default['Time of year (month)'] = lambda N: 12 * np.random.random(N)
    negative_default['Exposure to sun'] = lambda N: 0.3 + 0.1 * np.random.normal(size=N)
    negative_default['Sex'] = lambda N: np.random.random(N) > 0.5
    negative_default['Age'] = lambda N: 18 + 40 * np.random.random(N)
    negative_default['BMI'] = lambda N: 18.5 + (23 - 18.5) * np.random.random(N)
    negative_default['Weight (kg)'] = lambda N: 41.2769 + (53.5239 - 41.2769) * np.random.random(N)
    negative_default['Nationality'] = lambda N: np.random.random(N) > 0.5
    negative_default['Cardiovascular disease history'] = lambda N: np.random.random(N) < 0.05
    negative_default['Sickle Cell Trait (SCT)'] = lambda N: np.random.random(N) < 10e-5
    negative_default['Patient temperature'] = lambda N: 36.1 + (37.2 - 36.1) * np.random.normal(size=N)
    negative_default['Rectal temperature (deg C)'] = lambda N: 36.1 + (37.2 - 36.1) * np.random.random(N)
    negative_default['Daily Ingested Water (L)'] = lambda N: 1 + np.random.random(N) * 5
    negative_default['Sweating'] = lambda N: np.random.random(N) < 0.5
    negative_default['Skin color (flushed/normal=1, pale=0.5, cyatonic=0)'] = lambda N: 0.9 * np.random.random(N) * 0.1
    negative_default['Hot/dry skin'] = lambda N: np.random.random(N) < 0.05
    negative_default['Heart / Pulse rate (b/min)'] = lambda N: 120 + 30 * np.random.normal(size=N)
    negative_default['Systolic BP'] = lambda N: 110 + 10 * np.random.random(N)
    negative_default['Diastolic BP'] = lambda N: 80 + 10 * np.random.random(N)

    # Positive default values
    positive_default = copy.copy(negative_default)
    positive_default['Heat stroke'] = lambda N: np.ones(N)
    positive_default['Exertional (1) vs classic (0)'] = lambda N: np.random.random(N) > 0
    positive_default['Patient temperature'] = lambda N: 40.1 + 2 * np.random.normal(size=N)
    positive_default['Rectal temperature (deg C)'] = lambda N: 40.1 + 2 * np.random.normal(size=N)
    positive_default['Heat Index (HI)'] = lambda N: 105 + 5 * np.random.normal(size=N)
    positive_default['Environmental temperature (C)'] = lambda N: 37 + 3 * np.random.normal(size=N)
    positive_default['Relative Humidity'] = lambda N: 0.8 + 0.3 * np.random.normal(size=N)
    positive_default['Strenuous exercise'] = lambda N: np.random.ranodm(N)
    positive_default['Exposure to sun'] = lambda N: 0.5 + 0.2 * np.random.normal(size=N)

    important_features = pd.Index(negative_default.keys())
    negative_data_size = 200

    outcome_field = "Heat stroke"

    def __init__(self):
        """
        Constructor method. This sets up some file locations and instance fields.
        :return: None
        """
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.excel_file = os.path.join(self.project_dir, "data", "Literature_Data.xlsx")
        self.spreadsheet_name = "Individualized Data"
        self.filled_output_file = os.path.join(self.project_dir, "data", "filled_data.csv")
        self.output_file = os.path.join(self.project_dir, "data", "final.csv")
        self.use_fake_data = False # For testing
        # This instance value "self.df" is the pandas DataFrame that contains all of the data
        # from the literature case studies. Manipulating this field is the purpose of this class.
        
        self.num_negative = 500
        self.df = None

    def read_prefiltered_data(self):
        """
        This function reads data from an already saved file
        :return: None
        """
        logger.info("Reading prefiltered data from file: %s..." % os.path.basename(self.filled_output_file))
        self.df = pd.read_csv(self.filled_output_file)

    def read_and_filter_data(self):
        """
        This function reads data from an excel file into a pandas DataFrame, formats it, and adds negatives to it
        :return: None
        """
        if self.use_fake_data:
            self.df = HeatStrokeDataFiller.create_fake_test_data()
            return

        logger.info("Reading and cleaning data from file: %s..." % os.path.basename(self.excel_file))
        self.df = pd.read_excel(self.excel_file, sheetname=self.spreadsheet_name)
        logger.debug("Fixing time fields...")
        self.fix_time_fields()
        logger.debug("Filling missing data...")
        self.fill_missing()
        logger.debug("Fixing fields...")
        self.fix_fields()
        logger.debug("Filtering data features...")
        self.filter_data()
        logger.debug("Generating negative data...")
        self.make_and_append_negative_data()

        logger.debug("Casting to float...")
        self.df = self.df.astype(float)

    def fill_missing(self):
        """
        This function fills the missing values in the pandas DataFrame containing the data
        :return: None
        """
        df = self.df
        # Filling with default values
        logger.debug("Filling from distributions...")
        for field in HeatStrokeDataFiller.default_map or field in HeatStrokeDataFiller.positive_default:
            if field not in df.columns:
                logger.warning("(%s) missing from data-frame columns" % field)
                continue
            logger.debug("Setting missing in \"%s\" to default: %s" % (field, HeatStrokeDataFiller.default_map[field]))
            default_value = HeatStrokeDataFiller.default_map[field]

            where = HeatStrokeDataFiller.find_where_missing(df, field, find_nan=True, find_str=False)
            how_many_to_fill = np.sum(where)
            if field in HeatStrokeDataFiller.positive_default:
                # Use default positive dietributions
                distribution = HeatStrokeDataFiller.positive_default[field]
                df[field].loc[where] = distribution(how_many_to_fill)
            else:
                logger.debug("Using default %s for field: %s" % (default_value, field))
                # Use default values
                df[field].loc[where] = np.array([default_value] * how_many_to_fill)

        # Filling with Zeros
        logger.debug("Fillling with zeros...")
        for field in HeatStrokeDataFiller.fields_to_fill_with_zero:
            if field not in df.columns:
                logger.warning("\"%s\" missing from columns" % field)
                continue
            logger.debug("Setting missing in \"%s\" to 0" % field)

            where = HeatStrokeDataFiller.find_where_missing(df, field, find_nan=True, find_str=True)
            how_many_to_fill = np.sum(where)
            df[field].loc[where] = np.zeros(how_many_to_fill)

        # Filling in columns with the average from the rest of the column
        logger.debug("Filling with agerages...")
        for field in HeatStrokeDataFiller.fields_to_fill_with_average:
            if field not in df.columns:
                logger.warning("\"%s\" missing from data-frame columns" % field)
                continue

            where = HeatStrokeDataFiller.find_where_missing(df, field, find_nan=True, find_str=True)
            data = df[field][np.invert(where)]
            mean = np.mean(data)
            std = np.std(data)
            if mean == np.nan or std == np.nan:
                mean, std = (0, 0)
            logger.debug("Setting missing in \"%s\" with: %.3f +/- %.3f" % (field, mean, std))
            how_many_to_fill = np.sum(where)
            df[field].loc[where] = mean + std * np.random.random(how_many_to_fill)

        fields_not_modified = set(df.columns) - set(HeatStrokeDataFiller.default_map.keys()) - HeatStrokeDataFiller.fields_to_fill_with_zero - HeatStrokeDataFiller.fields_to_fill_with_zero
        logger.debug("Fields not modified: %s" % fields_not_modified.__str__())
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
        for temp_field in HeatStrokeDataFiller.temp_fields:
            where_string = HeatStrokeDataFiller.find_where_string(self.df, temp_field)
            self.df[temp_field][where_string] = 39
            farenheight_valus = self.df[temp_field] > 70
            self.df[temp_field].loc[farenheight_valus] = (self.df[temp_field][farenheight_valus] - 32.0) * 100.0 / (212 - 32)

    def fix_percentage_fields(self):
        self.percentage_fields = {"Humidity 8am", "Humidity noon", "Humidity 8pm"}
        for field in self.percentage_fields:
            where = self.df[field] > 1
            self.df[field].loc[where] = self.df[field][where] / 100

    def fix_keyword_fields(self):
        for field in self.df.columns:
            for i in range(self.df.shape[0]):
                value = self.df[field][i]
                if type(value) is str and value.replace("\"", "") in HeatStrokeDataFiller.keyword_map:
                    self.df[field].loc[i] = HeatStrokeDataFiller.keyword_map[value.replace("\"", "")]

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

    def fix_nationality_field(self):
        """
        This function changes any nationality that isn't "white" or "none" to a 1 (at risk)
        :return: None
        """
        where = self.df["Nationality"] == "None"
        where &= self.df["Nationality"] == "white"
        self.df["Nationality"].loc[where] = 0
        self.df["Nationality"].loc[np.invert(where)] = 1

    def fix_time_fields(self):
        """
        This function fixes time fields that may have been interpreted by pandas as datetime instances
        :return: None
        """
        time_fields = {"Time of day": lambda time: time.hour, "Time of year (month)": lambda time: time.month}
        for time_field in time_fields.keys():
            for i in range(self.df.shape[0]):
                value = self.df[time_field][i]
                if type(value) is datetime.time or type(value) is datetime.datetime:
                    self.df[time_field].loc[i] = time_fields[time_field](value)

    def combine_fields(self):
        """
        This function deals with combining several fields to create new fields.
        :return: None
        """
        humidity_fields = ['Humidity 8am', 'Humidity noon', 'Humidity 8pm']
        self.df['Relative Humidity'] = np.mean(self.df[humidity_fields], axis=1)

    def fix_fields(self):
        """
        This function fixes all the fields that have bad data in them that needs to be transformed
        :return: None
        """
        males = self.df["Sex"] == "M"
        self.df["Sex"] = np.array(males, dtype=int)

        logger.debug("Fixing bounded values...")
        self.fix_bounded_values()
        logger.debug("Fixing range values...")
        self.fix_range_fields()
        logger.debug("Fixing keyworded fields...")
        self.fix_keyword_fields()
        logger.debug("Fixing temperature fields...")
        self.fix_temperature_fields()
        logger.debug("Fixing nationality fields...")
        self.fix_nationality_field()
        logger.debug("Fixing percentage fields...")
        self.fix_percentage_fields()
        logger.debug("Combining fields...")
        self.combine_fields()

    def filter_data(self):
        """
        Returns the same data frame but only with the specified columns
        :param df: A pandas data frame
        :return: Another pandas dataframe but only with the important columns
        """
        self.df = self.df[HeatStrokeDataFiller.important_features]

    def make_and_append_negative_data(self):
        """
        This function generates some negative data points from default distributions and appends that to the data frame
        :return: None
        """
        negative_df = self.get_negative_data()
        self.df = pd.concat((self.df, negative_df))

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
                    logger.error("Type error: type %s for %s in \"%s\"" % (type(df[field][i]), df[field][i], field))
                    pass
        return where

    @staticmethod
    def find_where_string(df, field):
        # Finds where in the DataFrame there are strings in a certain column
        where = np.zeros(df.shape[0], dtype=bool)
        for i in range(df.shape[0]):
            if type(df[field][i]) is str:
                    where[i] = True
        return where

    def write_data(self):
        # For saving the data frame to file
        self.df.to_csv(self.filled_output_file)

    @staticmethod
    def create_fake_test_data(N=2000, num_fts=20):
        """
        This function is for getting FAKE data for testing
        :param N: Number of positive and negative data poitns
        :param num_fts: Number of features
        :return: a pandas DataFrame with the fake data
        """
        positive_data = np.ones((N, 1 + num_fts))
        negative_data = np.zeros((N, 1 + num_fts))
        for i in range(1, 1 + num_fts):
            positive_data[:, i] = i + np.random.normal(size=N)
            negative_data[:, i] = pow(i + 2, 2) + 1.25 * np.random.normal(size=N)
        columns = [HeatStrokeDataFiller.outcome_field] + list(string.ascii_lowercase[:num_fts])
        data = np.vstack((positive_data, negative_data))
        df = pd.DataFrame(data, columns=columns)
        return df

    def get_negative_data(self):
        """
        This function generates a pandas DataFrame with N elements each column/field. The data points are generated
        from functions (stored in the hash HeatStrokeDataFiller.negative_default) that each will take a parameter N
        as the number of points and return N data points drawn from an appropriate distribution for that field
        :param N: The number of data points
        :return: A pandas DataFrame
        """
        negative_df = pd.DataFrame(columns=HeatStrokeDataFiller.important_features, index=np.arange(self.num_negative))
        for field in negative_df.columns:
            parameter_distribution = HeatStrokeDataFiller.negative_default[field]
            negative_df[field].loc[:] = parameter_distribution(self.num_negative)
        return negative_df

def main():
    import argparse
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
        coloredlogs.install(level='DEBUG')
    elif args.verbose:
        warnings.filterwarnings('ignore')
        coloredlogs.install(level='INFO')
    else:
        warnings.filterwarnings('ignore')
        coloredlogs.install(level='WARNING')


    data_filter = HeatStrokeDataFiller()
    if args.input is not None:
        data_filter.excel_file = args.input
    if args.output is not None:
        data_filter.output_file = args.output

    data_filter.read_and_filter_data()
    data_filter.write_data()

if __name__ == "__main__":
    main()