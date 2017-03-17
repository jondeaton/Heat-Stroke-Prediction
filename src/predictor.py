#!/usr/bin/env python
'''
predictor.py

This script implements the HeatStrokePredictor class, which implements several 
heat stroke prediction algorithms. Objects of this class contain a HeatStrokeDataFiller
object which is used to retrieve patient data that is used for a logistic regression model.
'''

import os
import warnings
import logging
import meteocalc # For Heat Index calculation

# My modules
import user
import monitor
import read_data

# Machine Learning Modules
import numpy as np
import pandas as pd
from sklearn import linear_model

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

__author__ = "Jon Deaton"
__email__ = "jdeaton@stanford.edu"

class HeatStrokePredictor(object):

    def __init__(self):

        # For getting a risk assessment if there is no
        self.galvanic_response_predictor = None

        # For monitoring
        self.hypothalamic_regulation_failure = None

        # Wet Bulb Globe Temperature
        self.wbgt_predictor = None

        # Use prefiltered data or re-impute the literature data
        self.use_prefiltered = False

        logger.debug("Initializing data reader...")
        self.reader = read_data.HeatStrokeDataFiller()
        logger.debug("Reader initialized")

        self.use_all_fields = False
        self.fields_used = ['Patient temperature', 'Heat Index (HI)', 
        'Relative Humidity', 'Environmental temperature (C)']

        self.outcome_field = self.reader.outcome_field

        self.log_reg_predictor = None
        self.fit_log_reg_predictor = None

    def init_log_reg_predictor(self):
        # Logistic Regression Predictor Instantiation
        self.log_reg_predictor = linear_model.LogisticRegression(C=1e5)
        self.fit_log_reg_classifier()
        
    def load_data_into_reader(self):
        # This function either loads the data from a previous file or re-imputes the data
        if self.use_prefiltered:
            self.reader.read_prefiltered_data()
        else:
            self.reader.read_and_filter_data()

    def fit_log_reg_classifier(self):
        # Setup the Logistic Regression classifier using the data in reader.df

        # If we haven't already loaded the data make sure that we've done that
        if self.reader.df is None:
            self.load_data_into_reader()

        y = np.array(self.reader.df[self.outcome_field])
        X = self.reader.df.drop(self.outcome_field, axis=1)
        if not self.use_all_fields:
            X = X[self.fields_used]
        X = np.array(X)
        self.fit_log_reg_predictor = self.log_reg_predictor.fit(X, y)

    def make_log_reg_prediction(self, user_attributes):
        # This function returns a Logistic-Regression calculated probability
        # user_state needs to be
        # logger.debug("Making Logistic Regression prediction with:")
        # print(user_attributes)

        X = [[user_attributes[field] for field in self.fields_used]]
        probas = self.fit_log_reg_predictor.predict_proba(X)

        # predict_proba returns a 2D array of the probabilities of X being in the classes
        # with each class given by the columns of the returned array. To make sure that we
        # are getting the probability of posative classification, find out which column
        # of predict_proba is the one that corresponds to the posative (1) classification.
        index_of_posative_class = np.where(self.fit_log_reg_predictor.classes_ == 1)[0][0]
        return probas[0][index_of_posative_class]

    def calculate_heat_index(self, humidity, temperature, sun=0):
        # Heat Index Calculation
        # Make sure that humidity is a ratio. NOT A PERCENTAGE
        if humidity > 1 or humidity < 0:
            raise ValueError("Humidity: %f is not in range [0, 1]. Don't use percentages")
        temp = meteocalc.Temp(temperature, 'c')
        heat_index = meteocalc.heat_index(temperature=temp, humidity= 100 * humidity) # Stupid meteocalc likes percentages
        # Wikipedia: Exposure to full sunshine can increase heat index values by up to 8 °C (14 °F)[7]
        # Heat Index on the website of the Pueblo, CO United States National Weather Service.
        # Link: http://web.archive.org/web/20110629041320/http://www.crh.noaa.gov/pub/heat.php
        heat_index += meteocalc.Temp(8, 'c') * sun
        return heat_index

    def make_HI_risk_prediction(self, humidity, temperature, sun=0):
        # Heat Index Risk Predictor
        heat_index = self.calculate_heat_index(humidity, temperature, sun=sun)

        hi = heat_index.c # Heat index in Celsius
        low_sat = 30 # Lower temperature saturation in Celsius
        up_sat = 41 # Upper temperature saturatio in Celsius

        risk = 0 if hi < low_sat else (hi - low_sat) / (up_sat - low_sat) if hi < up_sat else 1 

        return risk

    def estimate_core_temperature(self, heart_rate_series, CTstart):
        '''
        heart_rate_series: A pandas Series mapping time --> heart rate, ideally once per minute
        CTstart: A core-temperature starting value

        Kalman Filter Model adapted from Buller et al.
        Source: Buller MJ et al. “Estimation of human core temperature from sequential 
        heart rate observations.” Physiol Meas. 2013 Jul;34(7):781-98. doi: 10.1088/0967-3334/34/7/781. Epub 2013 Jun 19.
        '''
        # Extended Kalman Filter Parameters
        a = 1;
        gamma = pow(0.022, 2)

        b_0 = -7887.1
        b_1 = 384.4286
        b_2 = -4.5714
        sigma = pow(18.88, 2)

        # Initialize Kalman filter
        x = CTstart
        v = 0 # v = 0 assumes confidence with start value

        core_temp_series = pd.Series()

        # Iterate through HR time sequence
        for time in heart_rate_series.keys():

            # Time Update Phase
            x_pred = a * x # Equation 3
            v_pred = pow(a, 2) * v + gamma # Equation 4

            #Observation Update Phase
            z = heart_rate_series[time]
            c_vc = 2 * b_2 * x_pred + b_1 # Equation 5
            k = (v_pred * c_vc) / (pow(c_vc, 2) * v_pred + sigma) # Equation 6
            x = x_pred + k * (z - (b_2 * pow(x_pred, 2) + b_1 * x_pred + b_0)) # Equation 7
            v = (1 - k * c_vc) * v_pred # Equation 8
            core_temp_series.set_value(time, x)

        return core_temp_series

    def core_temperature_risk(self, heart_rate_series, skin_temperature_series, CTstart=37):
        # This function estimates risk of Heat Stroke based on core temperature
        # estimation from heart rate

        self.core_temp_series = self.estimate_core_temperature(heart_rate_series, CTstart)
        
        # Average the estimate given by the Kalman Filter Model and skin temperature
        most_recent_CT_time = max(self.core_temp_series.keys())
        #most_recent_ST_time = mac(skin_temperature_series.keys())
        #CT = (self.core_temp_series[most_recent_CT_time] + skin_temperature_series[most_recent_ST_time]) / 2
        
        # Just kidding, use the heart-rate-estimated value
        CT = self.core_temp_series[most_recent_CT_time]

        upper = 40
        lower = 38
        x = (CT - lower) / (upper - lower)
        # Logistic Curve
        risk = 1 / (1 + np.exp(3.6 - 7 * x))

        # Alternatively
        #risk = 0 if x < 0 else x if x < 1 else 1

        return risk

    def _fill_current_attributes(self, user_attributes, verbose=True):
        # This function takes a pandas Series meant to represent the current user attributes
        # and fills it with values that may be missing from it so that it can be fed into the
        # Logistic Regression

        # Yeah these are super fucked
        if np.isnan(user_attributes.Acceleration):
            user_attributes.set_value('Acceleration', 0.5)

        if np.isnan(user_attributes['Skin color (flushed/normal=1, pale=0.5, cyatonic=0)']):
            user_attributes.set_value('Skin color (flushed/normal=1, pale=0.5, cyatonic=0)', 0.5)

        return user_attributes

    def make_predictions(self, user_attributes, heart_rate_stream, skin_temperature_series):
        # This function makes all Heat Stroke Risk Prediction

        # Core Temp Risk Estimation
        # This function will also update self.core_temp_series, which can be used to 
        # fill the current user_attributes Series with current core temperature
        try:
            CT_prob = self.core_temperature_risk(heart_rate_stream, skin_temperature_series)
        except:
            logger.warning("Could not make core temperature risk estimation")
            CT_prob = None

        # Heat Index Risk Estimation
        temp = user_attributes['Environmental temperature (C)']
        humidity = user_attributes['Relative Humidity']
        sun = user_attributes['Exposure to sun']
        try:
            HI_prob = self.make_HI_risk_prediction(humidity, temp, sun=sun)
        except:
            logger.warning("Could not make Heat Index risk estimation")
            HI_prob = None

        # Logistic Regression calculation
        user_attributes = self._fill_current_attributes(user_attributes)
        if np.any(np.isnan(user_attributes.values)):
            logger.warning("Insufficient data for Logistic Regression.")
            LR_prob = None
        else:
            LR_prob = self.make_log_reg_prediction(user_attributes)
        
        # Combined probability
        return CT_prob, HI_prob, LR_prob

    def combine_predictions(self, CT_prob, HI_prob, LR_prob):
        # This function decides how to combine the different predicted 
        # risks into one comprehensive
        CT_prob = CT_prob if CT_prob is not None else 0
        HI_prob = HI_prob if HI_prob is not None else 0
        LR_prob = LR_prob if LR_prob is not None else 0

        combined_prob = (CT_prob + HI_prob + LR_prob) / 3
        return combined_prob 

def main():
    logger.info("Making predictor object...")
    predictor = HeatStrokePredictor()
    logger.info("Predictor initialized")

    logger.warning("No tests implemented.")

if __name__ == "__main__":
    main()