#!/usr/bin/env python
'''
predictor.py

This script implements the HeatStrokePredictor class, which implements several 
heat stroke predictoin algorithms. Objects of this class contain a HeatStrokeDataFiller
object which is used to revrieve patient data that is used for a logistic regression model.
'''

import os
import warnings
import logging
import meteocalc # For heaola[t index calculation

# My modules
import user
import monitor
import read_data

# Machine Learning Modules
import numpy as np
from sklearn import linear_model

logging.basicConfig(format='[%(levelname)s][%(funcName)s] - %(message)s')
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

        self.use_prefiltered = False

        logger.debug("Initializing data reader...")
        self.reader = read_data.HeatStrokeDataFiller()
        logger.debug("Reader initialized")

        self.use_all_fields = False
        self.fields_used = ['Patient temperature', 'Heat Index (HI)', 'Relative Humidity', 'Environmental temperature (C)']
        self.outcome_field = self.reader.outcome_field

        self.log_reg_predictor = None
        self.fit_log_reg_predictor = None

    # Logistic Regression Predictor
    def init_log_reg_predictor(self):
        self.log_reg_predictor = linear_model.LogisticRegression(C=1e5)
        self.fit_log_reg_classifier()
        
    def load_data_into_reader(self):
        if self.use_prefiltered:
            self.reader.read_prefiltered_data()
        else:
            self.reader.read_and_filter_data()

    def fit_log_reg_classifier(self):
        if self.reader.df is None:
            self.load_data_into_reader()

        y = np.array(self.reader.df[self.outcome_field])
        X = self.reader.df.drop(self.outcome_field, axis=1)
        if not self.use_all_fields:
            X  = X[self.fields_used]
        X = np.array(X)
        self.fit_log_reg_predictor = self.log_reg_predictor.fit(X, y)

    def make_log_reg_prediction(self, user_attributes):
        probas = self.fit_log_reg_predictor.predict_proba(user_attributes)
        return probabs[0]

    # Heat Index Predictor
    def make_heat_index_prediction(self, humidity, temperature, sun=0):
        temp = meteocalc.Temp(temperature, 'c')
        heat_index = meteocalc.heat_index(temperature=temp, humidity=humidity)
        # Wikipedia: Exposure to full sunshine can increase heat index values by up to 8 °C (14 °F)[7]
        # Heat Index on the website of the Pueblo, CO United States National Weather Service.
        # Link: http://web.archive.org/web/20110629041320/http://www.crh.noaa.gov/pub/heat.php
        heat_index += meteocalc.Temp(8, 'c') * sun
        # TODO: this is a fake implementation that needs to be implemented for real
        if heat_index.f < 80:
            risk = 0
        elif heat_index.f < 105:
            risk = 0.333
        elif heat_index.f < 130:
            risk = 0.666
        else:
            risk = 1
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
        for time in heat_rate_stream.keys:

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
        core_temp_series = self.estimate_core_temperature(heart_rate_series, CTstart)
        most_recent_CT_time = max(core_temp_series.keys())
        most_recent_ST_time = mac(skin_temperature_series.keys())

        # Average the estimate given by the Kalman Filter Model and skin temperature
        CT = (core_temp_series[most_recent_CT_time] + skin_temperature_series[most_recent_ST_time]) / 2
        if CT < 38:
            risk = 0
        elif CT < 42:
            risk = (42 - CT) / (42 - 38)
        else:
            risk = 1

        return risk

    # Makes all predicitons and combines them
    def make_prediction(self, user_attributes, heart_rate_stream):
        
        # Logistic regression
        LR_prob = self.make_log_reg_prediction(user_attributes)
        
        # Heat
        temp = user_attributes['Environmental temperature (C)']
        humidity = user_attributes['Relative Humidity']
        sun = user_attributes['Exposure to sun']
        HI_prob = self.make_heat_index_prediction(humidity, temp, sun=sun)
        
        # Core Temp Estimation
        CT_prob = self.core_temperature_risk(heart_rate_stream, skin_temperature_series)

        # Combined probability
        combined_prob = (CT_prob + HI_prob + LR_prob) / 3

        return combined_prob


def main():
    logger.info("Making predictor object...")
    predictor = HeatStrokePredictor()
    logger.info("Predictor initialized")

    logger.warning("No tests implemented.")

if __name__ == "__main__":
    main()