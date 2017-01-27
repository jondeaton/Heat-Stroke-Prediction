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

# My modules
import user
import monitor
import read_data

logging.basicConfig(format='[%(levelname)s][%(funcName)s] - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

__author__ = "Jon Deaton"
__email__ = "jdeaton@stanford.edu"

class HeatStrokePredictor(object):

	def __init__(self):
		logger.info("Initializing data reader...")
		self.reader = reader.HeatStrokeDataFiller()
		logger.info("Reader initialized")

		self.log_reg_predictor = linear_model.LogisticRegression(C=1e5)
		fitted = classifier.fit(X[train], y[train])
        probas = fitted.predict_proba(X[test])

		# For calculating heat Index
		self.heat_index_predictor = None

		# For getting a risk assessment if there is no
		self.galvanic_response_predictor = None

		# For monitoring
		self.hypothalamic_regulation_failure = None

		# Wet Bulb Globe Temperature
		self.wbgt_predictor = None

		self.fields_used = ['Patient temperature', 'Heat Index (HI)', 'Relative Humidity', 'Environmental temperature (C)']


	def make_user_log_reg_input(self):
		# todo
		pass

	def make_prediction(self):
		pass

def main():
	logger.info("Making predictor object...")
	predictor = HeatStrokePredictor()
	logger.info("Predictor initialized")

	logger.info("Initializing data collection...")
	predictor.start_data_collectoin()

if __name__ == "__main__":
	main()