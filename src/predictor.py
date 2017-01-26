#!/usr/bin/env python

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

		self.log_reg_predictor = None

		# For calculating heat Index
		self.heat_index_predictor = None

		# For getting a risk assessment if there is no
		self.galvanic_response_predictor = None

		# For monitoring
		self.hypothalamic_regulation_failure = None

		# Wet Bulb Globe Temperature
		self.wbgt_predictor = None



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