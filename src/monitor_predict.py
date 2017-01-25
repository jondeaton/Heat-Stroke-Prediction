#!/usr/bin/env python

import os
import argparse
import warnings
import logging
import read_data
import user

logging.basicConfig(format='[%(levelname)s][%(funcName)s] - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

__author__ = "Jon Deaton"
__email__ = "jdeaton@stanford.edu"

def HeatStrokePredictor(object):

	def __init__(self):
		self.monitor = None
		self.user = None
		self.reader = read_data.HeatStrokeDataFiller()

		self.log_reg_predictor = None

		# For calculating heat Index
		self.heat_index_predictor = None

		# For getting a risk assessment if there is no
		self.galvanic_response_predictor = None

		# For monitoring
		self.hypothalamic_regulation_failure = None

		# Wet Bulb Globe Temperature
		self.wbgt_predictor = None

	def init_monitor(self):
		self.monitor = HeatStrokeMonitor()


	def make_prediction(self):


def main():
	logger.info("Making predictor object...")
	predictor = HeatStrokePredictor()
	

if __name__ == "__main__":
	main()