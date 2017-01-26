#!/usr/bin/env python

import time
import pandas as pd
import logging

logging.basicConfig(format='[%(levelname)s][%(funcName)s] - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

__author__ = "Jon Deaton"
__email__ = "jdeaton@stanford.edu"

class PredictionHandler(object)

	def __init__(self):

		self.risk_series = pd.Series()

		logger.info("Initializing user...")
		self.user = user.MonitorUser(load=True)
		logger.info("Monitor User: {name}".foramt(name=self.user.name))

		logger.info("Initializing monitor...")
		self.monitor = monitor.HeatStrokeMonitor()
		logger.info("Monitor initialized")

		logger.info("Initializing predictor...")
		self.predictor = predictor.HeatStrokePredictor()
		logger.info("Predictor initialized")

	def start_data_collection(self):
		monitor.read_data_from_port()




def main():
	import argparses
	script_description = "This script reads data from a monitor and uses"
    parser = argparse.ArgumentParser(description=script_description)

    input_group = parser.add_argument_group("Inputs")
    input_group.add_argument('-in', '--input', required=False, help='Input spreadsheet with case data')

    output_group = parser.add_argument_group("Outputs")
    output_group.add_argument("-out", "--output", help="Predictoin Output")

    options_group = parser.add_argument_group("Opitons")
    options_group.add_argument("-f", "--fake", action="store_true", help="Use fake data")
    options_group.add_argument('-p', '--prefiltered', action="store_true", help="Use pre-filtered data")
    options_group.add_argument('-all', "--all-fields", dest="all_fields", action="store_true", help="Use all fields")

    console_options_group = parser.add_argument_group("Console Options")
    console_options_group.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    console_options_group.add_argument('--debug', action='store_true', help='Debug console')

    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
        logging.basicConfig(format='[%(asctime)s][%(levelname)s][%(funcName)s] - %(message)s')
    elif args.verbose:
        warnings.filterwarnings('ignore')
        logger.setLevel(logging.INFO)
        logging.basicConfig(format='[%(asctime)s][%(levelname)s][%(funcName)s] - %(message)s')
    else:
        warnings.filterwarnings('ignore')
        logger.setLevel(logging.WARNING)
        logging.basicConfig(format='[log][%(levelname)s] - %(message)s')



if __name__ == "__main__":
	main()