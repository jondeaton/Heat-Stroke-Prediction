#!/usr/bin/env python
'''
prediction_handler.py

This script implements a class called PredictoinHandler.py which contains a MonitorUser, 
HeatStrokeMonitor, and HeatStrokePredictor object. Instances of this class instantiate these
objects and couriers data between them to get and report predictions of heat stroke risk.
'''

import time
import pandas as pd
import logging
import warnings
from termcolor import colored, cprint

import user
import monitor
import predictor

logging.basicConfig(format='[%(levelname)s][%(funcName)s] - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

__author__ = "Jon Deaton"
__email__ = "jdeaton@stanford.edu"

class PredictionHandler(object):

    def __init__(self):
        
        logger.info("Initializing user...")
        self.user = user.MonitorUser(load=True)
        logger.info("Monitor User: {name}".format(name=self.user.name))

        logger.info("Initializing monitor...")
        self.monitor = monitor.HeatStrokeMonitor()
        logger.info("Monitor initialized")

        logger.info("Initializing predictor...")
        self.predictor = predictor.HeatStrokePredictor()
        logger.info("Predictor initialized")
        
        self.current_fields = self.user.series.keys()
        self.user_fields = ['Age', 'Sex', 'Weight (kg)', 'BMI', 'Height (cm)',
                             'Nationality', 'Cardiovascular disease history', 'Sickle Cell Trait (SCT)'] 

        self.risk_series = pd.Series()

    def start_data_collection(self):
        monitor.read_data_from_port()

    def get_current_attributes(self):
        logger.warning("get_current_attributes not instantiated!")

    def make_predictoin(self):
        logger.warning("make_prediction not implemented!")


def test():
    logger.info("Initializing prediciton handler...")
    handler = PredictionHandler()
    logger.info("Instantiate prediction handler.")


def main():
    import argparse
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
    options_group.add_argument('-test', '--test', action="store_true", help="Implementation testing.")


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


    if args.test:
        test()
    else:
        logger.warning("Integrated testing not yet implemented. Use --test flag.")


if __name__ == "__main__":
    main()