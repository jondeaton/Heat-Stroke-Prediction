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
import emoji
import coloredlogs

import user
import monitor
import predictor


coloredlogs.install(level='DEBUG')
logging.basicConfig(format='[%(levelname)s][%(funcName)s] - %(message)s')
logger = logging.getLogger(__name__)

__author__ = "Jon Deaton"
__email__ = "jdeaton@stanford.edu"

class PredictionHandler(object):

    def __init__(self, username=None):
        
        logger.debug("Instantiating user...")
        self.user = user.MonitorUser(load=True, username=username)
        logger.info(emoji.emojize("Monitor user: %s %s" % (self.user.name, self.user.emoji)))

        logger.debug("Instantiating monitor...")
        self.monitor = monitor.HeatStrokeMonitor()
        logger.debug(emoji.emojize("Monitor instantiated :heavy_check_mark:"))

        logger.debug("Instantiating predictor...")
        self.predictor = predictor.HeatStrokePredictor()
        logger.debug(emoji.emojize("Predictor instantiated :heavy_check_mark:"))
        
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


def test(args):
    logger.debug("Instantiating prediciton handler...")
    handler = PredictionHandler(username=args.user)
    logger.debug(emoji.emojize("Prediction handler instantiated :heavy_check_mark:"))

    handler.predictor.use_prefiltered = args.prefiltered
    handler.predictor.init_log_reg_predictor()

    logger.info(emoji.emojize("Test complete. :heavy_check_mark:"))


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
    options_group.add_argument('-test', '--test', action="store_true", help="Implementation testing")
    options_group.add_argument('-u', '--user', default=None, help="Monitor user name")

    console_options_group = parser.add_argument_group("Console Options")
    console_options_group.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    console_options_group.add_argument('--debug', action='store_true', help='Debug console')

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(format='[%(asctime)s][%(levelname)s][%(funcName)s] - %(message)s')
        coloredlogs.install(level='DEBUG')
    elif args.verbose:
        warnings.filterwarnings('ignore')
        logging.basicConfig(format='[%(asctime)s][%(levelname)s][%(funcName)s] - %(message)s')
        coloredlogs.install(level='INFO')
    else:
        warnings.filterwarnings('ignore')
        logging.basicConfig(format='[log][%(levelname)s] - %(message)s')
        coloredlogs.install(level='WARNING')

    if args.test:
        logger.info(emoji.emojize('Initializing test... :fire: :fire: :fire:'))
        test(args)
    else:
        logger.warning("Integrated testing not yet implemented. Use --test flag.")


if __name__ == "__main__":
    main()