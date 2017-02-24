#!/usr/bin/env python
'''
prediction_handler.py

This script implements a class called PredictoinHandler.py which contains a MonitorUser, 
HeatStrokeMonitor, and HeatStrokePredictor object. Instances of this class instantiate these
objects and couriers data between them to get and report predictions of heat stroke risk.
'''

import os
import time
import threading
import logging
import warnings

import pandas as pd
import numpy as np

import emoji
import coloredlogs
from termcolor import colored

import user
import monitor
import predictor


coloredlogs.install(level='DEBUG')
logging.basicConfig(format='[%(levelname)s][%(funcName)s] - %(message)s')
logger = logging.getLogger(__name__)

__author__ = "Jon Deaton"
__email__ = "jdeaton@stanford.edu"


class LoopingThread(threading.Timer):
    # This is a thread that performs some action
    # repeatedly at a given interval

    def __init__(self, callback, wait_time):
        threading.Thread.__init__(self)
        self.callback = callback
        self.wait_time = wait_time
        self._is_running = True

    def run(self):
        while self._is_running:
            self.callback()
            time.sleep(self.wait_time)

    def stop(self):
        self._is_running = False


class PredictionHandler(object):

    def __init__(self, username=None, output_dir=None):
        
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

        # Allocate a risk series for a risk estimate time series
        self.risk_series = pd.Series()

        # Set the output directory and save files
        # Make a directory to contain the files if one doesn't already exist
        if output_dir and not os.path.isdir(output_dir): os.mkdir(output_dir)
        
        # Set the output directory to be the current directory if one was not provided
        current_dir = os.path.dirname(os.path.realpath(__file__))
        self.output_dir = output_dir if output_dir else current_dir

        # Set the output file paths inside of the output directory
        self.risk_csv_file = os.path.join(self.output_dir, "risk_series.csv")
        self.data_save_file = os.path.join(self.output_dir, "all_data.csv")

    def initialize_threads(self, test=False):
        # Make a threading object to collect the data
        self.monitor.set_threading_class(test=test)

        # Make a thread that preiodically makes a risk prediction
        self.prediciton_thread = LoopingThread(self.make_prediction, 5)

        # Make a thread that periodically saves all the data
        self.saving_thread = LoopingThread(self.save_all_data, 30)

    def start_data_collection(self):
        # This function initiates a thread (handled by HeatStrokeMonitor)
        # that will continuously try to read and parse data from the Serial port
        self.monitor.read_data_from_port()

    def start_prediction_thread(self):
        # Start the prediction thread
        self.prediciton_thread.start()

    def stop_prediction_thread(self):
        # For starting the predicont thread
        self.prediciton_thread.stop()

    def get_current_attributes(self):
        # This function gets data from the MonitorUser instantiation and formats it in a way
        #logger.warning("get_current_attributes not implemented!")
        user_attributes = self.user.get_user_attributes()

    def save_all_data(self):
        # This saves all the recorded data including risk estimates
        df = self.monitor.get_compiled_df()
        num_to_append = self.risk_series.size - df.shape[0]
        if num_to_append > 0:
            filler = np.empty()
            filler[:] = np.NAN
            df.append(filler)

        # Add the risk data to the data frame
        df.loc[range(self.risk_series.size), "time Risk"] = self.risk_series.keys()
        df.loc[range(self.risk_series.size), "Risk"] = self.risk_series.values

        # Save the data frame
        df.to_csv(self.data_save_file)

    def make_prediction(self):
        # This funciton makes a prediction
        user_attributes = self.get_current_attributes()
        try:
            risk = self.predictor.make_prediction(user_attributes, self.monitor.HR_stream)
        except:
            risk = np.random.random()
        now = time.time()
        emojis = ":fire: " * int(risk / 0.1) + ":snowflake: " * int((1 - risk) / 0.1)
        logger.info(colored(emoji.emojize("Current risk: %f %s" % (risk, emojis)), 'red'))
        self.risk_series.set_value(now, risk)

def test(args):
    logger.debug("Instantiating prediciton handler...")
    handler = PredictionHandler(username=args.user, output_dir=args.output)
    logger.debug(emoji.emojize("Prediction handler instantiated :heavy_check_mark:"))

    # Tell the prediction handler whether or not to use prefiltered data or to refilter it
    handler.predictor.use_prefiltered = args.prefiltered
    # Initialize the logistic regression predictor using the filtered data
    handler.predictor.init_log_reg_predictor()

    # Create all of the threads that the handler needs
    handler.initialize_threads(test=args.no_bean)

    # Start all of the threads
    logger.info("Starting data collection thread...")
    handler.start_data_collection()
    logger.info("Starting data saving thread...")
    handler.saving_thread.start()
    logger.info("Starting prediction thread...")
    handler.start_prediction_thread()

    try:
        logger.warning("Pausing main thread (control-C to abort)...")
        # This makes is so that the user can press any key on the keyboard
        # but it won't exit unless they KeyboardInterrupt the process
        while True:
            input("")
    except KeyboardInterrupt:
        logger.warning("Keyboard Interrupt. Terminating threads...")

    handler.stop_prediction_thread()
    handler.monitor.stop_data_read()
    handler.saving_thread.stop()

    handler.save_all_data()
    
    logger.info(emoji.emojize("Test complete. :heavy_check_mark:"))

def main():
    import argparse
    script_description = "This script reads data from a monitor and uses"
    parser = argparse.ArgumentParser(description=script_description)

    input_group = parser.add_argument_group("Inputs")
    input_group.add_argument('-in', '--input', required=False, help='Input spreadsheet with case data')

    output_group = parser.add_argument_group("Outputs")
    output_group.add_argument("-out", "--output", help="Output directory")

    options_group = parser.add_argument_group("Opitons")
    options_group.add_argument("-f", "--fake", action="store_true", help="Use fake data")
    options_group.add_argument('-p', '--prefiltered', action="store_true", help="Use pre-filtered data")
    options_group.add_argument('-all', "--all-fields", dest="all_fields", action="store_true", help="Use all fields")
    options_group.add_argument('-test', '--test', action="store_true", help="Implementation testing")
    options_group.add_argument('-u', '--user', default=None, help="Monitor user name")
    options_group.add_argument('-nb', '--no-bean', dest="no_bean", action="store_true", help="Don't read from serial port")

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
        logger.warning("Integrated prediction not yet implemented. Use the --test flag.")

if __name__ == "__main__":
    main()