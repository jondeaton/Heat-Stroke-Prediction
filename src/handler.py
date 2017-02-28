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

import numpy as np
import pandas as pd

import emoji
import coloredlogs
from termcolor import colored

import user
import monitor
import predictor

coloredlogs.install(level='DEBUG')
logger = logging.getLogger(__name__)

__author__ = "Jon Deaton"
__email__ = "jdeaton@stanford.edu"

class LoopingThread(threading.Timer):
    # This is a thread that performs some action
    # repeatedly at a given interval. Since this
    # interval may be long, this 

    def __init__(self, callback, wait_time):
        threading.Thread.__init__(self)
        self.callback = callback
        self.wait_time = wait_time

        self.loop_wait = 1 if wait_time > 1 else wait_time
        self.num_loops = int(wait_time / self.loop_wait)

        self._is_running = True

    def run(self):
        while self._is_running:
            for _ in range(self.num_loops):
                # Check to make sure the thread should still be running
                if not self._is_running: return
                time.sleep(self.loop_wait)
            self.callback()

    def stop(self):
        self._is_running = False

class PredictionHandler(object):

    def __init__(self, users_XML="users.xml", username=None, output_dir=None, timestamp_files=False, live_plotting=False):
        
        logger.debug("Instantiating MonitorUser...")
        self.user = user.MonitorUser(users_XML=users_XML, load=True, username=username)
        logger.info(emoji.emojize("MonitorUser name: %s %s" % (self.user.name, self.user.emoji)))

        logger.debug("Instantiating HeatStrokeMonitor...")
        self.monitor = monitor.HeatStrokeMonitor()
        logger.debug(emoji.emojize("HeatStrokeMonitor instantiated :heavy_check_mark:"))

        logger.debug("Instantiating HeatStrokePredictor...")
        self.predictor = predictor.HeatStrokePredictor()
        logger.debug(emoji.emojize("HeatStrokePredictor instantiated :heavy_check_mark:"))

        self.live_plotting = live_plotting
        if live_plotting:
            logger.debug("Instantiating LivePlotter....")
            self.plotter = plotter.LivePlotter()
            logger.debug(emoji.emojize("LivePlotter instantiated :heavy_check_mark:"))

        self.current_fields = self.user.series.keys()
        self.user_fields = ['Age', 'Sex', 'Weight (kg)', 'BMI', 'Height (cm)',
                             'Nationality', 'Cardiovascular disease history',
                             'Sickle Cell Trait (SCT)'] 

        # Allocate a risk series for a risk estimate time series
        self.risk_series = pd.Series()
        self.CT_risk_series = pd.Series()
        self.HI_risk_series = pd.Series()
        self.LR_risk_series = pd.Series()

        # Set all the output files to the appropriate paths
        self.set_output_files(output_dir, timestamp_files)

    # Thread control
    def initialize_threads(self, test=False):
        # Make a threading object to collect the data
        self.monitor.set_threading_class(test=test)

        # Make a thread that preiodically makes a risk prediction
        self.prediciton_thread = LoopingThread(self.make_predictions, 5)

        # Make a thread that periodically saves all the data
        self.saving_thread = LoopingThread(self.save_all_data, 30)

        # Make a thread for live plotting
        self.plotting_thread = LoopingThread(self.refresh_plots, 5)

    def start_data_collection(self):
        # This function initiates a thread (handled by HeatStrokeMonitor)
        # that will continuously try to read and parse data from the Serial port
        self.monitor.read_data_from_port(log=True)

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

        # We need to loop through all of the differen data streams coming from the monitor
        # and store the most recent value in ihe user's attributes
        # This dictionary provides a mapping from user attribute field name to
        # the relevant field
        stream_dict = {
        'Heart / Pulse rate (b/min)': self.monitor.HR_stream, 
        'Environmental temperature (C)': self.monitor.ETemp_stream,
        'Relative Humidity': self.monitor.EHumid_stream,
        'Skin Temperature': self.monitor.STemp_stream,
        'Sweating': self.monitor.GSR_stream,
        'Acceleration': self.monitor.Acc_stream,
        'Skin color (flushed/normal=1, pale=0.5, cyatonic=0)': self.monitor.Skin_stream
        }

        # Loop through all the streams and add the most recent value to user_attributes
        for field in stream_dict:
            stream = stream_dict[field]
            value = stream.iloc[-1] if stream.size else np.NAN
            if value is np.NAN: logger.error("No data for: \"%s\"" % field)
            user_attributes.set_value(field, value)

        user_attributes.set_value('Exposure to sun', 0)

        return user_attributes

    def make_predictions(self, verbose=True):
        # This funciton makes a Heat Stroke risk prediction        

        user_attributes = self.get_current_attributes()

        # Calculate the risk!!!
        CT_prob, HI_prob, LR_prob = self.predictor.make_predictions(user_attributes, self.monitor.HR_stream, self.monitor.STemp_stream)
        risk = self.predictor.combine_predictions(CT_prob, HI_prob, LR_prob)

        # Record the time that the risk assessment was made, and save it to the series
        now = time.time()
        self.risk_series.set_value(now, risk)
        self.CT_risk_series.set_value(now, CT_prob)
        self.HI_risk_series.set_value(now, HI_prob)
        self.LR_risk_series.set_value(now, LR_prob)

        # Log the risk to terminal if verbose
        if verbose:
            logger.info(colored("CT Risk: %.4s\t%s" % (CT_prob, progress_bar(CT_prob)), "yellow"))
            logger.info(colored("HI Risk: %.4s\t%s" % (HI_prob, progress_bar(HI_prob)), "yellow"))
            logger.info(colored("LR Risk: %.4s\t%s" % (LR_prob, progress_bar(LR_prob)), "yellow"))
            bar = progress_bar(risk, filler=":fire: ")
            logger.info(colored(emoji.emojize("Current risk: %.4f %s" % (risk, bar)), 'red'))
        
    def stop_all_threads(self, wait=False):
        # This function sends a stop signal to all threads
        self.stop_prediction_thread()
        self.monitor.stop_data_read()
        self.saving_thread.stop()

        # The optional 'wait' argument indicates whether this function should wait to return until it is sure
        # that all of the treads have stopped running
        if wait:
            logger.debug("Waiting for threads to die...")
            while True:
                try:
                    # Waiting for all the threads to stop
                    while threading.activeCount() > 1: time.sleep(0.1)
                    break
                except KeyboardInterrupt:
                    # This part just makes so that if the user mashes the KeyboardInterrupt
                    # the program will still exit gradefully without spitting out lots of errors
                    continue

            logger.debug("Threads died. Thread count: %d" % threading.activeCount())

    def refresh_plots(self):


    def save_all_data(self):
        # This saves all the recorded data including risk estimates
        logger.debug("Saving data to: %s ..." % os.path.basename(self.data_save_file))

        df = self.monitor.get_compiled_df()
        core_temperature_series = self.predictor.estimate_core_temperature(self.monitor.HR_stream, 37.6)

        # The dataframe returned by the monitor not be large enough to hold all of the 
        # risk series data so we need to make it bigger if necessary
        longest_risk_series = max(self.risk_series.size, self.CT_risk_series.size,
                                    self.HI_risk_series.size, self.LR_risk_series.size, core_temperature_series.size)

        num_to_append = longest_risk_series - df.shape[0]
        if num_to_append > 0:
            # Add a bunch of empty (NAN) values to the dataframe is we need extra space
            # for the risk vlaues
            filler = np.empty()
            filler[:] = np.NAN
            df.append(filler)

        # Add the risk/Estimated Core temperature data to the DataFrame
        df.loc[range(self.risk_series.size), "time Risk"] = self.risk_series.keys()
        df.loc[range(self.risk_series.size), "Risk"] = self.risk_series.values

        df.loc[range(self.HI_risk_series.size), "time HI Risk"] = self.HI_risk_series.keys()
        df.loc[range(self.HI_risk_series.size), "HI Risk"] = self.HI_risk_series.values

        df.loc[range(self.CT_risk_series.size), "time CT Risk"] = self.CT_risk_series.keys()
        df.loc[range(self.CT_risk_series.size), "CT Risk"] = self.CT_risk_series.values

        df.loc[range(self.LR_risk_series.size), "time LR Risk"] = self.LR_risk_series.keys()
        df.loc[range(self.LR_risk_series.size), "LR Risk"] = self.LR_risk_series.values

        df.loc[range(core_temperature_series.size), "time est CT"] = core_temperature_series.keys()
        df.loc[range(core_temperature_series.size), "est CT"] = core_temperature_series.values

        # Save the data frame to file! yaas!
        df.to_csv(self.data_save_file)

    def set_output_files(self, output_dir, timestamp_files):
        # Set the output directory and save files
        # Make a directory to contain the files if one doesn't already exist
        if output_dir and not os.path.isdir(output_dir): os.mkdir(output_dir)
        
        # Set the output directory to be the data directory if one was not provided
        current_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        self.output_dir = output_dir if output_dir else current_dir

        # A timestamp for for output files
        timestamp = time.strftime("_%Y.%m.%d-%H.%M.%S_") if timestamp_files else ""

        # Set the output file paths inside of the output directory
        self.risk_csv_file = os.path.join(self.output_dir, "risk_series%s.csv" % timestamp)
        self.data_save_file = os.path.join(self.output_dir, "all_data%s.csv" % timestamp)


def progress_bar(progress, filler="=", length=10):
    # This function makes a string that looks like a progress bar
    # Example: progress of 0.62 would give the following string: "[======    ]"
    progress = 0 if progress is None else progress
    return "[" + filler * int(0.5 + progress * length) + " " * (int(0.5 + (1 - progress) * length)) + "]"

def simulation(args):
    # This is for doing a simulation with data saved to file rather than read from the bean
    logger.error("Data simulation not yet implemented! Run without -S flag")

def run(args):
    logger.info(emoji.emojize('Running test: %s ...' % __file__ + ' :fire:' * 3))
    logger.debug("Instantiating prediciton handler...")
    handler = PredictionHandler(users_XML= args.users_XML, username=args.user, 
        output_dir=args.output, timestamp_files=args.timestamp_files, live_plotting=args.live_plotting)
    logger.debug(emoji.emojize("Prediction handler instantiated :heavy_check_mark:"))

    # Tell the prediction handler whether or not to use prefiltered data or to refilter it
    handler.predictor.use_prefiltered = args.prefiltered
    # Initialize the logistic regression predictor using the filtered data
    handler.predictor.init_log_reg_predictor()

    # Create all of the threads that the handler needs
    handler.initialize_threads(test=args.no_bean or args.test)

    start_time = time.time()

    # Start all of the threads
    logger.info("Starting data collection thread...")
    handler.start_data_collection()
    logger.info("Starting data saving thread...")
    handler.saving_thread.start()
    logger.info("Starting prediction thread...")
    handler.start_prediction_thread()

    try:
        logger.warning("Pausing main thread ('q' or control-C to abort)...")
        # This makes is so that the user can press any key on the keyboard
        # but it won't exit unless they KeyboardInterrupt the process
        while True:
            user_input = input("")
            if user_input == 'q':
                logger.warning("Exit signal recieved. Terminating threads...")
                break
    except KeyboardInterrupt:
        logger.warning("Keyboard Interrupt. Terminating threads...")

    # Save the data
    handler.save_all_data()
    # Stop the threads
    handler.stop_all_threads(wait=True)
    # Indicate that the test has finished
    logger.info(emoji.emojize("Test complete. :heavy_check_mark:"))

def main():
    import argparse
    script_description = "This script is the front-end handler of the Heat Stroke Monitor system."
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
    options_group.add_argument("--users", dest="users_XML", default=None, help="Monitor users XML file")
    options_group.add_argument('-nb', '--no-bean', dest="no_bean", action="store_true", help="Don't read from serial port")
    options_group.add_argument('-tf', '--timestamp-files', dest="timestamp_files", action="store_true", help="Save unique data files")

    simulation_group = parser.add_argument_group("Simulation")
    simulation_group.add_argument('-S', '--simulate', action="store_true", help="Simulate run with saved data")
    simulation_group.add_argument('-rt', '--real-time', dest="real_time", action="store_true", help="Simulate in real-time")

    plotting_group = parser.add_argument_group("Live Plotting")
    plotting_group.add_argument('-plot', '--live-plotting', dest="live_plotting", action="store_true", help="Display live plots")

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

    if args.simulate:
        simulate(args)
    else:
        if args.real_time: logger.warning("-rf (\"--real-time\") flag for use with simulation (-S)")
        run(args)

if __name__ == "__main__":
    main()