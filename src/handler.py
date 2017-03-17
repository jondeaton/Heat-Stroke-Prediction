#!/usr/bin/env python
'''
handler.py

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
import plotter

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

    def __init__(self, users_XML="users.xml", username=None, output_dir=None, timestamp_files=False, live_plotting=False, show_plots=False, plotly=False):

        # Set all the output files to the appropriate paths
        self.set_output_files(output_dir, timestamp_files)

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
            self.plotter = plotter.LivePlotter(self.data_save_file, output_directory=self.output_dir, show=show_plots, plotly=plotly)
            self.plotter.show = show_plots
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

        self.CT_stream = None # will be set by predictor.estimate_core_temperature
        self.HI_stream = pd.Series() # All of the Heat Index values with associated times


    # Thread control
    def initialize_threads(self, test=False):
        # Make a threading object to collect the data
        self.monitor.set_threading_class(test=test)

        # Make a thread that preiodically makes a risk prediction
        self.prediciton_thread = LoopingThread(self.make_predictions, 5)

        # Make a thread that periodically saves all the data
        self.saving_thread = LoopingThread(self.save_all_data, 30)

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

    def start_all_threads(self):
        # Start all of the threads
        logger.info("Starting data collection thread...")
        self.start_data_collection()
        logger.info("Starting data saving thread...")
        self.saving_thread.start()
        logger.info("Starting prediction thread...")
        self.start_prediction_thread()
        if self.live_plotting:
            logger.info("Starting plotting thread...")
            self.plotter.start_plotting()

    def stop_all_threads(self, wait=False):
        # This function sends a stop signal to all threads
        if self.live_plotting:
            self.plotter.stop_plotting()
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

    def update_user_attributes(self, verbose=True):
        # This function gets demographic data from the MonitorUser instance
        # in the form of a pandas Series and then adds fields to this series 
        # using data form the HeatStrokeMonitor instance so that it represents
        # the user's current state. This function will also set the core temperature
        # stream and Heat index stream which are both pandas Series to be updated using
        # values calculated by making calls to the HeatStrokePredictor instance.
        
        # Get basic demographic data from MonitorUser
        self.user_attributes = self.user.get_user_attributes()

        # Set the Core Temperature stream using Heart Rate Stream with prediction
        self.CT_stream = self.predictor.estimate_core_temperature(self.monitor.HR_stream, 37)
        # Also add the patient temperature to reflect
        if self.CT_stream.size == 0:
            # If there is no heart rate data then bummer
            current_extimated_CT = np.NAN
        else:
            current_extimated_CT = self.CT_stream[max(self.CT_stream.keys())]
        self.user_attributes.set_value('Patient temperature', current_extimated_CT)
        if verbose: logger.info("Estimated current core temperature: %.3f C" %  current_extimated_CT)

        # We need to loop through all of the different data streams coming from the monitor
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
        'Skin color (flushed/normal=1, pale=0.5, cyatonic=0)': self.monitor.Skin_stream}

        # Loop through all the streams and add the most recent value to user_attributes
        for field in stream_dict:
            stream = stream_dict[field]
            value = stream.iloc[-1] if stream.size else np.NAN
            if value is np.NAN: logger.error("No data for: \"%s\"" % field)
            self.user_attributes.set_value(field, value)

        # Add current Heat Index value calculated from the most recent environmental temperature
        # and humidity measurements. Time of Heat Index set to be the time of temperature measurement
        if self.monitor.ETemp_stream.size > 0:
            temp_time = max(self.monitor.ETemp_stream.keys())
            current_temp = self.user_attributes['Environmental temperature (C)']
            current_humidity = self.user_attributes['Relative Humidity']
            current_heat_index = self.predictor.calculate_heat_index(current_humidity, current_temp).c
            self.HI_stream.set_value(temp_time, current_heat_index)
            self.user_attributes.set_value('Heat Index (HI)', current_heat_index)
            if verbose: logger.info("Heat Index: %.3f C" % current_heat_index)

            # Set skin temperature to be the average between core temp and environmental
            self.user_attributes.set_value('Skin Temperature', np.mean([current_temp, current_extimated_CT]))

        # We don't have any sensor for this so we're just gonna set it to zero always... yeah caus science
        self.user_attributes.set_value('Exposure to sun', 0)

    def make_predictions(self, verbose=True):
        # This function makes a Heat Stroke risk prediction

        # Updates self.user_attributes to have all of the necessary  things to feed to the predictor
        self.update_user_attributes()

        # Calculate all risks
        tup = self.predictor.make_predictions(self.user_attributes, self.monitor.HR_stream, self.monitor.STemp_stream)
        CT_prob, HI_prob, LR_prob = tup

        # Combine the risks into one comprehensive value
        risk = self.predictor.combine_predictions(CT_prob, HI_prob, LR_prob)

        # Record the time that the risk assessment was made, and save it to the series
        now = time.time()
        self.risk_series.set_value(now, risk)
        self.CT_risk_series.set_value(now, CT_prob)
        self.HI_risk_series.set_value(now, HI_prob)
        self.LR_risk_series.set_value(now, LR_prob)

        # Log the risk to terminal if verbose
        if verbose:
            logger.info(colored("Core Temp Risk: %s  %s" % (progress_bar(CT_prob), CT_prob), "yellow"))
            logger.info(colored("Heat Idx. Risk: %s  %s" % (progress_bar(HI_prob), HI_prob), "yellow"))
            logger.info(colored("Log. Reg. Risk: %s  %s" % (progress_bar(LR_prob), LR_prob), "yellow"))
            bar = progress_bar(risk, filler=":fire: ", length=10)
            logger.info(colored(emoji.emojize("Current risk: %.4f %s" % (risk, bar)), 'red'))

    def save_all_data(self):
        # This saves all the recorded data including risk estimates
        logger.debug("Saving data to: %s ..." % os.path.basename(self.data_save_file))

        df = self.monitor.get_compiled_df()
        core_temperature_series = self.predictor.estimate_core_temperature(self.monitor.HR_stream, 37.6)

        # The DataFrame returned by the monitor not be large enough to hold all of the
        # risk series data so we need to make it bigger if necessary
        longest_risk_series = max(self.risk_series.size,
                                  self.CT_risk_series.size,
                                  self.HI_risk_series.size,
                                  self.LR_risk_series.size,
                                  core_temperature_series.size)

        num_to_append = longest_risk_series - df.shape[0]
        if num_to_append > 0:
            # Add a bunch of empty (NAN) values to the DataFrame is we need extra space for the risk values
            filler = np.empty()
            filler[:] = np.NAN
            df.append(filler)

        # Add the Risk/Heat Index/Estimated Core temperature data to the DataFrame
        df.loc[range(self.HI_stream.size), "time HI"] = self.HI_stream.keys()
        df.loc[range(self.HI_stream.size), "HI"] = self.HI_stream.values

        df.loc[range(core_temperature_series.size), "time est CT"] = core_temperature_series.keys()
        df.loc[range(core_temperature_series.size), "est CT"] = core_temperature_series.values

        df.loc[range(self.risk_series.size), "time Risk"] = self.risk_series.keys()
        df.loc[range(self.risk_series.size), "Risk"] = self.risk_series.values

        df.loc[range(self.HI_risk_series.size), "time HI Risk"] = self.HI_risk_series.keys()
        df.loc[range(self.HI_risk_series.size), "HI Risk"] = self.HI_risk_series.values

        df.loc[range(self.CT_risk_series.size), "time CT Risk"] = self.CT_risk_series.keys()
        df.loc[range(self.CT_risk_series.size), "CT Risk"] = self.CT_risk_series.values

        df.loc[range(self.LR_risk_series.size), "time LR Risk"] = self.LR_risk_series.keys()
        df.loc[range(self.LR_risk_series.size), "LR Risk"] = self.LR_risk_series.values

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


def progress_bar(progress, filler="=", length=16):
    # This function makes a string that looks like a progress bar
    # Example: progress of 0.62 would give the following string: "[======    ]"
    progress = 0 if progress is None else progress
    progress = 0 if progress < 0 else 1 if progress > 1 else progress
    return "[" + filler * int(0.5 + progress * length) + " " * (int(0.5 + (1 - progress) * length)) + "]"

def simulation(args):
    # This is for doing a simulation with data saved to file rather than read from the bean
    logger.error("Data simulation not yet implemented! Run without -S flag")

def pause_main_thread():
    # This function pauses this thread until the user tells the program to abort using the keyboard

    logger.warning("Pausing main thread ('q' + Enter to abort)...")
    try:
        while True:
            # This makes is so that the user can press any key on the keyboard
            # but it won't exit unless they KeyboardInterrupt the process
            user_input = input("")
            if user_input == 'q':
                logger.warning("Exit signal received. Terminating threads...")
                break
    except KeyboardInterrupt:
        logger.warning("Keyboard Interrupt. Terminating threads...")

def simulate(args):
    logger.error("Simulation not yet implemented!")

def run(args):
    # This function is the main function of this program that runs the real-time Heat Stroke risk process
    logger.info(emoji.emojize('Running test: %s ...' % __file__ + ' :fire:' * 3))
    logger.debug("Instantiating prediction handler...")
    handler = PredictionHandler(users_XML= args.users_XML, username=args.user, 
        output_dir=args.output, timestamp_files=args.timestamp_files, live_plotting=args.live_plotting, show_plots=args.show_plots, plotly=args.plotly)
    logger.debug(emoji.emojize("Prediction handler instantiated :heavy_check_mark:"))

    # Tell the prediction handler whether or not to use prefiltered data or to refilter it
    handler.predictor.use_prefiltered = args.prefiltered
    # Initialize the logistic regression predictor using the filtered data
    handler.predictor.init_log_reg_predictor()

    # Create all of the threads that the handler needs
    handler.initialize_threads(test=args.no_bean or args.test)

    # Start all of the threads    
    handler.start_all_threads()
    start_time = time.time()

    # Pause until the user quits the program
    pause_main_thread()

    # Save the data
    handler.save_all_data()
    # Stop the threads
    handler.stop_all_threads(wait=True)

    # Indicate that the test has finished
    runtime = time.time() - start_time
    m, s = divmod(runtime, 60)
    h, m = divmod(m, 60)
    time_str = "%d:%02d:%02d" % (h, m, s)
    logger.info(emoji.emojize("Test complete. (Duration: %s) :heavy_check_mark:" % time_str))

def main():
    # This function is basically just for parsing command line arguments and deciding what to do
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
    plotting_group.add_argument('-plot', '--live-plotting', dest="live_plotting", action="store_true", help="Make plots")
    plotting_group.add_argument('-show', '--show-plots', dest="show_plots", action="store_true", help="Display plots")
    plotting_group.add_argument('-plotly', '--plotly', action='store_true', help="Use Plotly to display plots")

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
        if args.real_time:
            logger.warning("-rf (\"--real-time\") flag for use with simulation (-S)")
        run(args)

if __name__ == "__main__":
    main()