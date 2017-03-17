#!/usr/bin/env python
'''
plotter.py

This script implements the LivePlotter which is used to make real-time updated plots in MatplotLib
'''

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import threading
from statsmodels.nonparametric.smoothers_lowess import lowess

import warnings
import logging
import coloredlogs
import emoji
from termcolor import colored

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

class LivePlotter(object):
    # Class used to make live plots

    def __init__(self, data_file, output_directory=None, interval=15, show=False, plotly=False):

        # Input / Output
        self.data_file = data_file
        out_dir = output_directory if output_directory is not None else "."
        self.set_output_directory(out_dir)

        self.interval = interval
        self.show_plots = show
        self.looping_thread = LoopingThread(self.update_plot, self.interval)
        self.image_view_application = "/Applications/Google\ Chrome.app/"

    # Thread control
    def start_plotting(self, interval=None):
        # This function starts the looping thread responsible for updating the plots

        # It's a useful feature to be able to adjust interval here
        if interval is not None:
            self.looping_thread.wait_time = interval
            self.interval = interval

        self.start_time = time.time()
        self.looping_thread.start()

    def stop_plotting(self):
        # Close the window and stop the refresh thread
        self.looping_thread.stop()

    def set_output_directory(self, new_output_directory):
        # Use this to change the output directory
        self.output_directory = new_output_directory
        self.set_output_files()

    def set_output_files(self):
        # sets all the names of the plots
        self.temperature_plot_file = os.path.join(self.output_directory, "temp_data.svg")
        self.risk_plot_file = os.path.join(self.output_directory, "risk_data.svg")
        self.heart_rate_plot_file = os.path.join(self.output_directory, "heart_rate_data.svg")
        self.GSR_plot_file = os.path.join(self.output_directory, "GSR_data.svg")
        self.combined_file = os.path.join(self.output_directory, "combined_data.svg")

    def plot_temperature(self, figure, axis, save_to=None):
        # Makes a temperature plot in a particular figure and axis
        start_time = min(self.df['time ET'])
        axis.plot(self.df['time ET'] - start_time, self.df.ET, 'r-', label="Environmental Temperature")
        axis.plot(self.df['time est CT'] - start_time, self.df['est CT'], 'k-', label="Estimated CT")
        axis.set_title("Room/Core Temp")
        axis.set_ylabel("Temperature (C)")
        axis.set_xlabel("time")
        axis.legend()
        axis.grid(True)
        if save_to is not None:
            figure.savefig(save_to)

    def plot_heart_rate(self, figure, axis, save_to=None):
        start_time = min(self.df['time HR'])
        axis.plot(self.df['time HR'] - start_time, self.df.HR, '-r', label="Heart Rate")
        axis.set_title("Heart Rate")
        axis.set_ylabel("Beats Per Minute")
        axis.set_xlabel("Time")
        axis.legend()
        axis.grid(True)
        if save_to is not None:
            figure.savefig(save_to)

    def plot_GSR(self, figure, axis, save_to=None):
        # This method makes a plot of GSR data
        start_time = min(self.df['time GSR'])
        time_data = self.df['time GSR'] - start_time

        filtered_data, filtered_time = reject_outliers(self.df.GSR, time_data)
        smoothed_data, smoothed_time = smooth_data(filtered_data, filtered_time)

        axis.scatter(time_data, self.df.GSR, c='b', s=0.5, alpha=0.05)
        axis.scatter(filtered_time, filtered_data, c='b', s=0.5, label="GSR", alpha=0.75)
        axis.plot(smoothed_time, smoothed_data, 'k', label="filtered")

        axis.set_title("Galvanic Skin Response")
        axis.set_ylabel("Arbitrary")
        axis.set_xlabel("Time")
        axis.legend()
        axis.grid(True)
        if save_to is not None:
            figure.savefig(save_to)

    def plot_risk(self, figure, axis, save_to=None):
        # Makes a temperature plot in a particular figure and axis

        start_time = min(self.df['time Risk'])
        axis.plot(self.df['time Risk'] - start_time, self.df.Risk, '-r', label="Combined Risk")
        axis.plot(self.df['time HI Risk'] - start_time, self.df['HI Risk'], '-k', label="HI Risk")
        axis.plot(self.df['time CT Risk'] - start_time, self.df['CT Risk'], '-b', label="CT Risk")
        axis.plot(self.df['time LR Risk'] - start_time, self.df['LR Risk'], '-m', label="LR Risk")

        axis.set_title("Heat Stroke Risk")
        axis.set_ylabel("Risk (Probability)")
        axis.set_xlabel("Time")
        axis.legend()
        axis.set_ylim((0, 1))
        axis.grid(True)
        if save_to is not None:
            figure.savefig(save_to)

    def update_plot(self):
        # This function update the plot and should be called periodically by a LoopingThread so that
        # the plot appears to be a live feed of the data coming in

        logger.info("Updating plots...")
        self.df = pd.read_csv(self.data_file)

        # Make the combined figure
        combined_fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(9, 6), sharex=True)

        # Plot 1: Temperature
        temp_fig, temp_ax = plt.subplots(1, 1, figsize=(9, 6))
        self.plot_temperature(temp_fig, temp_ax, save_to=self.temperature_plot_file)
        plt.close()

        # Plot 2: Risk
        risk_fig, risk_ax = plt.subplots(1, 1, figsize=(9, 6))
        self.plot_risk(risk_fig, risk_ax, save_to=self.risk_plot_file)
        plt.close()

        # Plot 3: Heart Rate
        HR_fig, HR_ax = plt.subplots(1, 1, figsize=(9, 6))
        self.plot_heart_rate(HR_fig, HR_ax, save_to=self.heart_rate_plot_file)
        plt.close()

        # Plot 4: GSR
        GSR_fig, GSR_ax = plt.subplots(1, 1, figsize=(9, 6))
        self.plot_GSR(GSR_fig, GSR_ax, save_to=self.GSR_plot_file)
        plt.close()

        # Combined Plot
        self.plot_temperature(combined_fig, ax1)
        self.plot_risk(combined_fig, ax2)
        self.plot_heart_rate(combined_fig, ax3)
        self.plot_GSR(combined_fig, ax4)
        plt.savefig(self.combined_file)

        # Finally show it
        if self.show_plots:
            logger.info("Opening plot...")
            os.system("open %s -a %s" % (self.combined_file, self.image_view_application))

def smooth_data(y, x):
    filtered = lowess(y, x, is_sorted=True, frac=0.025, it=0)
    smooth_x = filtered[:, 0]
    smooth_y = filtered[:, 1]
    return (smooth_y, smooth_x)

def reject_outliers(sr, time_data, iq_range=0.4):
    pcnt = (1 - iq_range) / 2
    qlow, median, qhigh = sr.dropna().quantile([pcnt, 0.50, 1-pcnt])
    iqr = qhigh - qlow
    where = (sr - median).abs() <= iqr
    return (sr[where], time_data[where])

def run(args):

    if args.output is not None and not os.path.isdir(args.output):
        logger.info("Directory: %s did not exist. Creating..." % args.output)
        os.mkdir(args.output)

    plotter = LivePlotter(args.input, output_directory=args.output, show=args.show)
    logger.info("Making plots using: %s ..." % os.path.basename(args.input))
    plotter.update_plot()
    logger.info("Saved plots to: %s" % plotter.output_directory)

def main():
    import argparse
    script_description = "This script makes plots of Heat Stroke Data using Matplotlib"
    parser = argparse.ArgumentParser(description=script_description)

    input_group = parser.add_argument_group("Inputs")
    input_group.add_argument('-in', '--input', required=True, help='Input file with data')

    output_group = parser.add_argument_group("Outputs")
    output_group.add_argument("-out", "--output", required=False, help="Output directory to save plots in")

    options_group = parser.add_argument_group("Options")
    options_group.add_argument('-test', '--test', action="store_true", help="This does nothing at the moment")
    options_group.add_argument('-s', '--show', action="store_true", help="Show plots on screen.")

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

    if args.test:
        logger.warning("The --test flag does nothing.")
    if args.plotly:
        logger.warning("Plotly functionality not yet implemented")

    run(args)

if __name__ == '__main__':
    main()