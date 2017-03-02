#!/usr/bin/env python
'''
plotter.py

This script implements the LivePlotter which is used to make real-time updated plots in MatPlotLib
'''
import time
import numpy as np
import matplotlib.pyplot as plt
import threading

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

    def __init__(self, callback, wait_time, sleep_callback=None):
        threading.Thread.__init__(self)
        self.callback = callback
        self.wait_time = wait_time
        self.sleep_callback = time.sleep if sleep_callback is None else sleep_callback
        self._is_running = False

    def run(self):
        self._is_running = True
        while self._is_running:
            #self.sleep_callback(self.wait_time)
            time.sleep(self.wait_time)
            self.callback()

    def stop(self):
        # This stops the thread
        self._is_running = False

class LivePlotter(object):

    def __init__(self, handler, monitor, refresh_rate=1):

        self.handler = handler
        self.monitor = monitor
        self.looping_thread = LoopingThread(self.update_plot, pow(refresh_rate, -1), sleep_callback=plt.pause)

        self.start_time = 0

        self.plot_drawn = False
        self.draw_plot()
        self.plot_drawn = True

    # Thread control
    def start_plotting(self, refresh_rate=None):
        # Open the plotting window and start the plot updater thread

        # It's a useful feature to be able to adjust this here
        if refresh_rate is not None:
            self.looping_thread.wait_time = pow(refresh_rate, -1)

        self.start_time = time.time()
        plt.show()
        self.looping_thread.start()

    def stop_plotting(self):
        # Close the window and stop the refresh thread
        plt.close()
        self.looping_thread.stop()

    def draw_plot(self):
        # This function just makes a matplotlib figure that will be used to display the live data
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
        plt.ion()

        self.ax1.set_xlabel("time")
        self.ax2.set_xlabel("time")
        self.ax3.set_xlabel("time")
        self.ax4.set_xlabel("time")

        self.ax1.set_title("Room/Core Temp")
        self.ax1.set_ylabel("Temperature (C)")
        self.ax1.set_ylim((30, 60))
        self.ax1.grid(True)

        self.ax2.set_ylabel("")
        self.ax2.grid(True)


        self.ax3.grid(True)

        self.ax4.set_ylim((0, 1))
        self.ax4.grid(True)
        self.ax4.set_ylabel("Risk")


    def update_plot(self):
        # This function update the plot and should be called periodically by a LoopingThread so that 
        # the plot appears to be a live feed of the data coming in 

        # Updating plot 1 (Temperature)
        self.ax1.plot(np.array(self.monitor.ETemp_stream.keys()) - self.start_time, self.monitor.ETemp_stream.values, 'r-o')
        if self.handler.CT_stream is not None:
            self.ax1.plot(np.array(self.handler.CT_stream.keys()) - self.start_time, self.handler.CT_stream.values - self.start_time, 'k-o')


        # Updating plot 2 (Risk)
        colors = ('r', 'k', 'b', 'm')
        risk_seties_set = (self.handler.risk_series, self.handler.CT_risk_series, self.handler.HI_risk_series, self.handler.LR_risk_series)
        for color, series in zip(colors, risk_seties_set):
            if series is not None and series.size > 0:
                self.ax4.plot(np.array(series.keys()) - self.start_time, series.values, '%s-o' % color)

        # Finally draw it
        plt.draw()

def test(args):
    # This is a test of live plotting
    logger.debug("Testing...")
    N = 100
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(9, 6))

    ax1.set_xlabel("x1")
    ax1.set_ylabel("y1")

    ax2.set_xlabel("x2")
    ax2.set_ylabel("y2")
    
    ax3.set_xlabel("x3")
    ax3.set_ylabel("y3")
    
    ax4.set_xlabel("x4")
    ax4.set_ylabel("y4")
    
    plt.ion()

    try:
       for i in range(N):
        y = np.random.random()
        for ax in (ax1, ax2, ax3, ax4):
            ax.scatter(i, y)
        plt.pause(0.3)
    except KeyboardInterrupt:
        logger.warning("Keyboard Interrupted... quitting")
        exit()

def main():
    import argparse
    script_description = "This script makes live plots wiith matplotlib"
    parser = argparse.ArgumentParser(description=script_description)

    input_group = parser.add_argument_group("Inputs")
    input_group.add_argument('-in', '--input', required=False, help='Input')

    output_group = parser.add_argument_group("Outputs")
    output_group.add_argument("-out", "--output", required=False, help="Output")

    options_group = parser.add_argument_group("Opitons")
    options_group.add_argument('-test', '--test', action="store_true", help="Implementation testing")

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

    if args.test:
        test(args)
    else:
        logger.info("No action specified. Run this script with the --test flag to do a real-time plotting test")


if __name__ == '__main__':
    main()