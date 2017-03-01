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

        self.loop_wait = 1 if wait_time > 1 else wait_time
        self.num_loops = int(wait_time / self.loop_wait)
        self._is_running = True

    def run(self):
        while self._is_running:
            for _ in range(self.num_loops):
                # Check to make sure the thread should still be running
                if not self._is_running: return
                self.sleep_callback(self.loop_wait)
            self.callback()

    def stop(self):
        # This stops the thread
        self._is_running = False


class LivePlotter(object):

    def __init__(self, handler, refresh_rate=1):

        self.handler = handler

        self.looping_thread = LoopingThread(self.update_plot, pow(refresh_rate, -1), sleep_callback=plt.pause)

        self.plot_drawn = False
        self.draw_plot()
        self.plot_drawn = True


    def draw_plot(self):
        # Hello?
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(9, 6))
        plt.ion()

    def update_plot(self):

        ax1.plot(self.handler.)
        logger.warning("LivePlotter.update_plot() called... not implemented")




def test(args):

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
        
        # plt.draw()
        # plt.show()
        # time.sleep(0.3)
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


if __name__ == '__main__':
    main()