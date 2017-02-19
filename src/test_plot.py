#!/usr/bin/env python


import argparse
import logging
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

import emoji
import coloredlogs
from termcolor import colored

coloredlogs.install(level='DEBUG')
logging.basicConfig(format='[%(levelname)s][%(funcName)s] - %(message)s')
logger = logging.getLogger(__name__)

__author__ = "Jon Deaton"
__email__ = "jdeaton@stanford.edu"


def make_plot(data_file, output, show=False):

    df = pd.read_excel(data_file)

    temp_time = df["Elapsed (min)"]
    temp = df["TC1 (°C)"]

    hr_time = df["Time (min)"]
    HR = df["HR (bpm)"]

    logger.info("Making plots...")

    win = signal.hann(300)
    start = 200
    filtered_temp = (signal.convolve(temp, win, mode='same') / np.sum(win))[start:]
    filtered_temp_time = temp_time[start:]

    win = signal.hann(300)
    start = 200
    filtered_HR = (signal.convolve(HR, win, mode='same') / np.sum(win))[start:]
    filtered_HR_time = hr_time[start:]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    raw_alpha = 0.3

    # Temperature plot
    ax1.plot(temp_time, temp, 'k', alpha=raw_alpha, label="Probe Temperature")
    ax1.plot(filtered_temp_time, filtered_temp, 'r', label="Filtered Temperature")
    ax1.set_xlabel("Elapsed Time (min)", fontsize=18)
    ax1.set_ylabel("Core Temperature (°C)", fontsize=18)
    ax1.set_title("Core Temperature", fontsize=18)
    ax1.set_ylim([36.5, 39.5])
    ax1.set_xlim([0, 56])
    ax1.grid()
    ax1.annotate("Drink of water", xy=(5, 36.75), xytext=(30, 37), xycoords='data',textcoords='data',
            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=7),
            horizontalalignment='right', verticalalignment='top')

    ax1.legend(loc="upper left")

    # Heart Rate Plot
    ax2.plot(hr_time, HR, 'k', alpha=raw_alpha, label="Heart Rate")
    ax2.plot(filtered_HR_time, filtered_HR, 'b', label="Filtered Heart Rate")
    ax2.set_xlabel("Elapsed Time (min)", fontsize=18)
    ax2.set_ylabel("HR (bpm)", fontsize=18)
    ax2.set_title("Heart Rate", fontsize=18)
    ax2.set_xlim([0, 56])
    ax2.set_ylim([80, 145])
    ax2.grid()
    ax2.legend(loc="upper left")

    if output is not None:
        logger.info("Saving plot to (%s)..." % output)
        plt.savefig(output)
    if show:
        plt.show("Displaying plot...")
        plt.show()


def main():
    import argparse
    
    script_description = "This script makes plots of a Heat-Stroke Experiment"
    parser = argparse.ArgumentParser(description=script_description)
    
    input_group = parser.add_argument_group("Inputs")
    input_group.add_argument('-in', '--input', required=True, help='Input spreadsheet data')

    output_group = parser.add_argument_group("Outputs")
    output_group.add_argument("-out", "--output", required=False, help="Plot outputs")

    options_group = parser.add_argument_group("Opitons")
    options_group.add_argument("-s", "--show", action="store_true", help="Display plots with matplotlib")

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

    make_plot(args.input, args.output, show=args.show)


if __name__ == '__main__':
    main()
