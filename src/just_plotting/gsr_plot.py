#!/usr/bin/env python
# This makes a plot

import logging
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from statsmodels.nonparametric.smoothers_lowess import lowess

import emoji
import coloredlogs
from termcolor import colored

coloredlogs.install(level='DEBUG')
logging.basicConfig(format='[%(levelname)s][%(funcName)s] - %(message)s')
logger = logging.getLogger(__name__)

__author__ = "Jon Deaton"
__email__ = "jdeaton@stanford.edu"

def smooth_data(y, x):
    filtered = lowess(y, x, is_sorted=True, frac=0.25, it=0)
    smooth_x = filtered[:, 0]
    smooth_y = filtered[:, 1]
    return (smooth_y, smooth_x)

def reject_outliers(sr, time_data, iq_range=0.4):
    pcnt = (1 - iq_range) / 2
    qlow, median, qhigh = sr.dropna().quantile([pcnt, 0.50, 1-pcnt])
    iqr = qhigh - qlow
    where = (sr - median).abs() <= iqr
    return (sr[where], time_data[where])

def plot_gsr_data(data, raw_time_data, axis, c='b', label=None, ls='solid'):
    start_time = min(raw_time_data)
    time_data = raw_time_data - start_time

    filtered_data, filtered_time = reject_outliers(data, time_data)
    smoothed_data, smoothed_time = smooth_data(filtered_data, filtered_time)

    #axis.scatter(time_data, data, c=c, s=0.5, alpha=0.05)
    axis.scatter(filtered_time, filtered_data, c=c, s=0.5, alpha=0.75, label=label)
    axis.plot(smoothed_time[50:], smoothed_data[50:], 'k', ls=ls)

def make_plot(hot_file, cold_file, output_file, show=False):
    from matplotlib import rcParams
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial']
    df_hot = pd.read_csv(hot_file)
    df_cold = pd.read_csv(cold_file)

    figure, axis = plt.subplots(1, 1)
    plot_gsr_data(df_hot['GSR'], df_hot['time GSR'] / 60, axis, label="Heat Stress GSR", c='b', ls='solid')
    plot_gsr_data(df_cold['GSR'], df_cold['time GSR'] / 60, axis, label="Control GSR", c='m', ls='dashed')

    axis.set_title("GSR vs. Time")
    axis.set_ylabel("Arbitrary")
    axis.set_xlabel("Time (minutes)")
    axis.set_ylim([0, 500])
    axis.legend()

    # axis.grid(True)
    if output_file is not None:
        figure.savefig(output_file)

    if show:
        logger.info("Showing teh plotz...")
        plt.show()


def main():
    import argparse
    
    script_description = "This script makes plots of a Heat-Stroke Experiment"
    parser = argparse.ArgumentParser(description=script_description)
    
    input_group = parser.add_argument_group("Inputs")
    input_group.add_argument('-hot', '--hot', required=True, help='Hot spreadsheet data')
    input_group.add_argument('-cold', '--cold', required=True, help="Cold spreadsheet data")

    output_group = parser.add_argument_group("Outputs")
    output_group.add_argument("-out", "--output", required=False, help="Plot outputs")

    parser.add_argument("-s", "--show", action="store_true", help="Display or not")

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

    make_plot(args.hot, args.cold, args.output, show=args.show)


if __name__ == '__main__':
    main()
