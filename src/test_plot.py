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

def estimate_core_temperature(heart_rate_series, CTstart):
        '''
        heart_rate_series: A pandas Series mapping time --> heart rate, ideally once per minute
        CTstart: A core-temperature starting value

        Kalman Filter Model adapted from Buller et al.
        Source: Buller MJ et al. “Estimation of human core temperature from sequential 
        heart rate observations.” Physiol Meas. 2013 Jul;34(7):781-98. doi: 10.1088/0967-3334/34/7/781. Epub 2013 Jun 19.
        '''
        # Extended Kalman Filter Parameters
        a = 1;
        gamma = pow(0.022, 2)

        b_0 = -7887.1
        #b_0 = -5000
        b_1 = 384.4286
        b_2 = -4.5714
        sigma = pow(18.88, 2)

        # Initialize Kalman filter
        x = CTstart
        v = 0 # v = 0 assumes confidence with start value

        core_temp_series = pd.Series()

        # Iterate through HR time sequence
        for time in heart_rate_series.keys():

            # Time Update Phase
            x_pred = a * x # Equation 3
            v_pred = pow(a, 2) * v + gamma # Equation 4

            #Observation Update Phase
            z = heart_rate_series[time]
            c_vc = 2 * b_2 * x_pred + b_1 # Equation 5
            k = (v_pred * c_vc) / (pow(c_vc, 2) * v_pred + sigma) # Equation 6
            x = x_pred + k * (z - (b_2 * pow(x_pred, 2) + b_1 * x_pred + b_0)) # Equation 7
            v = (1 - k * c_vc) * v_pred # Equation 8
            core_temp_series.set_value(time, x)

        return core_temp_series

def make_plot2(data_file, output, show=False):
    df = pd.read_excel(data_file)

    temp_time = df["Elapsed (min)"]
    temp = df["TC1 (°C)"]

    hr_time = df["Time (min)"]
    HR = df["HR (bpm)"]

    logger.info("Making estimation plot...")

    win = signal.hann(300)
    start = 200
    filtered_temp = (signal.convolve(temp, win, mode='same') / np.sum(win))[start:]
    filtered_temp_time = temp_time[start:]

    fig, ax1 = plt.subplots(1, 1, figsize=(7, 5))

    raw_alpha = 0.3

    # Temperature plot
    ax1.plot(temp_time, temp, 'k', alpha=raw_alpha, label="Probe Temperature")
    ax1.plot(filtered_temp_time, filtered_temp, 'r', label="Filtered Temperature")
    ax1.plot(core_temp_series.keys(), core_temp_series.values, 'g', label="Estimated Core Temperature")

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

    if output is not None:
        logger.info("Saving plot to (%s)..." % output)
        plt.savefig(output)
    if show:
        plt.show("Displaying plot...")
        plt.show()

def make_plot(data_file, output, show=False):

    df = pd.read_excel(data_file)

    temp_time = df["Elapsed (min)"]
    temp = df["TC1 (°C)"]

    hr_time = df["Time (min)"]
    HR = df["HR (bpm)"]

    logger.info("Making CT/HR plots...")

    start_temp = 37.3

    win = signal.hann(300)
    start = 200
    filtered_temp = (signal.convolve(temp, win, mode='same') / np.sum(win))[start:]
    filtered_temp_time = temp_time[start:]

    df['filtered TC1 (°C)'] = signal.convolve(temp, win, mode='same') / np.sum(win)
    df.to_excel("filtered_temp_data.xlsx")

    win = signal.hann(300)
    start = 200
    filtered_HR = np.array(signal.convolve(HR, win, mode='same') / np.sum(win))[start:]
    filtered_HR_time = np.array(hr_time[start:])

    heart_rate_series = pd.Series()
    predict_with_filtered = False
    if predict_with_filtered:
        for i in range(len(filtered_HR_time)):
            heart_rate_series.set_value(filtered_HR_time[i], filtered_HR[i])
    else:
        for i in range(df.shape[0]):
            heart_rate_series.set_value(hr_time[i], HR[i])

    core_temp_series = estimate_core_temperature(heart_rate_series, start_temp)


    bean_time = [0, 0.5, 1, 1.5, 2, 8.5, 9, 9.5, 10, 10.5, 18, 18.5, 19, 19.5, 20, 28, 28.5, 29, 29.5, 30, 39.5, 40][:-2]
    bean_hr = [103, 104, 97, 96, 95, 96, 96, 97, 98, 90, 101, 98, 100, 102, 103, 114, 113, 109, 109, 112, 126, 141][:-2]
    from scipy.interpolate import interp1d
    f = interp1d(bean_time, bean_hr)
    N = 50
    bean_interp_time = np.linspace(bean_time[0], bean_time[-1], num=N)
    bean_interp_hr = f(bean_interp_time)

    heart_rate_series2 = pd.Series()
    for i in range(N):
        heart_rate_series2.set_value(bean_interp_time[i], bean_interp_hr[i])

    core_temp_seires2 = estimate_core_temperature(heart_rate_series2, start_temp)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    raw_alpha = 0.3

    # Temperature plot
    ax1.plot(temp_time, temp, 'k', alpha=raw_alpha, label="Probe Temperature")
    ax1.plot(filtered_temp_time, filtered_temp, 'b', label="Filtered Probe Temperature")
    ax1.plot(core_temp_series.keys(), core_temp_series.values, 'm', label="Estimated Core Temperature")
    ax1.plot(core_temp_seires2.keys(), core_temp_seires2.values, 'g', label="Estimated from Pulse Monitor")
    ax1.set_xlabel("Elapsed Time (min)", fontsize=18)
    ax1.set_ylabel("Core Temperature (°C)", fontsize=18)
    ax1.set_title("Core Temperature", fontsize=18)
    ax1.set_ylim([36.5, 39.5])
    ax1.set_xlim([0, 56])
    ax1.grid()
    ax1.annotate("Drink of water", xy=(5, 36.75), xytext=(30, 37), xycoords='data',textcoords='data',
            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=7),
            horizontalalignment='right', verticalalignment='top')

    ax1.legend(loc="upper left", fontsize=15)

    # Heart Rate Plot
    ax2.plot(hr_time, HR, 'k', alpha=raw_alpha, label="Chest Strap HR")
    ax2.plot(filtered_HR_time, filtered_HR, 'b', label="Filtered Heart Rate")
    ax2.plot(bean_time, bean_hr, 'go', label="Pulse Monitor HR")
    ax2.set_xlabel("Elapsed Time (min)", fontsize=18)
    ax2.set_ylabel("HR (bpm)", fontsize=18)
    ax2.set_title("Heart Rate", fontsize=18)
    ax2.set_xlim([0, 56])
    ax2.set_ylim([80, 145])
    ax2.grid()
    ax2.legend(loc="upper left", fontsize=15)

    if output is not None:
        logger.info("Saving plot to (%s)..." % output)
        plt.savefig(output)
    if show:
        logger.info("Displaying plot...")
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
