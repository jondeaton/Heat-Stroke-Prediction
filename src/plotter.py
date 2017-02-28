#!/usr/bin/env python
'''
plotter.py

This script implements the LivePlotter which is used to make real-time updated plots in MatPlotLib
'''

import numpy as np
import matplotlib.pyplot as plt

import logging
import coloredlogs
import emoji
from termcolor import colored

coloredlogs.install(level='DEBUG')
logger = logging.getLogger(__name__)

__author__ = "Jon Deaton"
__email__ = "jdeaton@stanford.edu"


class LivePlotter(object):

    def __init__():
        self.figure = plt.figure()





def test(args):

    plt.axis([0, 10, 0, 1])
    plt.ion()

    for i in range(10):
        y = np.random.random()
        plt.scatter(i, y)
        plt.pause(0.05)

    while True:
        plt.pause(0.05)

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