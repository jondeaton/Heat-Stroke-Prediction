#!/usr/bin/env python
'''
monitor.py

This script implements a class called HeatStrokeMonitor that handles reading data from a serial
port that is being written to by e blue bean, and handles the storage of that data.
'''

import time
import logging
import coloredlogs
import threading
import pandas as pd
import emoji
import serial
import warnings
import random

import numpy as np

coloredlogs.install(level='INFO')
logger = logging.getLogger(__name__)

__author__ = "Jon Deaton"
__email__ = "jdeaton@stanford.edu"


class SerialReadThread(threading.Timer):
    # This class is a thread for reading lines
    # from a serial port

    def __init__(self, ser, callback, verbose=False):
        threading.Thread.__init__(self)
        self.ser = ser
        self.callback = callback
        self.verbose = verbose
        self.bytes_read = 0
        self._is_running = False

    def run(self):
        while self._is_running:

            if self.ser is None:
                logger.error("No open serial port!")
                return
                
            try:
                line = self.ser.readline()
                self.bytes_read += len(line)
                if verbose:
                    logger.info("Read line: %s" % line)
                self.callback(line)
            except:
                pass

            time.sleep(0.01)

    # For stopping the thread
    def stop(self):
        self._is_running = False

class TestSerialReadThread(threading.Timer):
    # This class is for testing without Bluetooth connectivity

    def __init__(self, ser, callback, verbose=True):
        threading.Thread.__init__(self)
        self.ser = ser
        self.callback = callback
        self.verbose = verbose
        self.bytes_read = 0
        self._is_running = True
        self.time_started = time.time()

        # These are some fake values to use for testing
        self.test_funcs = {
        'HR': lambda t: 100 + (180-100)*(t - self.time_started) / (0.75 * 60 * 60),
        'ET': lambda t: 36 + 5 * np.random.random(),
        'EH': lambda t: 0.8 + 0.1 * np.random.random(),
        'ST': lambda t: 100 + (180-100)*(t - self.time_started) / (1.2 * 60 * 60),
        'GSR': lambda t: 400,
        'Acc': lambda t: 0.5,
        'SR': lambda t: 0.5}

    def run(self):
        while self._is_running:
            fields = ['HR', 'ET', 'EH', 'ST', 'GSR', 'Acc', 'SR']
            field = random.choice(fields)
            value = self.test_funcs[field](time.time())
            line = "{field}: {value}".format(field=field, value=value)
            logger.info("Read line: \"%s\"" % line)
            self.callback(line)
            self.bytes_read += len(line)
            time.sleep(0.3)

    # For stopping the thread
    def stop(self):
        self._is_running = False

class HeatStrokeMonitor(object):

    def __init__(self, port=None):
        self.init_time = time.time
        self.data_save_file = "monitor_data.csv"

        if port is None:
            self.serial_ports = ['/tmp/cu.LightBlue-Bean',
                                 '/tmp/tty.LightBlue-Bean', 
                                 '/dev/cu.LightBlue-Bean', 
                                 '/dev/cu.Bluetooth-Incoming-Port']
        else:
            self.serial_ports = [port]

        self.set_threading_class(test=False)
        self.checker_thread = None

        self.ser = None
        self.bytes_read = 0
        self.open_port()
        
        self.HR_stream = pd.Series()
        self.ETemp_stream = pd.Series()
        self.EHumid_stream = pd.Series()
        self.STemp_stream = pd.Series()
        self.GSR_stream = pd.Series()
        self.Acc_stream = pd.Series()
        self.Skin_stream = pd.Series()

        self.parameters = ["HR", "ET", "EH", "ST", "GSR", "Acc", "SR"]
        self.fields = np.ravel([["time %s" % param, param] for param in self.parameters])

    def open_port(self, port=None):
        ports = self.serial_ports if port is None else [port]
        for port in ports:
            try:
                self.ser = serial.Serial(port)
                logger.info(emoji.emojize("Opened port: %s successfully :heavy_check_mark:" % port))
                break
            except:
                logger.warning(emoji.emojize("Failed opening serial port: %s" % port))

        if self.ser is None:
            logger.error(emoji.emojize("Failed opening on all %d serial ports!" % len(ports)))

    def read_data_from_port(self, log=False):
        # This function starts a new thread that will continuously read data from the serial port
        self.read_thread = self.threading_class(self.ser, self.parse_incoming_line, verbose=log)
        self._is_running = True
        logger.debug("Current number of threads: %d" % threading.activeCount())
        logger.debug("Starting data read thread...")
        self.read_thread.start()
        logger.debug("Current number of threads: %d" % threading.activeCount())

    def stop_data_read(self):
        # This stopps the thread that is reading the data
        logger.debug("Sending stop signal to data read thread...")
        self.read_thread.stop()
        logger.debug("Stop signal sent. Threads: %d" % threading.activeCount())

    def parse_incoming_line(self, line):
        now = time.time()

        parsed_line = parse(line)
        
        # Check to make sure that it was parsed
        if parsed_line is None:
            return

        if line.startswith("HR:"): # Heart Rate
            self.HR_stream.set_value(now, parsed_line)
        elif line.startswith("ET:"): # Environmental Temperature
            self.ETemp_stream.set_value(now, parsed_line)
        elif line.startswith("EH:"): # Environmental Humidity
            self.EHumid_stream.set_value(now, parsed_line)
        elif line.startswith("ST:"): # Skin Temperature
            self.STemp_stream.set_value(now, parsed_line)
        elif line.startswith("GSR:"): # Galvanic Skin Response
            self.GSR_stream.set_value(now, parsed_line)
        elif line.startswith("Acc:"): # Acceleration
            self.Acc_stream.set_value(now, parsed_line)
        elif line.startswith("SR:"): # Skin Reflectivity...? LOL
            self.Skin_stream.set_value(now, parsed_line)
        else:
            logger.warning("No parse: %s" % line)

    def get_compiled_df(self):
        # This function puts data that has been gathered to a pandas DataFrame
        max_num_measurements = max([self.HR_stream.size, self.ETemp_stream.size, 
                                    self.EHumid_stream.size, self.STemp_stream.size,
                                    self.GSR_stream.size, self.Acc_stream.size, 
                                    self.Skin_stream.size])

        df = pd.DataFrame(columns = self.fields, index=range(max_num_measurements))

        df.loc[range(self.HR_stream.size), "time HR"] = self.HR_stream.keys()
        df.loc[range(self.HR_stream.size), "HR"] = self.HR_stream.values

        df.loc[range(self.ETemp_stream.size), "time ET"] = self.ETemp_stream.keys()
        df.loc[range(self.ETemp_stream.size), "ET"] = self.ETemp_stream.values

        df.loc[range(self.EHumid_stream.size), "time EH"] = self.EHumid_stream.keys()
        df.loc[range(self.EHumid_stream.size), "EH"] = self.EHumid_stream.values

        df.loc[range(self.STemp_stream.size), "time ST"] = self.STemp_stream.keys()
        df.loc[range(self.STemp_stream.size), "ST"] = self.STemp_stream.values

        df.loc[range(self.GSR_stream.size), "time GSR"] = self.GSR_stream.keys()
        df.loc[range(self.GSR_stream.size), "GSR"] = self.GSR_stream.values

        df.loc[range(self.Acc_stream.size), "time Acc"] = self.Acc_stream.keys()
        df.loc[range(self.Acc_stream.size), "Acc"] = self.Acc_stream.values

        df.loc[range(self.Skin_stream.size), "time SR"] = self.Skin_stream.keys()
        df.loc[range(self.Skin_stream.size), "SR"] = self.Skin_stream.values

        return df

    def save_data(self, file=None):
        # Saves the data collected here to a CSV file
        df = self.get_compiled_df()
        save_file = file if file is not None else self.data_save_file
        logger.debug("Saving data to: %s" % save_file)
        df.to_csv(save_file)

    def set_threading_class(self, test=False):
        # This function decides whether the testing thread class or the real serial port
        # thread class will be used 
        self.threading_class = SerialReadThread if not test else TestSerialReadThread


def parse(line):
    # This function parses a string read from serial of the form
    # <identifier: <value>
    # for example, a heart rate reading looks like:
    # HR: 121
    # which would result in a floating point return of 121.0 from this function
    try:
        value = float(line[2 + line.index(":"):])
    except:
        value = None
    return value

def test(args):
    logger.info("Testing: %s ..." % __file__)
    logger.debug("Instantiating HeatStrokeMonitor object...")
    monitor = HeatStrokeMonitor(port=args.port)
    monitor.set_threading_class(test=args.test)

    logger.info("Starting data reading (control-C to exit)...")
    # monitor.read_data_from_port(print=True)
    monitor.read_data_from_port(log=True)

    sec = 5
    logger.debug("Pausing for %d seconds..." % sec)  
    time.sleep(sec)

    monitor.stop_data_read()

    logger.info(emoji.emojize(":heavy_check_mark: Test complete - %d bytes read" % monitor.read_thread.bytes_read))

def main():
    import argparse
    script_description = "This script tests Heat stroke monitor monitor connectivity"
    parser = argparse.ArgumentParser(description=script_description)

    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument("-p", "--port", required=False, help="Serial port specification")

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
        test(args)

if __name__ == "__main__":
    main()