#!/usr/bin/env python
'''
monitor.py

This script implements a class called HeatStrokeMonitor that handles reading data from a serial
port that is being written to by e blue bean, and handles the storage of that data.
'''

import os
import time
import serial
import random
import threading
import warnings
import logging
import coloredlogs
import emoji
import numpy as np
import pandas as pd

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
        self._is_running = True
        self.bytes_read = 0
        while self._is_running:

            # Check to make sure that a serial port is open
            if self.ser is None:
                logger.error("Cannot read data: No open serial port!")
                return

            # Try reading the next line from serial port
            try:
                line = self.ser.readline()
                self.bytes_read += len(line) # increment running number of read bytes
                if self.verbose:
                    logger.debug("Read line: %s" % line)
                self.callback(line) # Should be HeatStrokeMonitor.parse_incoming_line
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
        self._is_running = False
        self.time_started = time.time()

        # These are some fake values to use for testing
        self.test_funcs = {
        'HR': lambda t: 100 + (180 - 100)*(t - self.time_started) / (0.1 * 60 * 60),
        'ET': lambda t: 38 + 5 * np.random.random(),
        'EH': lambda t: 0.8 + 0.1 * np.random.random(),
        'ST': lambda t: 100 + (180-100)*(t - self.time_started) / (1.2 * 60 * 60),
        'GSR': lambda t: 100 + 100 * np.random.random(),
        'Acc': lambda t: 0.5,
        'SR': lambda t: 0.5}

    def run(self):
        self._is_running = True
        self.bytes_read = 0
        while self._is_running:
            fields = ['HR', 'ET', 'EH', 'ST', 'GSR', 'Acc', 'SR']
            field = random.choice(fields)
            value = self.test_funcs[field](time.time())
            line = "{field}: {value}".format(field=field, value=value)
            logger.debug("Read line: \"%s\"" % line)
            self.callback(line)
            self.bytes_read += len(line)
            time.sleep(0.3)

    # For stopping the thread
    def stop(self):
        self._is_running = False

class HeatStrokeMonitor(object):

    def __init__(self, port=None, open_port=False):
        self.init_time = time.time()
        self.data_save_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "monitor_data.csv")

        if port is None:
            # These are the default serial ports that will be attempted to be opened in this order
            self.serial_ports = ['/dev/cu.LightBlue-Bean',
                                 '/tmp/tty.LightBlue-Bean',
                                 '/tmp/cu.LightBlue-Bean']
        else:
            # If a particular port was specified, then only try opening that port
            self.serial_ports = [port]

        # Class that is used to instantiate a thread that
        # will continuously read from the serial port
        self.threading_class = None

        self.ser = None # The serial port
        self.bytes_read = 0 # The number of byts read
        if open_port:
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
            logger.error(emoji.emojize("Failed opening all %d serial ports!" % len(ports)))

    def read_data_from_port(self, log=False, port=None):
        # This function starts a new thread that will continuously read data from the serial port
        if self.threading_class is None:
            # Make sure we have a class specified for making data read threads
            self.set_threading_class(test=False)
        if self.ser is None:
            # Make sure we have an open serial port
            self.open_port(port=port)

        # Instantiate the thread for data reading
        self.read_thread = self.threading_class(self.ser, self.parse_incoming_line, verbose=log)
        logger.debug("Current number of threads: %d" % threading.activeCount())
        logger.debug("Starting data read thread...")
        self.read_thread.start() # Start the thread
        logger.debug("Current number of threads: %d" % threading.activeCount())

    def stop_data_read(self):
        # This stopps the thread that is reading the data
        self.read_thread.stop()
        logger.debug("Stopped data read: %d bytes read" % self.read_thread.bytes_read)

    def parse_incoming_line(self, line):
        now = time.time()
        line = line.strip()

        # The line may be formatted as bytes and if so it needs to be turned into a string
        if type(line) is bytes:
            line = line.decode("utf-8")

        parsed_line = parse(line)
        
        # Check to make sure that it was parsed
        if parsed_line is None:
            return

        if line.startswith("HR:"): # Heart Rate
            self.HR_stream.set_value(now, parsed_line)
        elif line.startswith("ET:"): # Environmental Temperature
            self.ETemp_stream.set_value(now, parsed_line)
        elif line.startswith("EH:"): # Environmental Humidity
            # Make sure that humidity is in ratio format not percentile format...
            parsed_line = parsed_line / 100.0 if parsed_line > 1 else parsed_line
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
        self.threading_class = TestSerialReadThread if test else SerialReadThread
        logger.debug("Set threading class: %s" % self.threading_class.__name__)

def parse(line):
    # This function parses a string read from serial of the form
    # <identifier: <value>
    # for example, a heart rate reading looks like:
    # HR: 121
    # which would result in a floating point return of 121.0 from this function
    try:
        return float(line[2 + line.index(":"):])
    except:
        logger.error("Malfored line: %s" % line)
        return None

def test(args):
    logger.info("Testing: %s ..." % __file__)
    logger.debug("Instantiating HeatStrokeMonitor object...")
    monitor = HeatStrokeMonitor(port=args.port)
    monitor.set_threading_class(test=args.test)

    logger.info("Starting data reading (control-C to exit)...")
    monitor.read_data_from_port(log=True)

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

    monitor.stop_data_read()

    logger.info(emoji.emojize(":heavy_check_mark: Test complete - %d bytes read" % monitor.read_thread.bytes_read))

def main():
    import argparse
    script_description = "This script tests Heat stroke monitor monitor connectivity"
    parser = argparse.ArgumentParser(description=script_description)

    parser.add_argument("--test", action="store_true", help="Run a test with fake data not read from seial port")
    parser.add_argument("-p", "--port", required=False, help="Specify the serial port")

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
        # LOL
        test(args)

if __name__ == "__main__":
    main()
