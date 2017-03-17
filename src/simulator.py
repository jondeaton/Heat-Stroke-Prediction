#!/usr/bin/env python
"""
simulator.py

This script implements a class called HeatStrokeSimulator for simulating past recorded heat stroke data.
"""

import os
import time
import random
import threading
import logging
import coloredlogs
import numpy as np
import pandas as pd

coloredlogs.install(level='INFO')
logger = logging.getLogger(__name__)

__author__ = "Jon Deaton"
__email__ = "jdeaton@stanford.edu"

class SimulationThread(threading.Timer):
    # This class is for testing without Bluetooth connectivity

    def __init__(self, ser, callback, verbose=True):
        threading.Thread.__init__(self)
        self.ser = ser
        self.callback = callback
        self.verbose = verbose
        self.bytes_read = 0
        self._is_running = False
        self.time_started = time.time()

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

class HeatStrokeSimulator(object):

    def __init__(self, data_file):

        self.init_time = time.time()
        self.data_save_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "simulated_data.csv")

        # Class that is used to instantiate a thread that
        # will continuously read from the serial port
        self.threading_class = SimulationThread

        self.ser = None  # The serial port
        self.bytes_read = 0  # The number of bytes read

        self.HR_stream = pd.Series()
        self.ETemp_stream = pd.Series()
        self.EHumid_stream = pd.Series()
        self.STemp_stream = pd.Series()
        self.GSR_stream = pd.Series()
        self.Acc_stream = pd.Series()
        self.Skin_stream = pd.Series()

        self.parameters = ["HR", "ET", "EH", "ST", "GSR", "Acc", "SR"]
        self.fields = np.ravel([["time %s" % param, param] for param in self.parameters])

        self.data_file = data_file
        self.df = pd.read_csv(self.data_file)


    def read_data_from_port(self, log=False, port=None):
        # This function starts a new thread that will continuously read data from the serial port
        if self.threading_class is None:
            # Make sure we have a class specified for making data read threads
            self.set_threading_class(test=False)
        if self.ser is None:
            # Make sure we have an open serial port
            self.open_port(port=port)

        # Instantiate the thread for data reading
        logger.debug("Starting simulation thread...")
        self.read_thread.start()  # Start the thread

    def stop_data_read(self):
        # This stopps the thread that is reading the data
        self.read_thread.stop()
        logger.debug("Stopped data read: %d bytes read" % self.read_thread.bytes_read)


    def get_compiled_df(self):
        # This function puts data that has been gathered to a pandas DataFrame
        max_num_measurements = max([self.HR_stream.size, self.ETemp_stream.size,
                                    self.EHumid_stream.size, self.STemp_stream.size,
                                    self.GSR_stream.size, self.Acc_stream.size,
                                    self.Skin_stream.size])

        df = pd.DataFrame(columns=self.fields, index=range(max_num_measurements))

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
