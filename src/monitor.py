#!/usr/bin/env python

import os
import time
import logging
import threading
import pandas as pd

logging.basicConfig(format='[%(levelname)s][%(funcName)s] - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

__author__ = "Jon Deaton"
__email__ = "jdeaton@stanford.edu"


# Parses a line in to a 
def parse(line):
	try:
		vlaue = float(line[line.index[":"]:])
	except:
		value = None
	return value

def HeatStrokeMonitor(object):

	def __init__(self):
		self.init_time = time.time
		self.data_save_file = "monitor_data.csv"

		self.serial_ports = ['/tmp/tty.LightBlue-Bean',
							 '/tmp/cu.LightBlue-Bean', 
							 '/dev/cu.LightBlue-Bean', 
							 '/dev/cu.Bluetooth-Incoming-Port']
		self.port = self.open_port()

		self.HR_stream = pd.Series()
		self.ETemp_stream = pd.Series()
		self.STemp_stream = pd.Series()
		self.GSR_stream = pd.Series()
		self.Acc_stream = pd.Series()
		self.Skin_stream = pd.Series()

	def open_port(self):
		self.port = None
		for port in self.serial_ports:
			try:
				sys.stdout.write("Trying serial port: %s... " % port)
				sys.stdout.flush()
				self.ser = serial.Serial(port)
				sys.stdout.write("succes!\n")
				break
			except:
				sys.stdout.write("failure\n")

	def read_data_from_port(self, print=False):
		if self.port = None:
			logger.error("No open serial port!")
			return
		try:
            line = self.ser.readline()
            if print:
            	logger.info("Read line: %s" % line)
            self.parse_incoming_line(line)
        except:
        	pass
        self.checker = threading.Timer(0.01, self.serial_checker)
        self.checker.start()

    def parse_incoming_line(self, line)
    	now = time.time()

    	parsed_line = parse(line)
    	# Check to make sure that it was parsed
    	if parsed_line is None:
    		return

    	if line.startswith("HR:"):
    		self.HR_stream[time] = parsed_line
    	elif line.startswith("ET:"):
    		self.ETemp_stream[time] = parsed_line
    	elif line.startswith("ST:"):
    		self.STemp_stream[time] = parsed_line
    	elif line.startswith("GSR:"):
    		self.GSR_stream[time] = parsed_line
    	elif line.startswith("Acc:"):
    		self.Acc_stream[time] = parsed_line
    	elif line.startswith("SR:"):
    		self.Skin_stream[time] = parsed_line
    	else:
    		logger.warning("No parse: %s" % line)

    def save_data(self, file=None):


def main():
	logger.info("Initiating HeatStrokeMonitor object...")
	monitor = HeatStrokeMonitor()
	logger.info("Initializing data reading...")
	monitor.read_data(print=True)


if __name__ == "__main__":
	main()