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

		self.fields = ["time HR", "HR", "time ET", "ET", "time ST", "ST",
		"time GSR", "GSR", "time Acc", "Acc", "time SR", "SR"]


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
        self.checker_thread = threading.Timer(0.01, self.serial_checker)
        self.checker_thread.start()

    def parse_incoming_line(self, line)
    	now = time.time()

    	parsed_line = parse(line)
    	# Check to make sure that it was parsed
    	if parsed_line is None:
    		return

    	if line.startswith("HR:"):
    		self.HR_stream.set_value(time, parsed_line)
    	elif line.startswith("ET:"):
    		self.ETemp_stream.set_value(time, parsed_line)
    	elif line.startswith("ST:"):
    		self.STemp_stream.set_value(time, parsed_line)
    	elif line.startswith("GSR:"):
    		self.GSR_stream.set_value(time, parsed_line)
    	elif line.startswith("Acc:"):
    		self.Acc_stream.set_value(time, parsed_line)
    	elif line.startswith("SR:"):
    		self.Skin_stream.set_value(time, parsed_line)
    	else:
    		logger.warning("No parse: %s" % line)

    def save_data(self, file=None):
    	df = pd.DataFrame(columns = self.fields)

    	df["time HR"].loc[0:self.HR_stream.size] = self.HR_stream.keys()
    	df["HR"].loc[0:self.HR_stream.size] = self.HR_stream.values

    	df["time ET"].loc[0:self.ETemp_stream.size] = self.ETemp_stream.keys()
    	df["ET"].loc[0:self.ETemp_stream.size] = self.ETemp_stream.values

    	df["time ST"].loc[0:self.STemp_stream.size] = self.STemp_stream.keys()
    	df["ST"].loc[0:self.STemp_stream.size] = self.STemp_stream.values

    	df["time GSR"].loc[0:self.GSR_stream.size] = self.GSR_stream.keys()
    	df["GSR"].loc[0:self.GSR_stream.size] = self.GSR_stream.values

    	df["time Acc"].loc[0:self.Acc_stream.size] = self.Acc_stream.keys()
    	df["Acc"].loc[0:self.Acc_stream.size] = self.Acc_stream.values

    	df["time SR"].loc[0:self.Skin_stream.size] = self.Skin_stream.keys()
    	df["SR"].loc[0:self.Skin_stream.size] = self.Skin_stream.values

    	save_file = file if file is not None else self.data_save_file
    	logger.info("Saving data to: %s" % save_file)
    	df.to_excel(save_file)
    	

def main():
	logger.info("Initiating HeatStrokeMonitor object...")
	monitor = HeatStrokeMonitor()
	logger.info("Initializing data reading...")
	monitor.read_data(print=True)

	logger.info("Reading data...")
	save_thread = threading.Timer(20, monitor.save_data)
	save_therad.start()


if __name__ == "__main__":
	main()