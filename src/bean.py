#!/usr/bin/env python

__author__ = "Jon Deaton"

import sys
import serial
import datetime
import time

def main():
	ports = ['/tmp/tty.LightBlue-Bean', '/tmp/cu.LightBlue-Bean', '/dev/cu.LightBlue-Bean', '/dev/cu.Bluetooth-Incoming-Port']
	ser = None
	for port in ports:
		try:
			sys.stdout.write("Opening serial port: %s... " % port)
			
			ser = serial.Serial(port)
			sys.stdout.write("Succes\n")
		except:
			sys.stdout.write("Failure\n")
	if ser is None:
		exit()

	while True:
		batteryVoltage = str(ser.readline())
		print('Date: ' + str(datetime.datetime.now().date()))
		print('Time: ' + str(datetime.datetime.now().time()))
		print(batteryVoltage)
		time.sleep(10)


if __name__ == "__main__":
	main()