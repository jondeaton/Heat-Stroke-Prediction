#!/usr/bin/env python

import xml
import pandas as pd
import xml.etree.ElementTree as ET

def MonitorUser(object):

	def __init__(self):
		self.users_file = 'users.xml'

		self.name = None
		self.age = None
		self.sex = None
		self.weight = None
		self.height = None
		self.BMI = None
		self.nationality = None
		self.cardiovascular_disease_history = None
		self.sickle_cell = None

	def update_series(self):


	def load_from_file(self, username, users_file=None):
		
		# Decide which file to use
		if users_file is not None:
			xml_users_file = users_file
		else:
			xml_users_file = self.users_file
		
		tree = ET.parse(xml_users_file)
		root = tree.getroot()
			
		xml_user = None
		for child in root:
			if child.attrib['name'] == username
				xml_user = child

		user_attributes = xml_user.attrib
		self.name = user_attributes['name']
		self.age = user_attributes['age']
		self.sex = user_attributes['sex']
		self.weight = user_attributes['weight']
		self.height = user_attributes['height']
		self.BMI = user_attributes['BMI']
		self.nationality = user_attributes['nationality']
		self.cardiovascular_disease_history = user_attributes['cardiovascular_disease_history']
		self.sickle_cell = user_attributes['sickle_cell']
