#!/usr/bin/env python

import xml
import pandas as pd
import xml.etree.ElementTree as ET

class MonitorUser(object):

	def __init__(self, load=False):
		self.users_file = 'users.xml'
		
		self.series = pd.Series()
		self.name = None
		self.age = None
		self.sex = None
		self.weight = None
		self.height = None
		self.BMI = None
		self.nationality = None
		self.cardiovascular_disease_history = None
		self.sickle_cell = None

		if load:
			self.load_from_file(None)
			
	def update_series(self):
		self.series['Name'] = self.name
		self.series['Sex'] = self.sex
		self.series['Age'] = self.age
		self.series['Weight (kg)'] = self.weight
		self.series['BMI'] = self.BMI
		self.series['Height (cm)'] = self.height
		self.series['Nationality'] = self.nationality
		self.series['Cardiovascular disease history'] = self.cardiovascular_disease_history
		self.series['Sickle Cell Trait (SCT)'] = self.sickle_cell

	def load_from_file(self, username, users_file=None):
		
		# Decide which file to use
		if users_file is not None:
			xml_users_file = users_file
		else:
			xml_users_file = self.users_file
		
		tree = ET.parse(xml_users_file)
		users = tree.getroot()

		for user in users:
			if username is None or user['name'] == username:
				user_attributes = user.attrib
				break

		self.name = user_attributes['name']
		self.age = int(user_attributes['age'])
		self.sex = 1 if user_attributes['sex'] == "Male" else 0
		self.weight = float(user_attributes['weight'])
		self.height = float(user_attributes['height'])
		self.BMI = float(user_attributes['BMI'])
		self.nationality = 0 if user_attributes['nationality'] == "White" else 1
		self.cardiovascular_disease_history = int(user_attributes['cardiovascular_disease_history'])
		self.sickle_cell = int(user_attributes['sickle_cell'])


