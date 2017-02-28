#!/usr/bin/env python
'''
user.py

This script contains the implementation for the MonitorUser class, which loads and stores demographic information
about a HeatStrokeMonitor.
'''

import os
import xml
import pandas as pd
import xml.etree.ElementTree as ET

class MonitorUser(object):

	def __init__(self, users_XML=None, load=False, username=None):
		self.users_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", 'users.xml') if users_XML is None else users_XML
		
		self.series = pd.Series()
		self.name = None
		self.emoji = None
		self.age = None
		self.sex = None
		self.weight = None
		self.height = None
		self.BMI = None
		self.nationality = None
		self.cardiovascular_disease_history = None
		self.sickle_cell = None

		if load:
			self.load_from_file(username)

	def get_user_attributes(self):
		# This function makes a pandas Series containing all of the user attributes
		user_attributes = pd.Series()
		user_attributes.set_value('Age', self.age)
		user_attributes.set_value('Sex', self.sex)
		user_attributes.set_value('Weight (kg)', self.weight)
		user_attributes.set_value('Height', self.height)
		user_attributes.set_value('BMI', self.BMI)
		user_attributes.set_value('Nationality', self.nationality)
		user_attributes.set_value('Cardiovascular disease history', self.cardiovascular_disease_history)
		user_attributes.set_value('Sickle Cell Trait (SCT)', self.sickle_cell)
		return user_attributes
		
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
			if username is None or user.attrib['name'].lower() == username.lower():
				user_attributes = user.attrib
				break

		self.name = user_attributes['name']
		self.emoji = user_attributes['emoji']
		self.age = int(user_attributes['age'])
		self.sex = 1 if user_attributes['sex'] == "Male" else 0
		self.weight = float(user_attributes['weight'])
		self.height = float(user_attributes['height'])
		self.BMI = float(user_attributes['BMI'])
		self.nationality = 0 if user_attributes['nationality'] == "White" else 1
		self.cardiovascular_disease_history = int(user_attributes['cardiovascular_disease_history'])
		self.sickle_cell = int(user_attributes['sickle_cell'])