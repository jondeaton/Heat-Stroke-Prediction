# Heat_Stroke_Prediction/src/

This file contains the main soruce code for the Heat Stroke Monitor prediction algorithm.

## Class Structure
![class_structure](https://cloud.githubusercontent.com/assets/15920014/23107814/e646ffc2-f6b8-11e6-9431-14a49f6e2d47.png)

Relationships between classes used in the software (implemented in Python) for Milestone 2. The MonitorUser class is a wrapper for patient demographic data. Static user data is stored in an XML file between runs, and the specific user can be specified on startup. HeatStrokeMonitor is a class that interfaces with the bluetooth Serial port through which data is transmitted from the physical monitor (sensor system), and also stores data retrieved from the data stream in time-associated tables. The PredictionHandler class (currently incomplete) periodically requests data from the HeatStrokeMonitor  class, combines this data with the demographic data from the MonitorUser class, and sends the combined data to the HeatStrokePredictor class. The HeatStrokePredictor class is designed to take user data streams and produce a risk assessment by retrieving case study data from the HeatStrokeDataFiller class.
