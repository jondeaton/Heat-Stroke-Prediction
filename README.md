# Heat_Stroke_Prediction

This is a repository for code used in Bioengineering Capstone at Stanford. (Bioe 141A/B)
Code in this repository is used for testing of methods for predicting heat stroke with a wearable device.

## System Diagram
![screen shot 2017-01-31 at 6 53 22 pm](https://cloud.githubusercontent.com/assets/15920014/22493706/9a73f36a-e7e6-11e6-9809-7b0827663e36.png)

We intend to implement a prototype that senses relevant parameters and need not necessarily be wearable, minimally intrusive, or connect wirelessly to a phone, as we aim to prove concept viability rather than produce a ready-to-use device. Our concept (Figure 2) involves sensors connected to a microprocessor that relays information to a computer, which predicts heat stroke risk using machine learning (ML) algorithms. By the en dof BIOE 141B, we aim to have developed hardware and software that allow us to accurately sense and predict heat stroke, as well as a series of experimental tests verifying that our device functions as expected.


## Class Structure
![class_structure](https://cloud.githubusercontent.com/assets/15920014/23107814/e646ffc2-f6b8-11e6-9431-14a49f6e2d47.png)

Relationships between classes used in the software (implemented in Python) for Milestone 2. The MonitorUser class is a wrapper for patient demographic data. Static user data is stored in an XML file between runs, and the specific user can be specified on startup. HeatStrokeMonitor is a class that interfaces with the bluetooth Serial port through which data is transmitted from the physical monitor (sensor system), and also stores data retrieved from the data stream in time-associated tables. The PredictionHandler class (currently incomplete) periodically requests data from the HeatStrokeMonitor  class, combines this data with the demographic data from the MonitorUser class, and sends the combined data to the HeatStrokePredictor class. The HeatStrokePredictor class is designed to take user data streams and produce a risk assessment by retrieving case study data from the HeatStrokeDataFiller class.


## Core Temperature Estimation
![test](https://cloud.githubusercontent.com/assets/15920014/23107769/21c1ae22-f6b8-11e6-90f3-63c299c9cffb.png)

Results from Heller Lab esophageal probe and heart rate monitor (used on Jon). Core temperature (left, grey) and heart rate (right, grey) were measured continuously throughout the non-exertional hyperthermia experiment using an esophageal probe and chest strap-mounted heart rate monitor. Core temperature initially decreased due to hypothalamic regulation. Heart rate increased throughout the experiment in an attempt to dissipate heat through the glabrous areas. Convolutional filtering was performed on both datasets to show general data trends and remove the presence of dips in core temperature measurement due to swallowing saliva. Several heart rate measurements were also made using our device’s pulse sensor on the finger at intervals of ~10 minutes (green, right). Estimations of core temperature from heart rate measurements were made using the Kalman Filter Model adapted from Buller et al. (2013) on the raw heart rate data from the chest strap (left, magenta), and the pulse monitor data after interpolation (left, green). The general upward trend of core temperature prediction from our pulse monitor readings and Heller Lab equipment matches those of the actual values, but the slopes are less steep.


## Supervised Learning Classification
<img width="517" alt="svm plots" src="https://cloud.githubusercontent.com/assets/15920014/23107806/cfb91920-f6b8-11e6-88ad-134be1713286.png">

The solid line in each plot shows an SVM classification of the data with margins (dotted lines). Many of the heat stroke data points are from true patient populations; however, all of the blue data is drawn from distributions of physiologically normal ranges. A) Environmental temperature vs. patient core temperature. B) Heart rate vs. patient temperature. Given that many points from both classifications lie on both sides of the line indicates that these two features alone may be insufficient in predicting heat stroke. C) Relative humidity vs. HI.



## Dependencies

1. Python 3.5
2. Anaconda 3+
3. gspread  + oath2client (For Google Sheet Download)
	
	`sudo pip install gspread && sudo pip install oauth2client`

    Note: must also specity cryptography 1.4 from pip like so:
    `cryptography=1.4`
    in requirements.txt

5. meteocalc (for calculating Heat Index)
    
    `pip install meteocalc`

6. termcolor (For pretty terminal colors)
    
    `pip install termcolor`

7. emoji (yes seriously)

    `pip install emoji --upgrade`
