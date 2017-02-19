# Heat_Stroke_Prediction/data/

This directory contains data that is used for supervised learning and testing of the Heat Stroke Prediction Algorithm.

## Core Temperature Estimation
![test](https://cloud.githubusercontent.com/assets/15920014/23107769/21c1ae22-f6b8-11e6-90f3-63c299c9cffb.png)

Results from Heller Lab esophageal probe and heart rate monitor (used on Jon). Core temperature (left, grey) and heart rate (right, grey) were measured continuously throughout the non-exertional hyperthermia experiment using an esophageal probe and chest strap-mounted heart rate monitor. Core temperature initially decreased due to hypothalamic regulation. Heart rate increased throughout the experiment in an attempt to dissipate heat through the glabrous areas. Convolutional filtering was performed on both datasets to show general data trends and remove the presence of dips in core temperature measurement due to swallowing saliva. Several heart rate measurements were also made using our device’s pulse sensor on the finger at intervals of ~10 minutes (green, right). Estimations of core temperature from heart rate measurements were made using the Kalman Filter Model adapted from Buller et al. (2013) on the raw heart rate data from the chest strap (left, magenta), and the pulse monitor data after interpolation (left, green). The general upward trend of core temperature prediction from our pulse monitor readings and Heller Lab equipment matches those of the actual values, but the slopes are less steep.


## Supervised Learning Classification
<img width="517" alt="svm plots" src="https://cloud.githubusercontent.com/assets/15920014/23107806/cfb91920-f6b8-11e6-88ad-134be1713286.png">

The solid line in each plot shows an SVM classification of the data with margins (dotted lines). Many of the heat stroke data points are from true patient populations; however, all of the blue data is drawn from distributions of physiologically normal ranges. A) Environmental temperature vs. patient core temperature. B) Heart rate vs. patient temperature. Given that many points from both classifications lie on both sides of the line indicates that these two features alone may be insufficient in predicting heat stroke. C) Relative humidity vs. HI.
