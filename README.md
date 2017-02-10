# Heat_Stroke_Prediction

This is a repository for code used in Bioengineering Capstone at Stanford. (Bioe 141A/B)
Code in this repository is used for testing of methods for predicting heat stroke with a wearable device.

![screen shot 2017-01-31 at 6 53 22 pm](https://cloud.githubusercontent.com/assets/15920014/22493706/9a73f36a-e7e6-11e6-9809-7b0827663e36.png)

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
