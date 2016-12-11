#!/usr/bin/env python
"""
refresh_data.py

This script is for refreshing the local data stored in a CSV from the Google Sheet that contains 
our data that all members of the team can access
"""

import logging
import download_GSheets

if __name__ == '__main__':
    logging.basicConfig(format='[%(funcName)s] - %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    key_file = "/Users/jonpdeaton/Downloads/HeatStroke-56ff10a91018.json"
    docid = "1kfRNddYhwTZq9K1ZGcfK21PeuzUmVBif-8-XB8dXfBY"
    destination = "/Users/jonpdeaton/Google Drive/school/BIOE 141A/Heat_Stroke_Prediction/data"
    logger.info("Getting Google Sheets...")
    logger.info("Docid: %s" % docid)
    download_GSheets.download_gsheets(key_file, docid, destination, name="hest_stroke")
    logger.info("Completed downloading Google Sheets.")
