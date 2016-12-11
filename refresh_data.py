#!/usr/bin/env python

import logging
import download_GSheets as gs

if __name__ == '__main__':
    logging.basicConfig(format='[%(funcName)s] - %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    key_file = "/Users/jonpdeaton/Downloads/HeatStroke-56ff10a91018.json"
    docid = "1kfRNddYhwTZq9K1ZGcfK21PeuzUmVBif-8-XB8dXfBY"
    destination = "/Users/jonpdeaton/Google Drive/school/BIOE 141A/Heat_Stroke_Prediction/data"
    logger.info("Getting Google Sheet...")
    logger.info("Docid: %s" % docid)
    gs.download_gsheets(key_file, docid, destination, name="hest_stroke")
    logger.info("Completed downloading Google Sheet")
