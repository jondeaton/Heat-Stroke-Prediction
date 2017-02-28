#!/usr/bin/env python
"""
download_GSheets.py

This script is for downloading a Google Sheets document
"""

import os
import csv
import gspread
import logging
from oauth2client.service_account import ServiceAccountCredentials

def download_gsheets(key_file, docid, output, name=None):
    scope = ['https://spreadsheets.google.com/feeds']
    credentials = ServiceAccountCredentials.from_json_keyfile_name(key_file, scope)
    client = gspread.authorize(credentials)
    try:
        spreadsheet = client.open_by_key(docid)
    except:
        print("Couldn't find Spread Sheet")
        print("docid: %s" % docid)
        exit(1)

    if name is None:
        name = docid

    for i, worksheet in enumerate(spreadsheet.worksheets()):
        filename = os.path.join(output, "{name}-worksheet{sheet}.csv".format(name=name, sheet=i))
        with open(filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(worksheet.get_all_values())

def get_docid(link):
    start_str = "spreadsheets/d/"
    end_str = "/edit#gid"
    if start_str not in link:
        return ""
    else:
        start = link.index(start_str) + len(start_str)
        if end_str in link:
            end = link.index(end_str)
        else:
            end = len(link)
    return link[start:end]

def main():
    import argparse

    logging.basicConfig(format='[%(funcName)s] - %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser("This script downloads Google Sheets")
    parser.add_argument('-key', '--key', help="Google API Developer Key file in json format")
    sheet_group = parser.add_argument_group("Sheet")
    sheet_group.add_argument('-docid', '--docid', help="Google Sheets docic")
    sheet_group.add_argument('-link', '--link', help="Google Sheets html link")
    outputs_group = parser.add_argument_group("Outputs")
    outputs_group.add_argument('-out', '--output', help="Directory to place files into")
    outputs_group.add_argument('-name', '--name', help="Name of ")
    args = parser.parse_args()

    if args.docid is None:
        if args.link is not None:
            docid = get_docid(args.link)
        else:
            print("Must provide either docid or link")
            exit()
    else:
        docid = args.docid

    logger.info("Getting Google Sheet...")
    logger.info("Docid: %s" % docid)
    download_gsheets(args.key, docid, args.output, name=args.name)
    logger.info("Completed downloading Google Sheet")

if __name__ == "__main__":
    main()
