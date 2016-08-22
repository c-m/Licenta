"""
Read moodle logs from .csv files in order to construct the "moodle logs" 
dataset. The purpose is to build a single .csv file from all the given 
/data_sets/moodle_logs/*.csv files, so we won't have to manually extract 
the data from them.
"""

import csv
import numpy as np
import os

DATA_PATH = '../data_sets/moodle_logs/'

class Keywords(object):

    ALL = 'all'
    ACTIVE_DAYS = 'active_days'
    COUNT = 'count'
    HW = 'hw'
    RES = 'res'
    POSTED = 'posted'
    VIEWED = 'viewed'

    def __init__(self):
        pass


def main():
    pass


if __name__ == '__main__':
    main()
