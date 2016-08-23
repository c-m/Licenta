"""
Read moodle logs from .csv files in order to construct the "moodle logs" 
dataset. The purpose is to build a single .csv file from all the given 
/data_sets/moodle_logs/*.csv files, so we won't have to manually extract 
the data from them.
"""

import csv
import glob
import numpy as np
import os

from collections import OrderedDict
from data_loader import DatasetContaier

DATA_PATH = '../data_sets/moodle_logs/'
OUTPUT_DATASET = '../logs_pp.csv'

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


def read_logs():

    all_dataset = dict()

    program_path = os.path.abspath(os.path.dirname(__file__))
    dataset_path = os.path.join(program_path, DATA_PATH, '*.csv')
    log_files = glob.glob(dataset_path)

    viewed_to_posted_ratio = 40.0

    for f in log_files:
        if Keywords.HW in f and Keywords.POSTED in f:
            with open(f) as csv_file:
                data = csv.reader(csv_file)
                keys = next(data)

                for i, ex in enumerate(data):
                    if ex[0] in all_dataset:
                        all_dataset[ex[0]]['x1'] += int(ex[1])
                    else:
                        all_dataset[ex[0]] = {'x1': int(ex[1]), 'x2': 0, 'x3': 0, 'x4': 0}
        if Keywords.HW in f and Keywords.VIEWED in f:
            with open(f) as csv_file:
                data = csv.reader(csv_file)
                keys = next(data)
                for i, ex in enumerate(data):
                    if ex[0] in all_dataset:
                        all_dataset[ex[0]]['x1'] += int(ex[1])/viewed_to_posted_ratio
                    else:
                        all_dataset[ex[0]] = {'x1': int(ex[1])/viewed_to_posted_ratio, 'x2':0, 'x3': 0, 'x4': 0}
        if Keywords.ACTIVE_DAYS in f:
            with open(f) as csv_file:
                data = csv.reader(csv_file)
                keys = next(data)

                for i, ex in enumerate(data):
                    if ex[0] in all_dataset:
                        all_dataset[ex[0]]['x2'] = int(ex[1])
                    else:
                        all_dataset[ex[0]] = {'x1': 0, 'x2': int(ex[1]), 'x3': 0, 'x4': 0}
        if Keywords.RES in f:
            with open(f) as csv_file:
                data = csv.reader(csv_file)
                keys = next(data)

                for i, ex in enumerate(data):
                    if ex[0] in all_dataset:
                        all_dataset[ex[0]]['x3'] += int(ex[1])
                    else:
                        all_dataset[ex[0]] = {'x1': 0, 'x2': 0, 'x3': int(ex[1]), 'x4': 0}
        if Keywords.ALL in f and Keywords.ACTIVE_DAYS not in f:
            with open(f) as csv_file:
                data = csv.reader(csv_file)
                keys = next(data)

                for i, ex in enumerate(data):
                    if ex[0] in all_dataset:
                        all_dataset[ex[0]]['x4'] += int(ex[1])
                    else:
                        all_dataset[ex[0]] = {'x1': 0, 'x2': 0, 'x3': 0, 'x4': int(ex[1])}

    sorted_dataset = OrderedDict(sorted(all_dataset.items()))
    del all_dataset

    return sorted_dataset


def write_logs(moodle_logs):
    
    program_path = os.path.abspath(os.path.dirname(__file__))
    output_path = os.path.join(program_path, DATA_PATH, OUTPUT_DATASET)

    with open(output_path, 'wb') as csv_file:
        fieldnames = ['full_name', 'x1', 'x2', 'x3', 'x4']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for k, v in moodle_logs.iteritems():
            writer.writerow({'full_name': k, 'x1': float('%.3f' % v['x1']), 'x2': v['x2'], 'x3': v['x3'], 'x4': v['x4']})


def main():
    moodle_logs = read_logs()
    write_logs(moodle_logs)


if __name__ == '__main__':
    main()
