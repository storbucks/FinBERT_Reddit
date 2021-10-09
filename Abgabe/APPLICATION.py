import csv
import time
import os

print(os.getcwd())  # make sure path is correct either here or in FUNCTIONS.py


csv.field_size_limit(1000000000)  # increase field size limit for raw input comment text

import FUNCTIONS


start_time = time.time()

slist = ['GME', 'CLOV', 'AMC', 'BABA', 'TSLA', 'AAPL', 'NVDA', 'BB']

#### scrape reddit between set dates ####
date_d = {7:31, 8:31, 9:30, 10:31, 11:30, 12:31}
for k, v in date_d.items():
    cs_time = time.time()
    FUNCTIONS.scraper(2020,k,1,2020,k,v)
    ce_time = time.time()
    print("Computation time: " + str((ce_time - cs_time)/3600) + "h")

#### filter per desired stock ####
for k, v in date_d.items():
    for s in slist:
        FUNCTIONS.stockfilter(s,2020,k)

#### classify each stock's comments using FinBERT ####
for s in slist:
    cs_time = time.time()
    FUNCTIONS.classify(s,2020,'Q3')
    ce_time = time.time()
    print("Computation time: " + str((ce_time - cs_time)/3600) + "h")


end_time = time.time()
print("Computation time: " + str((end_time-start_time)/3600) + "h")