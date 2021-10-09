# Imports
import pandas as pd
import torch
import csv
import ast
import matplotlib.pyplot as plt
import datetime as dt
from psaw import PushshiftAPI
from transformers import BertTokenizer, BertForSequenceClassification

## scrape raw comment data
def scraper(ys, ms, ds, ye, me, de):
    import csv
    csv.field_size_limit(100000000)

    start_date = dt.date(ys, ms, ds)
    end_date = dt.date(ye, me, de)
    delta = dt.timedelta(days=1)

    yesterdays = []
    todays = []

    while start_date <= end_date:
        yesterday = start_date
        today = yesterday + delta
        yesterdays.append(start_date)
        todays.append(today)
        start_date += delta

    subreddits = ['stocks' , 'stockmarket', 'investing', 'robinhood', 'gme', 'amcstock']
    # subreddits = ['wallstreetbets']

    api = PushshiftAPI()

    # scrape per day and add api generator object to list
    c_list = []
    for d1, d2 in zip(yesterdays, todays):
        c = list(api.search_comments(after=d1, before=d2, subreddit=subreddits, limit=None))
        c_list.append(c)

    # access to comment body of each generator object, store as list of string-> add each list in c_list to dict
    c_body = []
    for l in range(len(c_list)):
        a = c_list[l]
        intermediate = []
        for k in range(len(a)):
            b = a[k]
            intermediate.append(b.body)
        c_body.append(intermediate)

    # append dictionary with comment body
    in_dict = {i: [] for i in yesterdays}

    for d, item in zip(in_dict, c_body):
        in_dict[d].append(item)

    # flatten dictionary value
    for ent in in_dict:
        a = in_dict[ent]
        flat_a = [item for sublist in a for item in sublist]
        in_dict[ent] = flat_a

    in_sum = []
    for ent in in_dict:
        in_sum.append(len(in_dict[ent]))
    print("Comments scraped:", sum(in_sum))

    ## Export In_dict to CSV
    import csv

    with open(f'{ys}_{ms}_raw_dict.csv', 'w') as f:  # set path as desired
        writer = csv.writer(f)
        for k, v in in_dict.items():
            writer.writerow([k, v])

    print('Success!!')

## Import raw comment files
def rawimport(year, month):

    # import csv
    # import ast
    csv.field_size_limit(100000000)

    # Import raw dict
    with open(f'{year}_{month}_raw_dict.csv', mode='r') as inp:
        reader = csv.reader(inp)
        raw_dict = {rows[0]: rows[1] for rows in reader}

    # Transform string of list to list and replace Dict value
    for entry in raw_dict:
        x = raw_dict[entry]
        x = ast.literal_eval(x)
        raw_dict[entry] = x

    # Number of scraped comments total
    in_sum = []
    for ent in raw_dict:
        in_sum.append(len(raw_dict[ent]))

    return raw_dict

## filter for stock
def stockfilter(stock, year, month):

    # import csv
    csv.field_size_limit(100000000)

    # Define filter criteria based on stock:
    if stock == 'GME':
        stocklist = ['GME', 'gme', 'Gamestop', 'gamestop', '$GME']
    elif stock == 'CLOV':
        stocklist = ['CLOV', 'clov', 'Clover', '$CLOV']
    elif stock == 'AMC':
        stocklist = ['AMC', 'amc', '$AMC']
    elif stock == 'BABA':
        stocklist = ['BABA', 'baba', 'Alibaba', '$BABA']
    elif stock == 'WISH':
        stocklist = ['WISH', 'wish', 'ContextLogic', '$WISH']
    elif stock == 'PLTR':
        stocklist = ['PLTR', 'pltr', 'Palantir', '$PLTR']
    elif stock == 'TSLA':
        stocklist = ['TSLA', 'tsla', 'Tesla', '$TSLA']
    elif stock == 'AAPL':
        stocklist = ['AAPL', 'aapl', 'Apple', '$AAPL']
    elif stock == 'NVDA':
        stocklist = ['NVDA', 'nvda', 'Nvidia', '$NVDA']
    elif stock == 'BB':
        stocklist = ['BB', 'bb', 'Blackberry', '$BB']
    elif stock == 'MSFT':
        stocklist = ['MSFT', 'msft', 'Microsoft', '$MSFT']
    elif stock == 'FB':
        stocklist = ['FB', 'fb', 'Facebook', '$FB']
    elif stock == 'GOOG':
        stocklist = ['GOOG', 'goog', 'Google', 'Alphabet', '$GOOG']
    elif stock == 'AMZN':
        stocklist = ['AMZN', 'amzn', 'Amazon', '$AMZN']

    raw_dict = rawimport(year, month)

    try:
        # Stay safe!!!
        work_dict = raw_dict.copy()

        # filter for stocks and replace dict values
        for ent in work_dict:
            t = []
            for string in work_dict[ent]:  # work_dict[ent] is a list
                for stock in stocklist:
                    if stock in string:
                        t.append(string)
            work_dict[ent] = t

        if len(work_dict) == len(raw_dict):
            print(stock, "I.O.")  # means time series has correct length in the end

        work_sum = []
        for ent in work_dict:
            work_sum.append(len(work_dict[ent]))

        print("Comments filtered:", sum(work_sum))
        # print("Filter ratio:", sum(work_sum) / sum(in_sum) * 100, "%")

        with open(f'{year}_{month}_input_{stock}.csv', 'w') as f:
            writer = csv.writer(f)
            for k, v in work_dict.items():
                writer.writerow([k, v])

    except:
        print("Something went wrong")

## classify per stock
def classify(stock, year, month):

    csv.field_size_limit(100000000)
    print(stock, year, month)

    # Load pretrained model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('pytorch_model.bin', config='config.json', num_labels=3)
    # unlike BERT, FinBERT has 3 labels

    # Import CSV to Dict
    with open(f'{year}_{month}_input_{stock}.csv', mode='r') as inp:
        reader = csv.reader(inp)
        input_dict = {rows[0]: rows[1] for rows in reader}

    # Transform string of list to list and replace Dict value
    for entry in input_dict:
        x = input_dict[entry]
        x = ast.literal_eval(x)
        input_dict[entry] = x

    # Classify sentiment of each comment, build score, aggregate per day
    sentDict = {entry: [] for entry in input_dict}
    for entry in input_dict:  # access every post
        sentScore = []
        for comment in input_dict[entry]:  # access every comment of each post
            inputs = tokenizer(comment, return_tensors="pt", truncation=True, padding=True)
            outputs = model(**inputs)

            flat_list = [item for sublist in torch.softmax(outputs[0], 1).tolist() for item in sublist]
            # make tensor output to list and thus iterable
            score = flat_list[0] - flat_list[1]  # softmax probabilities from tensor used to calc sentiment score
            sentScore.append(score)

            sentDict[entry] = sum(sentScore)

    # Days without comments get score of 0
    for key, val in sentDict.items():
        if val == []:
            print(key)
            sentDict[key] = 0

    # sentDict to DF
    interm = pd.DataFrame(sentDict, index=['sentiment'])
    sentDf = interm.transpose()
    sentDf.index = pd.to_datetime(sentDf.index)

    sentDf.to_csv(f'{year}_{month}_output_{stock}.csv')

    plt.plot(sentDf)
    plt.show()
    # plt.savefig(f'{quarter}_data/{quarter}_output/{stock}.jpg')
