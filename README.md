# FinBERT_Reddit
Sentiment analysis on financial text data collected from Reddit via Pushshift API and FinBERT.

During my master thesis I applied the FinBERT classification algorithm from ProsusAI (https://huggingface.co/ProsusAI/finbert). The folder "Abgabe" contains all data necessary to repliacte my work.
* Files in "Abgabe/data/raw_comments" were filtered for each stock ticker
* Resulting in the files in "Abgabe/data/filtered_comments".
* Those files were now used during the classifcation task
* Resulting in sentiment scores, as can be found in "Abgabe/data/all_data_quant.csv"
