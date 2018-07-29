'''
Created on 20 jun. 2018

@author: fmplaza
'''

import pandas as pd
from nltk.tokenize import TweetTokenizer
import statistics as s

tknzr = TweetTokenizer()


def read_corpus():
    
    labels = []
    tweets_train_labels_numeric = []
    
    tweets_train = pd.read_csv('./corpus/train-v2.csv', sep="\t", header=0)
    tweets_train_labels = tweets_train['emotion']
    tweets_dev = pd.read_csv('./corpus/trial-v2.csv', sep="\t", header=0)
    tweets_dev_labels = pd.read_csv('./corpus/trial-v2.labels', sep="\t", header=0)
    

    return tweets_train, tweets_train_labels, tweets_dev, tweets_dev_labels

def calculate_position(tweets_train):
    
    
    position_triggerword = []
    len_tweet = []
    cont = 0
    for index, row in tweets_train.iterrows():
        text = tknzr.tokenize(row['tweet'])
        if '#TRIGGERWORD' in text:
            position = text.index('#TRIGGERWORD')
            position_triggerword.append(position)
            len_tweet.append(len(text))
        if(len(text) == 155):
            cont = cont + 1
    
    print("Position trigger word")
    print("Max position: ", max(position_triggerword))
    print("Mean position" , s.mean(position_triggerword))
    print("Mode position", s.mode(position_triggerword))
    
    print("Lenght tweet")
    print("Mean lenght", s.mean(len_tweet))
    print("Mode lenght", s.mode(len_tweet))
            
        

def main():
    tweets_train, tweets_train_labels, tweets_dev, tweets_dev_labels = read_corpus()
    calculate_position(tweets_train)

if __name__ == '__main__':
    main()
    pass