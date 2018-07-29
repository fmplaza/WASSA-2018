'''
Created on 20 jun. 2018

@author: fmplaza
'''

import os
from keras.datasets.imdb import get_word_index
from model.glove_word_embedings import GloveWordEmbednigs
import pandas as pd
from nltk.tokenize.casual import TweetTokenizer
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Bidirectional, Conv1D, GlobalAveragePooling1D, MaxPooling1D, Dropout, Activation, Flatten, GlobalMaxPooling1D, ActivityRegularization
from mpl_toolkits.axes_grid1.axes_size import Padded
from keras.utils import np_utils
from sklearn import metrics
from nltk.tokenize.casual import TweetTokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
from keras.initializers import glorot_normal, glorot_uniform
from keras import regularizers
import random
from tensorflow import set_random_seed
from scipy import stats
import csv 

RANDOM_SEED = 666

np.random.seed(RANDOM_SEED)
set_random_seed(RANDOM_SEED)

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
TWEET_TOKENIZER = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=False)
CLASSES = []
EMBEDDING_DIM = 214

#load GloVe model
glove = GloveWordEmbednigs()
glove_file = "./embeddings/glove.twitter.27B/glove.twitter.27B.200d.txt"
glove.path_file = glove_file

#Load the Glove vectors file into memory, 3 index reserved (0: paddind, 1: word not present in embedding, 2: magic word)
number_features = 500000
begin_ofset = 3
glove.load(number_features, begin_ofset)


#Load the WASSA corpus
def read_corpus():
    
    classes_append = CLASSES.append
    tweets_train_labels_numeric = []
    tweets_dev_labels_numeric = []
    
    tweets_train = pd.read_csv('./corpus/train-v2.csv', sep="\t", header=0)
    tweets_train_labels = tweets_train['emotion']
    tweets_dev = pd.read_csv('./corpus/trial-v2.csv', sep="\t", header=0)
    tweets_dev_labels = pd.read_csv('./corpus/trial-v2.labels', sep="\t", header=0)
    tweets_dev_labels = tweets_dev_labels['emotion']
    
    
    #convert categorical labels into numeric labels
    for label in tweets_train_labels.tolist():
        if(label not in CLASSES):
            classes_append(label)
        tweets_train_labels_numeric.append(CLASSES.index(label))
                
    for label in tweets_dev_labels.tolist():
        tweets_dev_labels_numeric.append(CLASSES.index(label))       
                
    tweets_train_labels_numeric = np_utils.to_categorical(tweets_train_labels_numeric)
    
    
    return tweets_train.tweet, tweets_train_labels_numeric, tweets_dev.tweet, tweets_dev_labels_numeric

def read_lexicon():
    
    #read lexicon emolex
    emolex_lexicon = {}
    with open('lexicon/emolex_english.csv') as csvfile:
        csvresult = csv.reader(csvfile, delimiter='\t')
        next(csvresult)
        for row in csvresult:
            term = row[0]
            anger = int(row[1])
            disgust = int(row[2])
            fear = int(row[3])
            joy = int(row[4])
            sadness = int(row[5])
            surprise = int(row[6])           
            trust = int(row[7])
            anticipation = int(row[8])
            
            list_emotion = [term, anger, disgust, fear, joy, sadness, surprise, trust, anticipation]
            if(term not in emolex_lexicon):
                list_emotion = [anger, disgust, fear, joy, sadness, surprise, trust, anticipation]
                emolex_lexicon[term] = list_emotion
            else:
                value = emolex_lexicon[term]
                value[0]+=anger
                value[1]+=disgust
                value[2]+=fear
                value[3]+=joy
                value[4]+=sadness
                value[5]+=surprise 
                value[6]+=trust
                value[7]+=anticipation
    
    #read lexicon emojis
    emoji_lexicon = {}
    with open('lexicon/emojis.txt') as csvfile:
        csvresult = csv.reader(csvfile, delimiter=' ')  
        next(csvresult)
        for row in csvresult:
            if(row[1] == "anger"):
                emoji_lexicon[row[0]] = [1,0,0,0,0,0]
            
            if(row[1] == "disgust"):
                emoji_lexicon[row[0]] = [0,1,0,0,0,0]
                
            if(row[1] == "fear"):
                emoji_lexicon[row[0]] = [0,0,1,0,0,0]
                
            if(row[1] == "joy"):
                emoji_lexicon[row[0]] = [0,0,0,1,0,0]
                
            if(row[1] == "sad"):
                emoji_lexicon[row[0]] = [0,0,0,0,1,0]
        
            if(row[1] == "surprise"):
                emoji_lexicon[row[0]] = [0,0,0,0,0,1]
    
    return emolex_lexicon, emoji_lexicon

def tokenize(text):
    #preprocessing data

    text_tokenized = TWEET_TOKENIZER.tokenize(text)
    
    return text_tokenized


def fit_transform_vocabulary(corpus):
    #generate vocabulary of corpus
    
    #index 0: padding
    #index 1: word not present in the embedding
    #index 2: word magic (triggerword)
    
    #corpus_indexes: index of each word of tweet in the embedding model
      
    corpus_indexes = []
    corpus_lengths = []
    own_append_corpus_lengths = corpus_lengths.append
    own_lower = str.lower
    for doc in corpus:
        tweet_indexes = []
        tokens = tokenize(doc)
        own_append_corpus_lengths(len(tokens))
        for token in tokens:
            if(token != "#triggerword"):
                if(glove.is_word(own_lower(token))):
                    word_index_embedding = glove.get_word_index(own_lower(token))
                    tweet_indexes.append(word_index_embedding)
                else:
                    index = 1
                    tweet_indexes.append(index)
            else:
                index = 2
                tweet_indexes.append(index)

                
        corpus_indexes.append(tweet_indexes)
    
    
    print(np.max(corpus_lengths))
    print(np.mean(corpus_lengths))
    print(stats.mode(corpus_lengths, axis=0))
    return corpus_indexes


def classification_embedings_rnn(tweets_train, tweets_train_labels_numeric, tweets_dev, emolex_lexicon, emoji_lexicon):
    #Classification with RNN and embedings (pre-trained) 
        
    #calculate vocabulary    
    corpus_train_index = fit_transform_vocabulary(tweets_train)
    corpus_dev_index = fit_transform_vocabulary(tweets_dev)

    max_len_input = 27
               
    train_features_pad = sequence.pad_sequences(corpus_train_index, maxlen=max_len_input, padding="post", truncating="post", value = 0)
    padded_docs_dev = sequence.pad_sequences(corpus_dev_index, maxlen=max_len_input, padding="post", truncating="post", value = 0)

    # define RNN model
    model = Sequential()
    
    #assign special index
    trigger_word_vector = 2 * 0.1 * np.random.rand(EMBEDDING_DIM) - 1
    glove.set_embedding_vector(1, trigger_word_vector)
    
    vector_word_not_present = 2 * 0.1 * np.random.rand(EMBEDDING_DIM) - 1
    glove.set_embedding_vector(2, vector_word_not_present)
    
            
    #number of features in embeddings model 
    feature_size = number_features + 3
    embedding_matrix = np.zeros((feature_size, EMBEDDING_DIM))
    
    for word, idx in glove.word_indexes.items():
        emolex_vector = [0,0,0,0,0,0,0,0]
        emoji_vector = [0,0,0,0,0,0]
        if(word in emolex_lexicon.keys()):
            emolex_vector = emolex_lexicon[word]
        if(word in emoji_lexicon.keys()):
            emoji_vector = emoji_lexicon[word]
        list_features_lexicons = emolex_vector + emoji_vector
        embedding_vec = glove.get_word_embedding(word)
        embedding_vec = np.concatenate((embedding_vec, list_features_lexicons), axis=0) 
        if embedding_vec is not None and embedding_vec.shape[0]==EMBEDDING_DIM:
            embedding_matrix[idx] = np.asarray(embedding_vec)
    
    #input_length:  Length of input sequences, when it is constant
    e = Embedding(feature_size, EMBEDDING_DIM, input_length=max_len_input, weights=[embedding_matrix], trainable=False)
    model.add(e)
    #number of features:_32 each vector of 200 dim is converted to a vector of 32 dim
    
    #model.add(LSTM(128, return_sequences=True))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Dense(128, activation='relu', kernel_initializer=glorot_uniform(seed=RANDOM_SEED)))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2, strides=1, padding="same"))
    model.add(Flatten())
    model.add(Dense(64, activation='relu', kernel_initializer=glorot_uniform(seed=RANDOM_SEED)))
    model.add(Dropout(0.5))
    model.add(Dense(len(CLASSES), activation='softmax'))
    model.add(ActivityRegularization(l1=0.0,l2=0.0001))
    
    # summarize the model
    print(model.summary())

    print("Compiling the model...")
    # compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    print("Training the model...")
    
    earlyStopping = EarlyStopping('loss', patience=3, mode='min')
    
    #model.fit(train_features_pad, tweets_train_labels_numeric, batch_size=64, epochs=1, verbose=1, validation_data=(train_features_pad,tweets_train_labels_numeric), callbacks=[earlyStopping])
    model.fit(train_features_pad, tweets_train_labels_numeric, batch_size=64, epochs=30, verbose=1, callbacks=[earlyStopping])
    loss, accuracy = model.evaluate(train_features_pad, tweets_train_labels_numeric, batch_size=64, verbose=1)
    print('Accuracy trainning: %f' % (accuracy*100))
    
    #prediction
    tweets_dev_classified_labels = model.predict_classes(padded_docs_dev, batch_size=64, verbose=1)
    return tweets_dev_classified_labels
    
def calculate_quality_performamnce(y_labels, y_classified_labels, model_name):
    
    classes_index = [CLASSES.index(c) for c in CLASSES]
    accruacy = metrics.accuracy_score(y_labels, y_classified_labels)
    macro_precision = metrics.precision_score(y_labels, y_classified_labels, labels=classes_index, average="macro")
    macro_recall = metrics.recall_score(y_labels, y_classified_labels, labels=classes_index, average="macro")
    macro_f1 = metrics.f1_score(y_labels, y_classified_labels, labels=classes_index, average="macro")
    
    print("\n*** Results " + model_name + " ***")
    print("Macro-Precision: " + str(macro_precision))
    print("Macro-Recall: " + str(macro_recall))
    print("Macro-F1: " + str(macro_f1))
    print("Accuracy: " + str(accruacy))
                          
def main ():

    tweets_train, tweets_train_labels_numeric, tweets_dev, tweets_dev_labels = read_corpus()
    emolex_lexicon, emoji_lexicon = read_lexicon()
    tweets_dev_classified_labels =  classification_embedings_rnn(tweets_train, tweets_train_labels_numeric, tweets_dev, emolex_lexicon, emoji_lexicon)
    calculate_quality_performamnce(tweets_dev_labels, tweets_dev_classified_labels, "RNN_LSTM")

if __name__ == '__main__':
    main()
