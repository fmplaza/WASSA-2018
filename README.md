# WASSA-2018

Contact person: Flor Miriam Plaza del Arco, fmplaza@ujaen.es

http://sinai.ujaen.es/es/

Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.

## Project Structure

* `folder/src` -- The code of the experiments
* `folder/src/corpus` -- The dataset provided by the organizers of WASSA 2018
* `folder/src/lexicon` -- The lexicon used for obtained the external knowledge (emotional features)
* `folder/src/embeddings_RNN.py` -- The file of the application without emotional features
* `folder/src/embeddings_lexicon_features_emotion_RNN.py` -- The file of the application with emotional features
* `folder/model` -- The files of the application


## Requirements

* python 3.5.4
* tensorflow 0.11.0rc2
* numpy 1.12.1
* scikit-learn 0.18
* NLTK 3.2.1
* If you use GPU: Cuda 8.0
* Word embeddings: You have to use the Twitter (2B tweets, 27B tokens, 1.2M vocab, uncased, 25d, 50d, 100d, & 200d vectors, 1.42 GB download) version of Glove: http://nlp.stanford.edu/data/glove.twitter.27B.zip


Run
------------------

To run the experiments you have to go to the folder code and run the following command:

Run RNN withouth emotional features configuration: 

python3 embeddings_RNN.py

Run RNN with emotional features configuration: 

python3 embeddings_lexicon_features_emotion_RNN.py




  
