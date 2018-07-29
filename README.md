# WASSA 2018 Implicit Emotion Shared Task

Emotion is a concept that is challenging to describe. Yet, as human beings, we understand the emotional effect situations have or could have on us and other people. How can we transfer this knowledge to machines? Is it possible to learn the link between situations and the emotions they trigger in an automatic way?

In the light of these questions, we proposed the Shared Task on Implicit Emotion Recognition, organized as part of WASSA 2018 at EMNLP 2018 aims at developing models which can classify a text into one of the following emotions: Anger, Fear, Sadness, Joy, Surprise, Disgust without having access to an explicit mention of an emotion word.

## Task Description

Participants were given a tweet from which a certain emotion word is removed. That word is one of the following: "sad", "happy", "disgusted", "surprised", "angry", "afraid" or a synonym of one of them. The task was to predict the emotion the excluded word expresses: Sadness, Joy, Disgust, Surprise, Anger, or Fear.

http://implicitemotions.wassa2018.com/

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

- Run RNN withouth emotional features configuration: 

python3 embeddings_RNN.py

- Run RNN with emotional features configuration: 

python3 embeddings_lexicon_features_emotion_RNN.py




  
