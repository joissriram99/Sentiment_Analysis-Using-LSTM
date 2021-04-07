import torch
from nltk.corpus import stopwords
import re
from collections import Counter
import numpy as np

vocab_size = 2000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
file_path = "C:/Users/12158/Desktop/SentimentAnalysis/Input/Dataset.csv"
len_sez = 100
batch_size = 16
lr = 0.001
epochs = 10
n_layers = 2
embedding_dim = 40
hidden_dim = 256

def preprocess(input):
    # Remove all non-word characters (everything except numbers and letters)
    input = re.sub(r"[^\w\s]", '', input)
    # Replace all runs of whitespaces with no space
    input = re.sub(r"\s+", '', input)
    # replace digits with no space
    input = re.sub(r"\d", '', input)

    return input

def tokenize(x_train,y_train,x_val,y_val):

    word_list = []
    stop_words = set(stopwords.words("english"))
    for sent in x_train:
        for word in sent.lower().split():
            word = preprocess(word)
            if word not in stop_words and word !=" ":
                word_list.append(word)
    vocab_len = Counter(word_list)
    corpus = sorted(vocab_len,key = vocab_len.get,reverse=True)[:vocab_size]
    onehotDict = {w:i+1 for i,w in enumerate(corpus)}

    train_list,eval_list = [],[]

    for sent in x_train:
        train_list.append([onehotDict[preprocess(word)] for word in sent.lower().split() if preprocess(word) in onehotDict.keys()])
    for sent in x_val:
        eval_list.append([onehotDict[preprocess(word)] for word in sent.lower().split() if preprocess(word) in onehotDict.keys()])

    train_label = [1 if label=="positive" else 0 for label in y_train]
    eval_label = [1 if label =="positive" else 0 for label in y_val]

    return np.array(train_list),np.array(train_label),np.array(eval_list),np.array(eval_label),onehotDict

def padding(sentence,seq_len):
    features = np.zeros((len(sentence),seq_len))
    for i,review in enumerate(sentence):
        if len(review) !=0:
            features[i,-len(review):] = np.array(review)[:seq_len]
    return features


