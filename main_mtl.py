import torch
import argparse
import pandas as pd
from load_data_mtl import return_data
from train_mtl import train
from test import test

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, required=True, choices=['train', 'test'])
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--lr', type=float, default=5e-5)



train_data,test_data=return_data()

args = parser.parse_args()
batch_size=int(args.batch_size)
epochs=int(args.epochs)
learning_rate=float(args.lr)
if args.mode=="train":
    train_sentences_1=train_data["sentences_1"]
    train_sentences_2=train_data["sentences_2"]
    train_labels_1=train_data["labels1"]
    train_labels_2=train_data["labels2"]
    

    
    train(train_sentences_1,train_sentences_2,train_labels_1, train_labels_2,
          batch_size, epochs, learning_rate)
    

else:
    test_sentences_1=test_data["sentences_1"]
    test_sentences_2=test_data["sentences_2"]
    test_labels=test_data["labels"]
    acc,f1,precision,recall=test(test_sentences_1,test_sentences_2,test_labels)



