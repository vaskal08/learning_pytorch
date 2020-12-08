from AmazonDataset import AmazonDataset, Key
import collections
import re


trainset = AmazonDataset(maxrows=100)

vocab_occ = trainset.vocab_occ
vocab = trainset.vocab

for i, data in enumerate(trainset):
    inputs, labels = data
    print (inputs)
    print (labels)