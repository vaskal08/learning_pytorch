import torch
from torch.utils.data import Dataset
import csv, re
from enum import Enum
import numpy as np

class Key(Enum):
    ID='Id'
    PRODUCTID='ProductId'
    USERID='UserId'
    PROFILENAME='ProfileName'
    HELPNUM='HelpfulnessNumerator'
    HELPDEN='HelpfulnessDenominator'
    SCORE='Score'
    TIME='Time'
    SUMMARY='Summary'
    TEXT='Text'

class AmazonDataset(Dataset):
    def __init__(self, loc='../../datasets/amazon_reviews/Reviews.csv', maxrows=-1, reviewlen=250):
        self.loc = loc
        self.reviews = []
        self.encoded_reviews = []
        self.reviewlen = reviewlen
        self.vocab_occ = {}
        self.vocab = {}
        with open(loc, encoding='utf8', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            
            keys = None
            i=0
            for row in reader:
                review = {}
                if keys == None:
                    keys = row
                else:
                    x=0
                    for o in row:
                        review[keys[x]] = o
                        x+=1
                
                if i > 0:
                    self.reviews.append(review)
                if maxrows > 0 and i >= maxrows:
                    break
                i+=1
        
        self.__process_vocab()
        self.__process_inputs()

    def __process_vocab(self):
        vocab_occ = {}
        vocab = {}
        for row in self.reviews:
            text = row[Key.TEXT.value]
            text = re.sub('<[^<]+?>', '', text)
            text = re.sub(r'[^\w\s]','',text)

            words = text.split(' ')
            for word in words:
                word = word.lower().strip()
                if len(word) < 1:
                    continue
                if word in vocab_occ:
                    vocab_occ[word] = vocab_occ[word] + 1
                else:
                    vocab_occ[word] = 1

        for k in list(vocab_occ.keys()):
            if vocab_occ[k] < 2:
                del vocab_occ[k]

        x = 1
        for w in sorted(vocab_occ, key=vocab_occ.get, reverse=True):
            if w not in vocab:
                vocab[w] = x
            x+=1
        self.vocab_occ = vocab_occ
        self.vocab = vocab
    
    def __process_inputs(self):
        for row in self.reviews:
            score = int(row[Key.SCORE.value])
            label = 1 if score >= 4 else 0
            label = torch.tensor(label)

            text = row[Key.TEXT.value]
            text = re.sub('<[^<]+?>', '', text)
            text = re.sub(r'[^\w\s]','',text)

            words = text.split(' ')
            encoded = list()
            for word in words:
                if word in self.vocab:
                    encoded.append(self.vocab[word])
                else:
                    encoded.append(0)

            review_len=len(encoded)
            if (review_len<=self.reviewlen):
                zeros=list(np.zeros(self.reviewlen-review_len))
                new=zeros+encoded
            else:
                new=encoded[:self.reviewlen]
            review_np = np.array(new, dtype=np.float32)
            review_t = torch.from_numpy(review_np)
            self.encoded_reviews.append((review_t, label))
    
    def make_input(self, input_str):
        text = re.sub('<[^<]+?>', '', input_str)
        text = re.sub(r'[^\w\s]','',text)

        words = text.split(' ')
        encoded = list()
        for word in words:
            print (word)
            word = word.lower()
            if word in self.vocab:
                encoded.append(self.vocab[word])
            else:
                encoded.append(0)
        review_len=len(encoded)
        if (review_len<=self.reviewlen):
            zeros=list(np.zeros(self.reviewlen-review_len))
            new=zeros+encoded
        else:
            new=encoded[:self.reviewlen]
        torch.set_printoptions(precision=4)
        review_np = np.array(new, dtype=np.float32)
        review_t = torch.from_numpy(review_np)

        return review_t.unsqueeze(0)

    
    def __len__(self):
        return len(self.encoded_reviews)
    
    def __getitem__(self, idx):
        return self.encoded_reviews[idx]