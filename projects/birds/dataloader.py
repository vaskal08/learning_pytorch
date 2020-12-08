import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import random

test_dir = "../../datasets/birds/test"
train_dir = "../../datasets/birds/train"
valid_dir = "../../datasets/birds/valid"
consolidated_dir = "../../datasets/birds/consolidated"

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def load_image(path):
    image = Image.open(path)
    x = transforms.functional.to_tensor(image)
    return x

def load_train(species=-1, shuffle=False):
    l = sorted(os.listdir(train_dir))
    if species > 0:
        l = l[:species]
    images = []
    i = 0
    for s in l:
        d = train_dir+'/'+s
        img_list = os.listdir(d)
        for img in img_list:
            img_path = d+'/'+img
            images.append((img_path, i))
        i = i+1
    
    if shuffle:
        random.shuffle(images)
    
    return (l, images)

def load_test(species=-1, shuffle=False):
    l = sorted(os.listdir(test_dir))
    if species > 0:
        l = l[:species]
    images = []
    i = 0
    for s in l:
        d = test_dir+'/'+s
        img_list = os.listdir(d)
        for img in img_list:
            img_path = d+'/'+img
            images.append((img_path, i))
        i = i+1
    
    if shuffle:
        random.shuffle(images)
    
    return (l, images)
