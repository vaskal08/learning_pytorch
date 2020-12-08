import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.datasets as datasets
import net as n
import dataloader as d
import torchvision.models as models
import BirdDataset as bd

net = n.Net()
vgg = models.vgg16()

model = nn.Sequential(vgg, net)

# CUDA
is_cuda = True and torch.cuda.is_available()
device = torch.device('cuda') if is_cuda else torch.device('cpu')

if is_cuda:
    torch.cuda.empty_cache()

print (device)

model.to(device)

model.load_state_dict(torch.load('../../models/birds/birds10_2epoch_4batch.pt'))

testset = bd.BirdDataset(species=10, train=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=1,shuffle=True)

correct = 0
total = 0  

print (len(testset))

with torch.no_grad(): 
    for i, data in enumerate(testloader):
        inputs, labels = data

        if is_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        output = model(inputs)

        _, prediction = torch.max(output.data, 1)

        total += 1
        if prediction == labels:
            correct += 1

print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))