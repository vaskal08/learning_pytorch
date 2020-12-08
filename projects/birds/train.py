import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import dataloader as d
import net as n
import torchvision.models as models
import BirdDataset as bd
import time

trainset = bd.BirdDataset(species=10)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,shuffle=True)

net = n.Net()
vgg = models.vgg16(pretrained=True)

model = nn.Sequential(vgg, net)


# CUDA
is_cuda = True and torch.cuda.is_available()
device = torch.device('cuda') if is_cuda else torch.device('cpu')

if is_cuda:
    torch.cuda.empty_cache()

print (device)

model.to(device)

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

start_time = time.time()

for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader):
        inputs, labels = data
        
        if is_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        optimizer.zero_grad()

        output = model(inputs)

        _, pred_label = torch.max(output.data, 1)
        
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 50 == 49:    # print every 50 mini-batches
            print('[%d %5d] loss: %.3f' %
                    (epoch+1, i+1, running_loss / 50))
            running_loss = 0.0

end_time = time.time()
print ('\n-------- DONE --------')
print ('start time: {}'.format(start_time))
print ('end time: {}\n'.format(end_time))

duration = end_time-start_time
print ('training duration: {}'.format(duration))

torch.save(model.state_dict(), '../../models/birds/birds{}.pt'.format(end_time))