import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import net as n
import torchvision.models as models
from AmazonDataset import AmazonDataset, Key
import time


# parameters
maxrows = 100000
reviewlen = 250
batch_size = 4
epochs = 1

# model 
model = n.Net()

# CUDA
is_cuda = True and torch.cuda.is_available()
device = torch.device('cuda') if is_cuda else torch.device('cpu')
if is_cuda:
    torch.cuda.empty_cache()
print (device)
model.to(device)

# learning
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

trainset = AmazonDataset(maxrows=maxrows, reviewlen=reviewlen)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

start_time = time.time()

for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        output = model(inputs)
        loss = criterion(output, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        if i % 1000 == 999:    # print every 50 mini-batches
            print('[%d %5d] loss: %.3f' %
                    (epoch+1, i+1, running_loss / 1000))
            running_loss = 0.0


end_time = time.time()
print ('\n-------- DONE --------')
print ('start time: {}'.format(start_time))
print ('end time: {}\n'.format(end_time))

duration = end_time-start_time
print ('training duration: {}'.format(duration))

torch.save(model.state_dict(), '../../models/amazon_reviews/reviews{}.pt'.format(end_time))