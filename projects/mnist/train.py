import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import Net as n

use_cuda = torch.cuda.is_available()
print (use_cuda)

trans = transforms.Compose([transforms.ToTensor()])
trainset = datasets.MNIST(root='../../datasets/mnist', train=True, download=True, transform=trans)
train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=1, shuffle=True)

net = n.Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)
running_loss = 0.0
for i, (x, target) in enumerate(train_loader):

    optimizer.zero_grad()

    output = net(x)
    _, pred_label = torch.max(output.data, 1)

    print (output)
    print (target)

    loss = criterion(output, target)
    loss.backward()

    optimizer.step()
    running_loss += loss.item()
    if i % 2000 == 1999:    # print every 2000 mini-batches
        print('[%5d] loss: %.3f' %
                (i + 1, running_loss / 2000))
        running_loss = 0.0
    

torch.save(net.state_dict(), '../../models/mnist/mnist.pt')