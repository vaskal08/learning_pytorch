import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.datasets as datasets
import Net as n

trans = transforms.Compose([transforms.ToTensor()])
test_set = datasets.MNIST(root="../../datasets/mnist", train=False, transform=trans, download=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=1, shuffle=False)

model = n.Net()
model.load_state_dict(torch.load('../../models/mnist/mnist.pt'))
model.eval()

criterion = nn.CrossEntropyLoss()

correct_cnt, ave_loss = 0, 0
total_cnt = 0
for batch_idx, (x, target) in enumerate(test_loader):
    with torch.no_grad():
        x, target = Variable(x), Variable(target)
    out = model(x)
    loss = criterion(out, target)
    _, pred_label = torch.max(out.data, 1)
    total_cnt += x.data.size()[0]
    correct_cnt += (pred_label == target.data).sum()
    # smooth average
    ave_loss = ave_loss * 0.9 + loss.data * 0.1
    
    if(batch_idx+1) % 100 == 0 or (batch_idx+1) == len(test_loader):
        print ("==>>> batch index: {}, test loss: {:.6f}, acc: {:.3f}".format(batch_idx+1, ave_loss, correct_cnt * 1.0 / total_cnt))