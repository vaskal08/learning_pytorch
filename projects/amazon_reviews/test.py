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

# load dataset
trainset = AmazonDataset(maxrows=maxrows, reviewlen=reviewlen)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

# # load model 
# model = n.Net()
# model.load_state_dict(torch.load('../../models/amazon_reviews/reviews1607200499.496372.pt'))

# # CUDA
# is_cuda = True and torch.cuda.is_available()
# device = torch.device('cuda') if is_cuda else torch.device('cpu')
# if is_cuda:
#     torch.cuda.empty_cache()
# print (device)
# model.to(device)

# # learning
# criterion = nn.CrossEntropyLoss()
# #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# optimizer = optim.Adam(model.parameters(), lr=0.0001)

# with torch.no_grad():
#     example = "I hate this product. It was the worst thing i have ever tasted."
#     example_input = trainset.make_input(example).to(device)

#     print (example_input)

#     output = model(example_input)
#     _, prediction = torch.max(output.data, 1)
#     print (output)
#     print (prediction)