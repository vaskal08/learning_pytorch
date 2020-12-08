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
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.cuda.empty_cache()

print (device)

# load model
model.to(device)
model.load_state_dict(torch.load('./models/birds10_2epoch_4batch.pt'))


labels, images = d.load_test(species=10)
path, target = images[24]

image = d.load_image(path)

i = image.unsqueeze(0)
i = i.to(device)

print (labels[target])

output = model(i)
_, prediction = torch.max(output.data, 1)
prediction = prediction.cpu()

print (labels[prediction.item()])

d.imshow(image)