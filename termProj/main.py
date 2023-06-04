import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib as plt
import os
import sys


# device setting
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'{device} is available.')

# data load & preprocessing
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                        transform=transform)
trainloader = DataLoader(trainset, batch_size=8, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                       transform=transform)
testloader = DataLoader(testset, batch_size=8, shuffle=False)

dataiter = iter(trainloader)
images, labels = next(dataiter)

for i in range(4):
  plt.subplot(1, 4, i+1)
  plt.imshow(images[i][0], cmap='gray')

plt.show()

# making model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.feature_extractor = nn.Sequential(nn.Conv2d(3, 6, 5),
                                               nn.ReLU(),
                                               nn.MaxPool2d(2, 2),
                                               nn.Conv2d(6, 16, 5),
                                               nn.ReLU(),
                                               nn.MaxPool2d(2, 2))

        self.classifier = nn.Sequential(nn.Linear(16 * 5 * 5, 120),
                                        nn.ReLU(),
                                        nn.Linear(120, 10),
                                        nn.Softmax(dim=1))
        
    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.classifier(x)
        return x

net = Net().to(device)
print(net)

# trainig model
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)

epoch = 10
loss_ = []
n = len(trainloader)

for ep in range(epoch):
  running_loss = 0.0
  for i, data in enumerate(trainloader, 0):
    inputs, labels = data[0].to(device), data[1].to(device)

    optimizer.zero_grad()

    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
  
  loss_.append(running_loss / n)
  print(f'[{ep + 1}] loss: {running_loss / n}')

print('Finished Trainning')

# evaluate model
plt.plot(loss_)
plt.title('Training Loss')
plt.xlabel('epoch')
plt.show()

PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

correct = 0
total = 0

with torch.no_grad():
  for data in testloader:
    images, labels = data[0].to(device), data[1].to(device)
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10,000 test images: %d %%' % (100 * correct / total))