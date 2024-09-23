from torch import nn
from tqdm import tqdm
from dataclasses import dataclass
import os
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

@dataclass
class HP: 
    NUM_EPOCHS = 5
    LR = 0.01
    MOMENTUM = 0.9
    BATCH_SIZE = 32

train_dataset = torchvision.datasets.CIFAR10("data/cifar10", train=True, download=True, transform=transforms.ToTensor())
test_dataset = torchvision.datasets.CIFAR10("data/cifar10", train=False, download=True, transform=transforms.ToTensor())


def view_data():
    print("attributes: {}".format(dir(train_dataset)))
    print("sample train data: {}".format(train_dataset[0]))

    NUM_SAMPLES = 40
    plt.figure(figsize=(40, 18))
    for i in range(NUM_SAMPLES):
        plt.subplot(4, 10, i+1)
        data, label = train_dataset[i]
        label = train_dataset.classes[label]

        data = data.permute(1, 2, 0)
        plt.title(label)
        plt.imshow(data)
    plt.show()

class VGGNet(nn.Module):
    ARCH ={ 'DEFAULT': [32, 64, 'M', 128, 'M', 256, 512],
            'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
          }
    
    def __init__(self, in_channels, num_classes=10, image_dims=(32, 32), vgg_type="DEFAULT"):
        super(VGGNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.layers = []

        ARCH = VGGNet.ARCH[vgg_type]
        prev = self.in_channels
        for idx in range(len(ARCH)):
            arch = ARCH[idx]
            self.add(arch, prev)
            if arch != 'M':
                prev = arch
        
        self.pipeline = nn.Sequential(*self.layers)
        print("kernel sizes: {}".format(list(map(lambda x: x.weight.size(), list(filter(lambda x: type(x) == nn.Conv2d, self.layers))))))
        self.fcs = nn.Sequential(nn.Linear(512 * 8 * 8, 4096), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(4096, self.num_classes), nn.LogSoftmax(dim=1) )

    def add(self, arch, prev):
        if arch != 'M':
            self.layers.append(nn.Conv2d(in_channels=prev, out_channels=arch, kernel_size=(3, 3), stride=(1, 1), padding=1))
            #self.layers.append(nn.BatchNorm2d(arch))
            self.layers.append(nn.ReLU(True))
        else:
            self.layers.append(nn.MaxPool2d(2, 2))
        
    def forward(self, x):
        print("input shape: {}".format(x.size()))
        output = self.pipeline(x)
        print("output shape: {}".format(output.size()))
        class_probs = self.fcs(output.reshape(output.shape[0], -1))
        return class_probs

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = VGGNet(in_channels=3).to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=HP.LR, momentum=HP.MOMENTUM)
loss_fn = torch.nn.NLLLoss()
train_loader = DataLoader(train_dataset, batch_size=HP.BATCH_SIZE, shuffle=True)

print(f"Number of total examples in train dataset: {len(train_loader)}")
for epoch in tqdm(range(1, 5)):
    accuracy = 0
    for idx, (inp, label) in enumerate(train_loader): # labels are batched into tensor of dim (64,)
        optimizer.zero_grad()
        output = model(inp)
        
        expected_classes = torch.argmax(output, dim=1)
        one_hot = torch.nn.functional.one_hot(label, num_classes=10).float()
        #print(f"one hot encoded expected classes: {one_hot}, {one_hot.size()}")
        #print(f"output probabilities tensor: {output}, {output.size()}")

        accuracy += (expected_classes == label).sum().item()

        loss_val = loss_fn(output, label)
        loss_val.backward()
        optimizer.step() # update gradients on every example

        if idx % 4 == 0:
            print("Accuracy on epoch {} iteration {}: {:.2f}".format(epoch, idx, accuracy / len(train_loader)))

for idx, (inp, label) in enumerate(test_loader):
    pass
    

