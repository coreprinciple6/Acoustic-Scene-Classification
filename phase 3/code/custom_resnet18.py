import torch
import PIL
import numpy as np
import sys
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import *
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa

n_epochs = 2
batch_size = 32


class bottleNeck(torch.nn.Module):
    expansion = 4

    def __init__(self, input_planes, planes, stride=1, dim_change=None):
        super(bottleNeck, self).__init__()

        self.conv1 = torch.nn.Conv2d(input_planes, planes, kernel_size=1, stride=1)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.conv3 = torch.nn.Conv2d(planes, planes * self.expansion, kernel_size=1)
        self.bn3 = torch.nn.BatchNorm2d(planes * self.expansion)
        self.dim_change = dim_change

    def forward(self, x):
        res = x

        output = F.relu(self.bn1(self.conv1(x)))
        output = F.relu(self.bn2(self.conv2(output)))
        output = self.bn3(self.conv3(output))

        if self.dim_change is not None:
            res = self.dim_change(res)

        output += res
        output = F.relu(output)
        return output


class ResNet(torch.nn.Module):
    def __init__(self, block, num_layers, classes=2):
        super(ResNet, self).__init__()
        # according to research paper:

        # self.encoder_hidden_layer_1 = nn.Linear(in_features=4096, out_features=2048)
        # self.encoder_hidden_layer_2 = nn.Linear(in_features=2048, out_features=1024)
        # self.encoder_output_layer = nn.Linear(in_features=1024, out_features=512)
        # self.decoder_hidden_layer_1 = nn.Linear(in_features=512, out_features=1024)
        # self.decoder_hidden_layer_2 = nn.Linear(in_features=1024, out_features=2048)
        # self.decoder_output_layer = nn.Linear(in_features=2048, out_features=4096)


        self.input_planes = 64
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.layer1 = self._layer(block, 64, num_layers[0], stride=1)
        self.layer2 = self._layer(block, 128, num_layers[1], stride=2)
        self.layer3 = self._layer(block, 256, num_layers[2], stride=2)
        self.layer4 = self._layer(block, 512, num_layers[3], stride=2)
        self.averagePool = torch.nn.AvgPool2d(kernel_size=4, stride=1)
        self.fc = torch.nn.Linear(8192, classes)
        self.sigmoid = torch.nn.Sigmoid()

    def _layer(self, block, planes, num_layers, stride=1):
        dim_change = None
        if stride != 1 or planes != self.input_planes * block.expansion:
            dim_change = torch.nn.Sequential(
                torch.nn.Conv2d(self.input_planes, planes * block.expansion, kernel_size=1, stride=stride),
                torch.nn.BatchNorm2d(planes * block.expansion))
        netLayers = []
        netLayers.append(block(self.input_planes, planes, stride=stride, dim_change=dim_change))
        self.input_planes = planes * block.expansion
        for i in range(1, num_layers):
            netLayers.append(block(self.input_planes, planes))
            self.input_planes = planes * block.expansion

        return torch.nn.Sequential(*netLayers)

    def forward(self, x):

        x = torch.reshape(x, (x.shape[0], 3, 64, 64))

        x = F.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x


width, height = 64, 64

# transforms = ImgAugTransform()

transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((width, height)),
    torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
    torchvision.transforms.ToTensor()
])





def test(train, test):
    # Load train and test set:
    # train = torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)
    # trainset = ast.literal_eval(torch.utils.data.DataLoader('/content/drive/My Drive/ASC/Final/Train',batch_size=128,shuffle=True))

    train_dataset = datasets.ImageFolder('bindata/NAImgsTrain', transform=transforms)
    test_dataset = datasets.ImageFolder('bindata/NAImgsTest', transform=transforms)

    # MFCCs converted to full images (the second set)
    # train_dataset = datasets.ImageFolder('I:/Code/Python/ASC New Data/MFCCImgs/Train', transform=transforms)
    # test_dataset = datasets.ImageFolder('I:/Code/Python/ASC New Data/MFCCImgs/Test', transform=transforms)

    trainset = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testset = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # ResNet-50
    net = ResNet(bottleNeck, [3, 4, 6, 3])
    net.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-5)

    def accuracy(out, labels):
        _, pred = torch.max(out, dim=1)
        return torch.sum(pred == labels).item()

    print_every = 200
    valid_loss_min = np.Inf
    val_loss = []
    val_acc = []
    train_loss = []
    train_acc = []
    total_step = len(trainset)
    for epoch in range(1, n_epochs + 1):
        running_loss = 0.0
        correct = 0
        total = 0
        print(f'Epoch {epoch}\n')
        for batch_idx, (data_, target_) in enumerate(trainset):
            data_, target_ = data_.to(device), target_.to(device)
            optimizer.zero_grad()

            outputs = net(data_)
            loss = criterion(outputs, target_)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, pred = torch.max(outputs, dim=1)
            correct += torch.sum(pred == target_).item()
            total += target_.size(0)
            if (batch_idx) % print_every == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch, n_epochs, batch_idx, total_step, loss.item()))
        train_acc.append(100 * correct / total)
        train_loss.append(running_loss / total_step)
        print(f'\ntrain-loss: {np.mean(train_loss):.4f}, train-acc: {(100 * correct / total):.4f}')
        batch_loss = 0
        total_t = 0
        correct_t = 0

        with torch.no_grad():
            net.eval()
            for data_t, target_t in (testset):
                data_t, target_t = data_t.to(device), target_t.to(device)
                outputs_t = net(data_t)
                loss_t = criterion(outputs_t, target_t)
                batch_loss += loss_t.item()
                _, pred_t = torch.max(outputs_t, dim=1)
                correct_t += torch.sum(pred_t == target_t).item()
                total_t += target_t.size(0)
            val_acc.append(100 * correct_t / total_t)
            val_loss.append(batch_loss / len(testset))
            network_learned = batch_loss < valid_loss_min
            print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t / total_t):.4f}\n')

            if network_learned:
                valid_loss_min = batch_loss
                torch.save(net.state_dict(), 'resnet.pt')
                print('Improvement-Detected, save-model')
        net.train()

    fig = plt.figure(figsize=(20, 10))
    print("Plotting")
    plt.title("Train-Validation Accuracy")
    plt.plot(train_acc, label='train')
    plt.plot(val_acc, label='validation')
    plt.xlabel('num_epochs', fontsize=12)
    plt.ylabel('accuracy', fontsize=12)
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':

    # print("With Augmented")
    # test('bindata/ImgsTrain', 'bindata/ImgsTest')

    print("Without Augmented")
    test('bindata/NAImgsTrain', 'bindata/NAImgsTest')

