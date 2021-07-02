import time
import torch
import matplotlib.pyplot as plt
from torch import nn, optim
import torchvision
import numpy as np
import sys
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import *
from collections import Counter
import json
from torchsummary import summary


print("Loading Indoor Training")
x_in_train = np.load('I:/Code/Python/AudioProcessing/MFCCsNpy/xmin_in_train.npy')

print("Loading Outdoor Training")
x_out_train = np.load('I:/Code/Python/AudioProcessing/MFCCsNpy/xmin_out_train.npy')

print("Loading Test")
test_data_list = np.load('I:/Code/Python/AudioProcessing/MFCCsNpy/xmin_test.npy')

train_data_list = np.concatenate((x_in_train, x_out_train))
print(train_data_list.shape)

train_labels_list = []
test_labels_list = []

for i in range(27648):
    if i < 13824:
        train_labels_list.append(0)
    else:
        train_labels_list.append(1)

print("Number of Training data:", Counter(train_labels_list))

for i in range(6912):
    if i < 3456:
        test_labels_list.append(0)
    else:
        test_labels_list.append(1)

print("Number of Training data:", Counter(test_labels_list))


class MFCCDataset(Dataset):
    def __init__(self, data, labels, transforms=None):

        self.X = data
        self.y = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):

        data = self.X[i].astype(np.float32)
        label = self.y[i]

        if self.transforms:
            data = self.transforms(self.X)

        if self.y is not None:
            return data, label

        else:
            return data, label
"""

train_data_loc = "JSON_Data/Train"
test_data_loc = "JSON_Data/Test"

train_all_data = []
train_all_labels = []
test_all_data = []
test_all_labels = []

all_files = os.listdir(train_data_loc)

for i in range(len(all_files)):

    # For binary
    if (all_files[i][:3] == 'zin'):
        label = 0
        train_all_labels.append(label)
    else:
        label = 1
        train_all_labels.append(label)

    # For multiclass
    # if (all_files[i][:12] == 'zin_zpcm_air'):
    #     label = 0
    #     test_all_labels.append(label)
    # elif (all_files[i][:12] == 'zin_zpcm_met'):
    #     label = 1
    #     test_all_labels.append(label)
    # elif (all_files[i][:12] == 'zin_zpcm_sho'):
    #     label = 2
    #     test_all_labels.append(label)
    # elif (all_files[i][:12] == 'zout_zpcm_pa'):
    #     label = 3
    #     test_all_labels.append(label)
    # elif (all_files[i][:12] == 'zout_zpcm_pu'):
    #     label = 4
    #     test_all_labels.append(label)
    # else:
    #     label = 5
    #     test_all_labels.append(label)

    with open(train_data_loc+'/'+all_files[i]) as json_file:
        data = json.load(json_file)
        data_arr = np.array(data["embedding"]).astype(np.uint8)
        train_all_data.append(data_arr)

all_files = os.listdir(test_data_loc)

for i in range(len(all_files)):

    # For Binary
    if (all_files[i][:3] == 'zin'):
        label = 0
        test_all_labels.append(label)
    else:
        label = 1
        test_all_labels.append(label)

    # For multiclass
    # if (all_files[i][:12] == 'zin_zpcm_air'):
    #     label = 0
    #     test_all_labels.append(label)
    # elif (all_files[i][:12] == 'zin_zpcm_met'):
    #     label = 1
    #     test_all_labels.append(label)
    # elif (all_files[i][:12] == 'zin_zpcm_sho'):
    #     label = 2
    #     test_all_labels.append(label)
    # elif (all_files[i][:12] == 'zout_zpcm_pa'):
    #     label = 3
    #     test_all_labels.append(label)
    # elif (all_files[i][:12] == 'zout_zpcm_pu'):
    #     label = 4
    #     test_all_labels.append(label)
    # else:
    #     label = 5
    #     test_all_labels.append(label)

    with open(test_data_loc+'/'+all_files[i]) as json_file:
        data = json.load(json_file)
        data_arr = np.array(data["embedding"]).astype(np.uint8)
        test_all_data.append(data_arr)

print(len(train_all_data), len(test_all_data), len(train_all_labels), len(test_all_labels))

"""

class VectorDataset(Dataset):
    def __init__(self, data, labels, transforms=None):

        # self.loc = data_loc
        # self.X = os.listdir(data_loc)

        self.X = data
        self.y = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):

        data = self.X[i].astype(np.float32)
        label = self.y[i]

        if self.transforms:
            data = self.transforms(self.X)

        if self.y is not None:
            return data, label

        else:
            return data, label


class AlexNet(nn.Module):
    def __init__(self, classes):
        super(AlexNet, self).__init__()

        self.conv = nn.Sequential(
        	# in_channels, out_channels, kernel_size, stride, padding
            nn.Conv2d(1, 256, 11, 5),
            nn.ReLU(),
            # kernel_size, stride
            nn.MaxPool2d(3, 2),
            # Reduce the convolution window, use a padding of 2 to make the input and output height and width consistent, and increase the number of output channels
            nn.Conv2d(256, 256, 3, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            # Consecutive 3 convolutional layers, and use a smaller convolution window. In addition to the final convolutional layer, the number of output channels is further increased.
            # After the first two convolutional layers, the pooling layer is not used to reduce the height and width of the input
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(1536, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, classes),
        )

    def forward(self, img):

        # img = torch.reshape(img, (img.shape[0], 1, 32, 20))
        # for Vector

        img = torch.reshape(img, (img.shape[0], 1, 167, 120))
        # for MFCC

        feature = self.conv(img)
        # print(img.shape)
        # print((feature.view(img.shape[0], -1)).shape)
        output = self.fc(feature.view(img.shape[0], -1))
        return output


def train_and_test_alexnet(train, test, classes):

    n_epochs = 40
    batch_size = 16

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    # train_dataset = datasets.ImageFolder(train, transform=transforms)
    # test_dataset = datasets.ImageFolder(test, transform=transforms)

    # train_dataset = VectorDataset(data=train_all_data, labels=train_all_labels)
    # test_dataset = VectorDataset(data=test_all_data, labels=test_all_labels)

    train_dataset = MFCCDataset(data=train_data_list, labels=train_labels_list)
    test_dataset = MFCCDataset(data=test_data_list, labels=test_labels_list)

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    use_cuda = torch.cuda.is_available()
    # New AlexNet network
    net = AlexNet(classes=classes)
    net = net.cuda() if device else net


    lr = 0.001
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.001)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.0001, betas=(0.9, 0.999),
                                                              eps=1e-08, weight_decay=0.0001, amsgrad=False)
    criterion = nn.CrossEntropyLoss()

    print_every = 200
    valid_loss_min = np.Inf
    val_loss = []
    val_acc = []
    train_loss = []
    train_acc = []
    total_step = len(trainloader)

    summary(net.cuda(), (1, 167, 120))

    for epoch in range(1, n_epochs + 1):
        running_loss = 0.0
        correct = 0
        total = 0
        print(f'Epoch {epoch}\n')
        for batch_idx, (data_, target_) in enumerate(trainloader):

            # print(data_, target_)
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

            for data_t, target_t in (testloader):
                data_t, target_t = data_t.to(device), target_t.to(device)
                outputs_t = net(data_t)
                loss_t = criterion(outputs_t, target_t)
                batch_loss += loss_t.item()
                _, pred_t = torch.max(outputs_t, dim=1)
                correct_t += torch.sum(pred_t == target_t).item()
                total_t += target_t.size(0)
            val_acc.append(100 * correct_t / total_t)
            val_loss.append(batch_loss / len(testloader))
            network_learned = batch_loss < valid_loss_min
            print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t / total_t):.4f}\n')

            if network_learned:
                valid_loss_min = batch_loss
                torch.save(net.state_dict(), 'resnet.pt')
                print('Improvement-Detected, save-model')

        net.train()


    fig = plt.figure(figsize=(20, 10))
    plt.title("Train-Validation Accuracy")
    plt.plot(train_acc, label='train')
    plt.plot(val_acc, label='validation')
    plt.xlabel('num_epochs', fontsize=12)
    plt.ylabel('accuracy', fontsize=12)
    plt.legend(loc='best')
    plt.show()


# Training on 6 classes without Aug
train_and_test_alexnet(train='bindata/NAImgsTrain', test='bindata/NAImgsTest', classes=2)
