import torch
import json
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from imgaug import augmenters as iaa
from torch.utils.data import DataLoader, Dataset
from torchvision import *
from sklearn import preprocessing
import python_speech_features as mfcc
from scipy.io import wavfile
from collections import Counter
from sklearn.metrics import confusion_matrix
from torchsummary import summary



print("Running")

height = 64
width = 64
batch_size = 32
n_epochs = 60
list_of_accs = []

"""

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
    # if (all_files[i][:12] == 'zin_pcm_airp'):
    #     label = 0
    #     train_all_labels.append(label)
    # elif (all_files[i][:12] == 'zin_pcm_metr'):
    #     label = 1
    #     train_all_labels.append(label)
    # elif (all_files[i][:12] == 'zin_pcm_shop'):
    #     label = 2
    #     train_all_labels.append(label)
    # elif (all_files[i][:12] == 'zout_pcm_par'):
    #     label = 3
    #     train_all_labels.append(label)
    # elif (all_files[i][:12] == 'zout_pcm_pub'):
    #     label = 4
    #     train_all_labels.append(label)
    # else:
    #     label = 5
    #     train_all_labels.append(label)

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


transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

# train_dataset = datasets.ImageFolder('bindata/BinNAImgsTrain', transform=transforms)
# test_dataset = datasets.ImageFolder('bindata/BinNAImgsTest', transform=transforms)

train_dataset = VectorDataset(data=train_all_data, labels=train_all_labels)
test_dataset = VectorDataset(data=test_all_data, labels=test_all_labels)
#
# train_dataset = MFCCDataset(data=train_data_list, labels=train_labels_list)
# test_dataset = MFCCDataset(data=test_data_list, labels=test_labels_list)

trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

class ZFNet(nn.Module):

    def __init__(self, channels, class_count):
        super(ZFNet, self).__init__()
        self.channels = channels
        self.class_count = class_count

        self.conv_net = self.get_conv_net()
        self.fc_net = self.get_fc_net()

    def get_conv_net(self):
        layers = []

        # in_channels = self.channels, out_channels = 96
        # kernel_size = 7x7, stride = 2
        layer = nn.Conv2d(
            self.channels, 96, kernel_size=7, stride=2, padding=1)
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
        layers.append(layer)
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        layers.append(nn.LocalResponseNorm(5))

        # in_channels = 96, out_channels = 256
        # kernel_size = 5x5, stride = 2
        layer = nn.Conv2d(96, 256, kernel_size=5, stride=2)
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
        layers.append(layer)
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        layers.append(nn.LocalResponseNorm(5))

        # in_channels = 256, out_channels = 384
        # kernel_size = 3x3, stride = 1
        layer = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
        layers.append(layer)
        layers.append(nn.ReLU(inplace=True))

        # in_channels = 384, out_channels = 384
        # kernel_size = 3x3, stride = 1
        layer = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
        layers.append(layer)
        layers.append(nn.ReLU(inplace=True))

        # in_channels = 384, out_channels = 256
        # kernel_size = 3x3, stride = 1
        layer = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
        layers.append(layer)
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2))

        return nn.Sequential(*layers)

    def get_fc_net(self):
        layers = []

        # in_channels = 9216 -> output of self.conv_net
        # out_channels = 4096
        layer = nn.Linear(9216, 4096)
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
        layers.append(layer)
        layers.append(nn.Dropout())

        # in_channels = 4096
        # out_channels = self.class_count
        layer = nn.Linear(4096, self.class_count)
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
        layers.append(layer)
        layers.append(nn.Dropout())

        return nn.Sequential(*layers)

    def forward(self, x):
        y = self.conv_net(x)
        y = y.view(-1, 9216)
        y = self.fc_net(y)
        return y

class autoencoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        # nn.Linear creates linear functions (ax+b)

        # self.encoder_hidden_layer_1 = nn.Linear(in_features=4096, out_features=2048)
        # self.encoder_hidden_layer_2 = nn.Linear(in_features=2048, out_features=1024)
        # self.encoder_output_layer = nn.Linear(in_features=1024, out_features=512)
        # self.decoder_hidden_layer_1 = nn.Linear(in_features=512, out_features=1024)
        # self.decoder_hidden_layer_2 = nn.Linear(in_features=1024, out_features=2048)
        # self.decoder_output_layer = nn.Linear(in_features=2048, out_features=4096)

        self.encoder_hidden_layer_1 = nn.Linear(in_features=640, out_features=320)
        self.encoder_hidden_layer_2 = nn.Linear(in_features=320, out_features=160)
        self.encoder_hidden_layer_3 = nn.Linear(in_features=160, out_features=80)
        self.encoder_output_layer = nn.Linear(in_features=80, out_features=2)
        self.decoder_hidden_layer_0 = nn.Linear(in_features=2, out_features=80)
        self.decoder_hidden_layer_1 = nn.Linear(in_features=80, out_features=160)
        self.decoder_hidden_layer_2 = nn.Linear(in_features=160, out_features=320)
        self.decoder_output_layer = nn.Linear(in_features=320, out_features=640)
        # For Binary
        # self.fc = nn.Linear(4096, 2)
        # For all Classes
        self.fc = nn.Linear(640, 2)

    def forward(self, x):
        # print(x.shape)

        x = x.view(-1, 640)
        # print("0.5", x.shape)

        x = self.encoder_hidden_layer_1(x)
        x = torch.relu(x)
        x = self.encoder_hidden_layer_2(x)
        x = torch.relu(x)
        x = self.encoder_hidden_layer_3(x)
        x = torch.relu(x)
        x = self.encoder_output_layer(x)
        x = torch.relu(x)

        x = self.decoder_hidden_layer_0(x)
        x = torch.relu(x)
        x = self.decoder_hidden_layer_1(x)
        x = torch.relu(x)
        x = self.decoder_hidden_layer_2(x)
        x = torch.relu(x)
        x = self.decoder_output_layer(x)
        x = torch.relu(x)

        # print("1.5", x.shape)

        x = self.fc(x)
        # print(x.shape)
        return x

class bottleNeck(nn.Module):
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

class resnet(nn.Module):
    def __init__(self, block, num_layers, classes=6):
        super(resnet, self).__init__()
        # according to research paper:

        self.input_planes = 32
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=2, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.layer1 = self._layer(block, 32, num_layers[0], stride=2)
        self.layer2 = self._layer(block, 64, num_layers[1], stride=2)
        self.layer3 = self._layer(block, 128, num_layers[2], stride=2)
        self.layer4 = self._layer(block, 256, num_layers[3], stride=2)
        self.averagePool = torch.nn.AvgPool2d(kernel_size=4, stride=1)
        # For MFCCs
        # self.fc = torch.nn.Linear(20480, classes)
        # For Vectors
        self.fc = torch.nn.Linear(1024, classes)
        # For Spectro
        # self.fc = torch.nn.Linear(50176, classes)
        self.dropout = torch.nn.Dropout(0.10)

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
        # print(1, x.shape)

        x = torch.reshape(x, (x.shape[0], 1, 32, 20))
        # for Vector

        # x = torch.reshape(x, (x.shape[0], 1, 167, 120))
        # # for MFCC
        # print(2, x.shape)

        x = self.conv1(x)
        # print(2.5, x.shape)

        x = self.bn1(x)
        # print(2.75, x.shape)

        x = self.layer1(x)
        # print(4, x.shape)

        x = self.layer2(x)
        # print(5, x.shape)

        x = self.layer3(x)
        # print(6, x.shape)

        x = self.layer4(x)
        # print(7, x.shape)

        x = F.avg_pool2d(x, 2)
        # print(8, x.shape)

        x = x.view(x.size(0), -1)
        # print(9, x.shape)

        x = self.dropout(x)

        x = self.fc(x)
        # print(10, x.shape)

        return x


net = autoencoder()
# net = ZFNet(3, 2)
# net = resnet(bottleNeck, [2, 2, 2, 2])
# net = resnet(bottleNeck, [3, 4, 6, 3])

net = net.cuda() if device else net

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
# optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.0001, betas=(0.9, 0.999),
#                              eps=1e-08, weight_decay=0.0001, amsgrad=False)
use_cuda = torch.cuda.is_available()


def accuracy(out, labels):
    _, pred = torch.max(out, dim=1)
    return torch.sum(pred == labels).item()


print_every = 200
valid_loss_min = np.Inf
val_loss = []
val_acc = []
train_loss = []
train_acc = []
total_step = len(trainloader)

# summary(net.cuda(), (1, 32, 20))

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

    confusion_matrix = torch.zeros(6, 6)
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
            for t, p in zip(target_t.view(-1), pred_t.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

        val_acc.append(100 * correct_t / total_t)
        val_loss.append(batch_loss / len(testloader))
        network_learned = batch_loss < valid_loss_min
        print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t / total_t):.4f}\n')

        if network_learned:
            valid_loss_min = batch_loss
            torch.save(net.state_dict(), 'resnet.pt')
            print('Improvement-Detected, save-model')


    print(confusion_matrix)
    acc_per_class = confusion_matrix.diag() / confusion_matrix.sum(1)
    print(acc_per_class)
    list_of_accs.append(acc_per_class)
    net.train()

fig = plt.figure(figsize=(20, 10))
plt.title("Train-Validation Accuracy")
plt.plot(train_acc, label='train')
plt.plot(val_acc, label='validation')
plt.xlabel('num_epochs', fontsize=12)
plt.ylabel('accuracy', fontsize=12)
plt.legend(loc='best')
plt.show()

