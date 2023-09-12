from torch.utils.data import random_split
import random
import torch
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST, ImageNet
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from functions_new import create_dir_partition, create_pat_partition
#from exp_args import *

from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        return image, label

    def __len__(self):
        return len(self.images)

def data_2_loader(data_x, data_y, bs, test=False):
    train_dataset = CustomDataset(data_x, data_y)
    if test:
        loader = DataLoader(train_dataset, batch_size=bs, shuffle=False)
    else:
        loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    return loader

data_list = ['cifar10', 'cifar100', 'imagenet', 'fashionmnist']
model_list = ['vgg11', 'alexnet', 'resnet18', 'lenet5','resnet50']
cifar100_mean = [x / 255.0 for x in [0.507, 0.487, 0.441]]
cifar100_std = std=[x / 255.0 for x in [0.267, 0.256, 0.276]]



def load_data(data_name='cifar10'):
    """Load different datasets. And transform"""
    if data_name == 'cifar100':
        DATA_ROOT: str = "~/icdcsdata/cifar-100"
        transform_train = transforms.Compose(
            [transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
             transforms.RandomHorizontalFlip(), transforms.ToTensor(),
             transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])
        transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=cifar100_mean,
                                                                  std=cifar100_std)])
        trainset = CIFAR100(DATA_ROOT, train=True, download=True, transform=transform_train)
        testset = CIFAR100(DATA_ROOT, train=False, download=True, transform=transform_test)
        return trainset, testset
    elif data_name == 'fashionmnist':
        DATA_ROOT: str = "~/icdcsdata/fasion-mnist"
        transform_train = transforms.Compose([transforms.ToTensor()])
        transform_test = transforms.Compose([transforms.ToTensor()])
        trainset = FashionMNIST(DATA_ROOT, train=True, download=True, transform=transform_train)
        testset = FashionMNIST(DATA_ROOT, train=False, download=True, transform=transform_test)
        return trainset, testset
    elif data_name == 'imagenet':
        DATA_ROOT: str = "~/icdcsdata/image-net"
        transform_train = transforms.Compose([transforms.ToTensor()])
        transform_test = transforms.Compose([transforms.ToTensor()])
        trainset = ImageNet(root=DATA_ROOT, split='train', transform=transform_train)
        testset = ImageNet(root=DATA_ROOT, split='val', ransform=transform_test)
        return trainset, testset
    elif data_name == 'cifar10':
        DATA_ROOT: str = "~/icdcsdata/cifar-10"
        CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
        CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

        # train_transform.transforms.append(Cutout(16))

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

        trainset = CIFAR10(DATA_ROOT, train=True, download=True, transform=transform_train)
        testset = CIFAR10(DATA_ROOT, train=False, download=True, transform=transform_test)
        return trainset, testset

    else:
        exit('Unknown dataset. Please try datasets in hand.')


def preprocessed_data(data_list, batch_size, n_users, data_name='cifar10', num_work=2, partition_method='iid',
                      num_classes=10):
    if data_name in data_list:
        torch.manual_seed(123)
        # preparing the train, validation and test dataset
        print(f'Dataset experimented is: {data_name} with batch size {batch_size} and {n_users} users')
        if data_name == 'cifar100':
            num_classes=100
        train_ds, test_ds = load_data(data_name=data_name)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_work, pin_memory=True)
        test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_work, pin_memory=True)

        user_index = list(range(n_users))
        idx_users = list()

        # Extract dataset content and labels
        train_data = [data[0] for data in train_ds]
        train_labels = [data[1] for data in train_ds]
        test_data = [data[0] for data in test_ds]
        test_labels = [data[1] for data in test_ds]

        if partition_method == 'pat':
            train_dataidx_map = create_pat_partition((train_data, train_labels), num_clients=n_users,
                                                     num_classes=num_classes, batch_size=batch_size)
            test_dataidx_map = create_pat_partition((test_data, test_labels), num_clients=n_users,
                                                    num_classes=num_classes, batch_size=batch_size)
        elif partition_method == 'dir':
            train_dataidx_map = create_dir_partition((train_data, train_labels), num_clients=n_users,
                                                     num_classes=num_classes)
            test_dataidx_map = create_dir_partition((test_data, test_labels), num_clients=n_users,
                                                    num_classes=num_classes)
        else:  # Default to IID partitioning
            indices_by_label = [[] for _ in range(num_classes)]
            for i, label in enumerate(train_labels):
                indices_by_label[label].append(i)

            # Shuffle indices within each label
            for indices in indices_by_label:
                random.shuffle(indices)

            # Assign instances to clients, ensuring each client gets an equal distribution of labels
            idx_users = [[user_id, ([], [])] for user_id in range(n_users)]
            for label in range(num_classes):
                indices = indices_by_label[label]
                # Distribute instances of this label equally among users
                for i, index in enumerate(indices):
                    idx_users[i % n_users][1][0].append(index)

            # Repeat the process for the test set
            indices_by_label_test = [[] for _ in range(num_classes)]
            for i, label in enumerate(test_labels):
                indices_by_label_test[label].append(i)

            for indices in indices_by_label_test:
                random.shuffle(indices)

            for label in range(num_classes):
                indices = indices_by_label_test[label]
                for i, index in enumerate(indices):
                    idx_users[i % n_users][1][1].append(index)

            # Return even weights for IID case
            even_weights = [1 / n_users] * n_users
            #train_labels_all = []
            #test_labels_all = []

            #for user_id, (train_indices, test_indices) in idx_users:
            #    train_labels_for_user = [train_labels[i] for i in train_indices]
            #    test_labels_for_user = [test_labels[i] for i in test_indices]
            #    train_labels_all.append(train_labels_for_user)
            #    test_labels_all.append(test_labels_for_user)

            return [user_index, idx_users], (train_dl, test_dl), even_weights

        for client in user_index:
            idx_users.append([client, [train_dataidx_map[client], test_dataidx_map[client]]])

        # Calculate the number of samples for each client and generate the percentage weights
        total_samples = len(train_ds)
        client_sample_counts = [len(train_dataidx_map[client]) for client in user_index]
        client_percentage_weights = [sample_count / total_samples for sample_count in client_sample_counts]

        # Print the number of samples and percentage weights for each client
        for i, (sample_count, weight) in enumerate(zip(client_sample_counts, client_percentage_weights)):
            print(f"Client {i}: {sample_count} samples ({weight * 100:.2f}% of total)")

        return [user_index, idx_users], (train_dl, test_dl), client_percentage_weights
    else:
        print('Please try other datasets included.')


class LeNet(nn.Module):

    # network structure
    def __init__(self):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        '''
        One forward pass through the network.

        Args:
            x: input
        '''
        x = self.features(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.classifier(x)
        return x

    def num_flat_features(self, x):
        '''
        Get the number of features in a batch of tensors `x`.
        '''
        size = x.size()[1:]
        return np.prod(size)


import torch
import torch.nn as nn


class AlexNet(nn.Module):

    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        logits = self.classifier(x)
        probas = F.softmax(logits, dim=1)
        return logits

import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])



def model_selected(model_list, model_name='alexnet', pre_trained=False, data_name='cifar10'):
    if model_name in model_list:
        print('Model exists in the selected model ranges')
        if model_name == 'vgg11' and data_name == 'cifar10':
            print(f'Model {model_name} is matched with {data_name}')
            model = models.vgg11(pretrained=pre_trained)
            model.classifier[6] = nn.Linear(4096, 10, bias=True)
            return model
        elif model_name == 'resnet18' and data_name == 'cifar100':
            print(f'Model {model_name} is matched with {data_name}')
            model = ResNet18()
            return model
        elif model_name == 'resnet50' and data_name == 'cifar100':
            print(f'Model {model_name} is matched with {data_name}')
            model = ResNet50()
            return model
        elif model_name == 'lenet5' and data_name == 'fashionmnist':
            print(f'Model {model_name} is matched with {data_name}')
            model = LeNet()
            return model
        elif model_name == 'alexnet' and data_name =='cifar10':
            print(f'Model {model_name} is matched with {data_name}')
            model = AlexNet(num_classes=10)
            return model
        else:
            exit('(Warning) Dataset and model are not matched. No model output')
    else:
        print('Please try other models included')

