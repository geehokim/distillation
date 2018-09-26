#Utility

import numpy as np
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import matplotlib.pyplot as plt



def image_load(train_batch, test_batch):

    tra_batch =train_batch
    te_batch = test_batch

    print('데이터야 잘 나와라')
    # Data, transforms.RandomCrop(32, padding=4)은 아직 안하기
    # Data aug는 RandomHorizontalFlip만 사용
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    dataloader = datasets.CIFAR100
    num_classes = 1000


    trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=tra_batch, shuffle=True, num_workers=0)
    #traionloader를 클래스로 반환해서 enumerate를 시키면 index와 (배치이미지, 라벨)반환

    testset = dataloader(root='./data', train=False, download=True, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=te_batch, shuffle=False, num_workers=0)

    return trainloader, testloader


class AverageMeter(object):

    """Computes and stores the average and current value
           Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def top1AccPlot(res_train, res_test, epoch):
    x = []
    for i in range(epoch+1):
        x.append(i + 1)

    plt.plot(x, res_train, color='lightblue', linewidth=3)
    plt.plot(x, res_test, color='darkgreen', linewidth=3)

    plt.xlim(0.5, epoch + 10)
    plt.ylim(0, 100)
    plt.show()