import argparse
import os
import time
import random


import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torch.optim as optim
import numpy as np

from utils import image_load, top1AccPlot
from models import resnet, vgg
from train import train
from test import test

parser = argparse.ArgumentParser(description='Intern programming')
# Optimization options
parser.add_argument('--arch', '-a', metavar='ARCH',
                    help='model archiecture(teacher or student)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')

parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
args = parser.parse_args()

def main():

    start_epoch = 0
    epochs = 150
    cuda = '0'
    res_train= []
    res_test = []

    # Use CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    use_cuda = torch.cuda.is_available()

    #Data
    trainLoader, testLoader = image_load(args.train_batch, args.test_batch)

    #Load Network
    if(args.arch == 'teacher'):
        teacher_net = resnet.resnet50(pretrained=True)

        #fine_tuning

        teacher_net.avgpool = nn.AvgPool2d(1, stride=1)
        teacher_net.fc = nn.Linear(2048, 100)
        model = teacher_net

    elif(args.arch == 'student'):
        student_net = vgg.vgg11(pretrained=False)
        model = student_net

    model = model.cuda()

    #nn.CrossEntropyLoss에 softmax가 포함되어 있다.

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    for epoch in range(start_epoch, epochs):

        trainTop1Val = train(trainLoader, model, criterion, optimizer, epoch , print_freq=100)
        testTop1Val = test(testLoader, model,criterion, epoch, print_freq=100)
        a = model.state_dict()
        b = optimizer.state_dict()
        state = {'epoch': epoch+1,
                 'arch': 'teacherNet',
                 'state_dict': model.state_dict(),
                 'optimizer': optimizer.state_dict()
                 }
        filename =  'studentNet_'+'checkpoint.pth'
        torch.save(state, filename)

        res_train.append(trainTop1Val)
        res_test.append(testTop1Val)

    print(res_test, res_train)
    top1AccPlot(res_train, res_test, epoch)





if __name__ == '__main__':
    main()



