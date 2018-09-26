import time
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torch.optim as optim
import numpy as np
from utils import AverageMeter

from train import accuracy

def test(testLoader, model, criterion,epoch ,print_freq):


    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    #set model mode to evaluation, batch_norm or dropout layers wlll work in eval model instead of training mode
    model.eval()

    #torch.no_grad sets deactivate autograd engine and speed up computations but no backprop
    with torch.no_grad():
        end = time.time()

    for i , (input, target) in enumerate(testLoader):
        #data loading time
        data_time.update(time.time() - end)
        input = input.cuda()
        target = target.cuda()

        output = model(input)
        loss = criterion(output, target)

        prec1, prec5= accuracy(output, target, topk=(1,5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Test\t'
                  'Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                 epoch, i, len(testLoader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

    print('TOP1 PRECISION: {top1.avg:.3f} TOP5 PRECISION: {top5.avg:.3f}'.format(
        top1=top1, top5=top5)
    )
    return top1.val



