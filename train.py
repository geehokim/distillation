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

from utils import AverageMeter

def train(trainLoader, model, criterion, optimizer, epoch,  print_freq):

    model.train()

    batch_time = AverageMeter()
    data_time= AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(trainLoader):
        # measure data loading time
        data_time.update(time.time() - end)

        #if cuda is not None:
        #   input = input.cuda(cuda, non_blocking=True)
        #target = target.cuda(cuda, non_blocking=True)

        input = input.cuda()
        target = target.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Train\t'
                  'Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(trainLoader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
    return top1.val

def distilliation_train(trainLoader, model, criterion, optimizer, epoch,  print_freq):

    model.train()

    batch_time = AverageMeter()
    data_time= AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(trainLoader):
        # measure data loading time
        data_time.update(time.time() - end)

        #if cuda is not None:
        #   input = input.cuda(cuda, non_blocking=True)
        #target = target.cuda(cuda, non_blocking=True)

        input = input.cuda()
        target = target.cuda()

        # compute output
        output = model(input)

        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Train\t'
                  'Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(trainLoader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
    return top1.val

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        a = output
        k = output.topk(maxk, 1, True, True)
        values, indices = output.topk(maxk, 1, True, True)
        pred = indices.t()

        correct = torch.eq(pred, target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            #correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res