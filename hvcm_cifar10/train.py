from __future__ import print_function

import argparse
import csv
import os, logging

import numpy as np
import torch
from torch.autograd import Variable, grad
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

import models
from utils import *
from datasets import load_dataset

parser = argparse.ArgumentParser(description='HVCM Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--model', default="CIFAR_ResNet18", type=str, help='model type (32x32: CIFAR_ResNet18, CIFAR_ResNet34, ResNet18Gram, ResNet34Gram)')
parser.add_argument('--name', default='0', type=str, help='name of run')
parser.add_argument('--batch-size', default=128, type=int, help='batch size')
parser.add_argument('--epoch', default=200, type=int, help='total epochs to run')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--ngpu', default=1, type=int, help='number of gpu')
parser.add_argument('--sgpu', default=0, type=int, help='gpu index (start)')
parser.add_argument('--dataset', default='cifar10', type=str, help='the name for dataset')
parser.add_argument('--dataroot', default='data/', type=str, help='data directory')
parser.add_argument('--saveroot', default='results/', type=str, help='save directory')

parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')

# ---- HVCM Hyperparameter ----
parser.add_argument('--hvcm', action='store_true', help='adding hvcm loss')
parser.add_argument('--attri_dim', default=1024, type=int, help="""Dimensionality of
        the HVCM head output. For complex and large datasets large values work well.""")
parser.add_argument('--num_kernel', default=32, type=int, help="Number of Gaussian components of GMM")
parser.add_argument('--alpha', default=1.6e-4, type=float, help="weight for KL(a||mu)")
parser.add_argument('--beta', default=1.6e-5, type=float, help="weight for KL(mu||a)")
parser.add_argument('--gamma', default=1e-4, type=float, help="weight for KL(mu||a)")

args = parser.parse_args()
use_cuda = torch.cuda.is_available()

best_val = 0  # best validation accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

cudnn.benchmark = True

# Data
print('==> Preparing dataset: {}'.format(args.dataset))
trainloader, valloader = load_dataset(args.dataset, args.dataroot, batch_size=args.batch_size)

num_class = trainloader.dataset.num_classes
print('Number of train dataset: ' ,len(trainloader.dataset))
print('Number of validation dataset: ' ,len(valloader.dataset))

# Model
print('==> Building model: {}'.format(args.model))

net = models.load_model(args.model, num_class)
embed_dim = net.linear.weight.shape[1]

# --- Add HVCM projection ---
if args.hvcm:
    net = NetworkWrapper(net, HVCMhead(
        embed_dim, 
        args.attri_dim, 
        num_kernel=args.num_kernel
    ))
# ---------------------------

if use_cuda:
    torch.cuda.set_device(args.sgpu)
    net.cuda()
    print(torch.cuda.device_count())
    print('Using CUDA..')

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.sgpu, args.sgpu + args.ngpu)))

if not args.hvcm:
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)
# --- Init HVCM loss ---
else:
    hvcmloss = HVCMLoss(num_class, feat_dim=[args.num_kernel, args.attri_dim // args.num_kernel])
    if use_cuda:
        hvcmloss = hvcmloss.cuda()
    optimizer = optim.SGD([{'params': net.parameters()}, 
                           {'params': hvcmloss.parameters()}], lr=args.lr, momentum=0.9, weight_decay=args.decay)
# ---------------------

logdir = os.path.join(args.saveroot, args.dataset, args.model, args.name)
set_logging_defaults(logdir, args)
logger = logging.getLogger('main')
logname = os.path.join(logdir, 'log.csv')


# Resume
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(os.path.join(logdir, 'ckpt.t7'))
    net.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1
    rng_state = checkpoint['rng_state']
    torch.set_rng_state(rng_state)
    # --- HVCM ---
    if args.hvcm:
        if 'hvcmloss' in checkpoint.keys():
            hvcmloss.load_state_dict(checkpoint['hvcmloss'])
        else:
            raise ValueError("hvcmloss not in this checkpoint!")
    # ------------

criterion = nn.CrossEntropyLoss()

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    train_hvcm_loss = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        batch_size = inputs.size(0)

        if not args.hvcm:
            outputs = net(inputs)
            loss = torch.mean(criterion(outputs, targets))

            train_loss += loss.item()
        # --- Add HVCM loss ---
        else:
            outputs, (weights, attri) = net(inputs)
            celoss = torch.mean(criterion(outputs, targets))
            hvloss = hvcmloss(weights, attri, targets, epoch, args)
            loss = hvloss

            train_loss += celoss.item()
            train_hvcm_loss += hvloss.item()
        # ---------------------

        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).sum().float().cpu()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        progress_bar(batch_idx, len(trainloader),
                     'Loss: %.3f | Acc: %.3f%% (%d/%d) | HVCM: %.3f'
                     % (train_loss/(batch_idx+1), 
                        100.*correct/total, 
                        correct, 
                        total, 
                        # --- hvcm loss ---
                        train_hvcm_loss/(batch_idx+1) / args.beta))

    logger = logging.getLogger('train')
    logger.info('[Epoch {}] [Loss {:.3f}] [hvcm {:.3f}] [Acc {:.3f}]'.format(
        epoch,
        train_loss/(batch_idx+1),
        # --- hvcm loss ---
        train_hvcm_loss/(batch_idx+1) / args.beta,
        100.*correct/total))

    acc = 100.*correct/total
    checkpoint(acc, epoch)

    if epoch % args.saveckp_freq == 0:
        checkpoint(acc, epoch, f'{epoch:04}')    

    return train_loss/batch_idx, 100.*correct/total


def checkpoint(acc, epoch, suffix=''):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state(),
    }
    if args.hvcm:
        state['hvcmloss'] = hvcmloss.state_dict()
    torch.save(state, os.path.join(logdir, f'ckpt{suffix}.t7'))


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = args.lr
    if epoch >= 0.5 * args.epoch:
        lr /= 10
    if epoch >= 0.75 * args.epoch:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# Logs
for epoch in range(start_epoch, args.epoch):
    train_loss, train_acc = train(epoch)
    adjust_learning_rate(optimizer, epoch)
