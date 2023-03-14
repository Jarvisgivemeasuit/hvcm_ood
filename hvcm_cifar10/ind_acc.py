import argparse
import torch
from progress.bar import Bar

import models
from utils import *
from datasets import load_dataset

parser = argparse.ArgumentParser(description='HVCM InD Acc')
parser.add_argument('--model', default="CIFAR_ResNet18", type=str, help='model type')
parser.add_argument('--name', default='0', type=str, help='name of run')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--dataset', default='cifar10', type=str, help='the name for dataset')
parser.add_argument('--dataroot', default='data/', type=str, help='data directory')
parser.add_argument('--saveroot', default='results/', type=str, help='save directory')
parser.add_argument('--num_classes', default=10, type=int, help='Number of labels for linear classifier')

parser.add_argument('--attri_dim', default=1024, type=int, help="""Dimensionality of
        the HVCM head output. For complex and large datasets large values work well.""")
parser.add_argument('--num_kernel', default=32, type=int, help="Number of Gaussian components of GMM")

args = parser.parse_args()
use_cuda = torch.cuda.is_available()


def ind_acc(args):
    print("==> Preparing model..")
    logdir = os.path.join(args.saveroot, args.dataset, args.model, args.name)
    checkpoint = torch.load(os.path.join(logdir, 'ckpt.t7'))

    net = models.load_model(args.model, args.num_classes)
    embed_dim = net.linear.weight.shape[1]
    net = NetworkWrapper(net, HVCMhead(
        embed_dim, 
        args.attri_dim, 
        num_kernel=args.num_kernel
    ))
    net.load_state_dict(checkpoint['net'])

    print("==> Preparing group centers..")
    centers = checkpoint['hvcmloss']['centers']
    clses, kers, dims = centers.shape
    acc = Accuracy()

    if use_cuda:
        net = net.cuda()

    print("==> Preparing val dataset..")
    _, val_loader = load_dataset(args.dataset, args.dataroot, batch_size=args.batch_size)
    num_batch = len(val_loader)
    bar = Bar('Calculating ind_acc:', max=num_batch)

    for idx, (x, cls) in enumerate(val_loader):
        x, cls = x.cuda(), cls.cuda()
        with torch.no_grad():
            _, (_, q) = net(x)
        bs = q.shape[0]

        centers_ = centers.unsqueeze(0).expand(bs, clses, kers, dims)
        q = q.unsqueeze(1).expand(bs, clses, kers * dims).reshape(-1, dims)

        centers_ = centers_.reshape(-1, clses, kers * dims).cuda()
        q = q.reshape(-1, clses, kers * dims)

        # calculate cosine similarity
        results = torch.cosine_similarity(q, centers_, dim=-1)
        results = torch.argmax(results, 1)

        acc.update(results, cls)
        bar.suffix = f'{idx+1}/{num_batch}, acc:{(acc.get_top1()*100):.3f}'
        bar.next()
    bar.finish()

if __name__ == '__main__':
    ind_acc(args)