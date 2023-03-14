import argparse
import os

import models
from utils import *
from datasets import *


parser = argparse.ArgumentParser(description='GMM')
parser.add_argument('--model', default="CIFAR_ResNet18", type=str, help='model type')
parser.add_argument('--model_id', default="", type=str, help='checkpoint id in model storage')
parser.add_argument('--name', default='0', type=str, help='name of run')
parser.add_argument('--ngpu', default=1, type=int, help='number of gpu')
parser.add_argument('--sgpu', default=0, type=int, help='gpu index (start)')
parser.add_argument('--dataset', default='cifar10', type=str, help='the name for dataset')
parser.add_argument('--dataroot', default='data/', type=str, help='data directory')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--saveroot', default='results/', type=str, help='save directory')
parser.add_argument('--num_kernel', default=32, type=int, help="Number of Gaussian components of GMM")
parser.add_argument('--attri_dim', default=1024, type=int, help="""Dimensionality of
        the HVCM head output. For complex and large datasets large values work well.""")
args = parser.parse_args()
use_cuda = torch.cuda.is_available()
assert "cifar" in args.dataset

print('==> Preparing dataset: {}'.format(args.dataset))
num_classes = 100 if args.dataset == "cifar100" else 10
trainloader, _ = load_dataset(args.dataset, args.dataroot, batch_size=args.batch_size)

print('==> Building model: {}'.format(args.model))
net = models.load_model(args.model, num_classes)
embed_dim = net.linear.weight.shape[1]
net = NetworkWrapper(net, HVCMhead(
    embed_dim, 
    args.attri_dim, 
    num_kernel=args.num_kernel
))

if use_cuda:
    torch.cuda.set_device(args.sgpu)
    net.cuda()
    print(torch.cuda.device_count())
    print('Using CUDA..')

logdir = os.path.join(args.saveroot, args.dataset, args.model, args.name)

print('==> Resuming from checkpoint..')
checkpoint = torch.load(os.path.join(logdir, f'ckpt{args.model_id}.t7'))
net.load_state_dict(checkpoint['net'])
rng_state = checkpoint['rng_state']
torch.set_rng_state(rng_state)

ind_means = torch.tensor([])
ind_covs_inv = torch.tensor([])

# Get all samples attributes
attris = [[] for i in range(num_classes)]
for x, y in trainloader:
    x = x.cuda()
    with torch.no_grad():
        _, (_, attri) = net(x)
    for i in range(x.shape[0]):
        attris[y[i]].append(attri[i].reshape(1, 1, *attri[i].shape))

attris = [torch.cat(attris[i], dim=1) for i in range(len(attris))]
attris = torch.cat(attris).cpu()
print(f"==> Sample attris shape: {attris.shape}")

# Class-by-class calculation of the Mean and Cov.
for i in range(num_classes):
    print(f"==> Preparing class {i}..")
    attri = attris[i]
    attri = attri.reshape(attri.shape[0], args.num_kernel, -1)
    dim = attri.shape[-1]

    cls_mean = torch.tensor([]).cuda()
    cls_cov = torch.tensor([]).cuda()

    for gau in range(args.num_kernel):
        samples = attri[:, gau:gau + 1].cuda()
        mean = samples.mean(0)

        # Whitening
        samples = (samples - mean).reshape(-1, dim, 1).double()

        # Dot product
        cov = torch.bmm(samples, samples.permute(0, 2, 1))

        # Unbiased estimation
        cov = cov.reshape(-1, 1, dim, dim).sum(0)
        cov = (cov / (attri.shape[0] - 1))
        de = torch.linalg.cholesky(cov)

        cls_mean = torch.cat([cls_mean, mean], dim=0)
        cls_cov = torch.cat([cls_cov, cov], dim=0)
        
    cls_mean = cls_mean.cpu().unsqueeze(0)
    cls_cov = cls_cov.cpu().unsqueeze(0)

    ind_means = torch.cat([ind_means, cls_mean], dim=0)
    ind_covs_inv = torch.cat([ind_covs_inv, torch.linalg.inv(cls_cov)], dim=0)

torch.save(ind_means, os.path.join(logdir, 'means.pt'))
torch.save(ind_covs_inv, os.path.join(logdir, 'covs_inv.pt'))