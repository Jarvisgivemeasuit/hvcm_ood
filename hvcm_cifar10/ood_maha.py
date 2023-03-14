import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from progress.bar import Bar

import models
from utils import *
from datasets import load_dataset

parser = argparse.ArgumentParser(description='OOD Detection on CIFAR10')
parser.add_argument('--model', default="CIFAR_ResNet18", type=str, help='model type')
parser.add_argument('--model_id', default="", type=str, help='checkpoint id in model storage')
parser.add_argument('--name', default='0', type=str, help='name of run')
parser.add_argument('--batch-size', default=128, type=int, help='batch size')
parser.add_argument('--dataset', default='cifar100', type=str, help='the name for dataset')
parser.add_argument('--ind_path', default='data/', type=str, help='ind data directory')
parser.add_argument('--ood_path', default='data/ood/', type=str, help='ood data directory')
parser.add_argument('--saveroot', default='results/', type=str, help='save directory')
parser.add_argument('--num_classes', default=100, type=int, help='Number of labels for linear classifier')

parser.add_argument('--attri_dim', default=1024, type=int, help="""Dimensionality of
        the HVCM head output. For complex and large datasets large values work well.""")
parser.add_argument('--num_kernel', default=32, type=int, help="Number of Gaussian components of GMM")

args = parser.parse_args()
use_cuda = torch.cuda.is_available()


def ood_detect(args):
    print("==> Preparing Model..")
    logdir = os.path.join(args.saveroot, args.dataset, args.model, args.name)
    checkpoint = torch.load(os.path.join(logdir, f'ckpt{args.model_id}.t7'))

    net = models.load_model(args.model, args.num_classes)
    embed_dim = net.linear.weight.shape[1]
    net = NetworkWrapper(net, HVCMhead(
        embed_dim, 
        args.attri_dim, 
        num_kernel=args.num_kernel
    ))
    net.load_state_dict(checkpoint['net'])

    print("==> Preparing GMM..")
    means = torch.load(os.path.join(logdir, 'means.pt'))
    cinvs = torch.load(os.path.join(logdir, 'covs_inv.pt'))
    weights = checkpoint['hvcmloss']['gmm_weights'].cpu()

    if use_cuda:
        net = net.cuda()
        means, cinvs, = means.cuda(), cinvs.cuda()

    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    # CIFAR10
    _, indloader = load_dataset(args.dataset, args.ind_path, batch_size=args.batch_size)
    results_in = get_scores(net, indloader, 
                            means, cinvs, weights, 
                            args.attri_dim, args.num_kernel)
    
    datasets = ['texture', 'svhn', 'places', 'iSUN', 'LSUN','LSUN_resize']
    for data_name in datasets:
        oodloader = get_loader(data_name, args.ood_path, transform_test)
        results_out = get_scores(net, oodloader,
                                means, cinvs, weights, 
                                args.attri_dim, args.num_kernel)

        auroc, fpr95, aupr = get_results(results_in, results_out)
        print(f"AUROC:{auroc * 100:.2f}, FPR95:{fpr95 * 100:.2f}, AUPR:{aupr * 100:.2f}")
        print()

    
def get_loader(data_name, data_path, transform):
    print(f'==> Preparing {data_name}..')

    if data_name == 'svhn':
        oodset = SVHN(os.path.join(data_path, 'selected_svhn_32x32.mat'), 
                      transform=transform)
    elif data_name == 'texture':
        oodset = datasets.ImageFolder(os.path.join(data_path, 'dtd', 'images'),
                                      transform=transform)
    else:
        oodset = datasets.ImageFolder(os.path.join(data_path, data_name),
                                      transform=transform)
    ood_loader = DataLoader(oodset, 
                            batch_size=args.batch_size, 
                            shuffle=True,
                            num_workers=4,
                            pin_memory=True)
    print(f'Data loaded with {len(oodset)} out-of-distribuion test images.')
    return ood_loader


class SVHN(Dataset):
    def __init__(self, data_path, transform) -> None:
        super().__init__()

        from scipy import io
        self.matdata = io.loadmat(data_path)
        self.transform = transform
        self.X = self.matdata['X'].transpose(3, 0, 1, 2)
        self.Y = self.matdata['y']

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, index):
        img = self.X[index]
        img = self.transform(img)
        
        return img, self.Y[index]
    

def get_scores(model, loader, means, covs_inv, gmm_weights, out_dim, num_kernel):
    num_iter = len(loader)
    bar = Bar('Getting results:', max=num_iter)
    results = []

    for idx, (inp, _) in enumerate(loader):
        if (idx + 1) > 79:
            break 
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        # forward
        with torch.no_grad():
            _, (_, q) = model(inp)
        maha = get_maha_score(means, 
                              covs_inv, 
                              gmm_weights, 
                              q.reshape(-1, 32, out_dim // num_kernel))
        output = maha.cpu().tolist()
        results.extend(output)

        bar.suffix = '({batch}/{size}) | Total:{total:} | ETA:{eta:}'.format(
            batch=idx + 1,
            size=num_iter,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            )
        bar.next()
    bar.finish()
    return results


def get_maha_score(mu, cov_inv, gmm_weights, x):
    '''
    Args:
        x: features of input with shape (num_samples, kernels, dimensions)
        mu: centers of gmm of all classes with shape (classes, kernels, dimensions)
        det_sigma: Determinant of covariance matrix with shape (classes, kernels)
        cov_inv: The inverse matrix of sigma which has shape (classes, kernels, dimensions, dimensions)
        gmm_weights: weights of gmm with shape (classes, kernels)
    '''
    cls, kers, dims = mu.shape
    num = x.shape[0]

    for i in range(cls):
        # expand mean and cov+inv
        mu_ = mu[i:i+1].expand(num, kers, dims)
        cov_inv_ = cov_inv[i:i+1].expand(num, kers, dims, dims)

        # reshape for calculation
        x = x.reshape(-1, 1, dims).double()
        mu_ = mu_.reshape(-1, 1, dims).double()
        cov_inv_ = cov_inv_.reshape(-1, dims, dims)

        # calculate the maha distance: (x-μ)Σ^(-1)(x-μ)^T
        maha = torch.bmm((x - mu_), cov_inv_)
        maha =  0.5 * torch.bmm(maha, (x - mu_).permute(0, 2, 1)).reshape(num, 1, kers)
        maha = (maha.cpu() * gmm_weights[i]).sum(-1)
        if i == 0:
            mahas = maha
        else:
            mahas = torch.cat([mahas, maha], dim=1)
    min_maha, _ = mahas.min(1)

    return min_maha


def get_results(res_in, res_out):
    tar_in, tar_out = np.zeros(len(res_in)), np.ones(len(res_out))
    # tar_in, tar_out = np.ones(len(res_in)), np.zeros(len(res_out))
    res, tar = [], []
    res.extend(res_in)
    res.extend(res_out)
    tar.extend(tar_in.tolist())
    tar.extend(tar_out.tolist())
    
    auroc = calc_auroc(res, tar)
    fpr95 = calc_fpr(res, tar)
    aupr = average_precision_score(tar, res)
    return auroc, fpr95, aupr


def calc_fpr(scores, trues):
    tpr95=0.95
    fpr, tpr, thresholds = roc_curve(trues, scores)
    fpr0=0
    tpr0=0
    for i,(fpr1,tpr1) in enumerate(zip(fpr,tpr)):
        if tpr1>=tpr95:
            break
        fpr0=fpr1
        tpr0=tpr1
    fpr95 = ((tpr95-tpr0)*fpr1 + (tpr1-tpr95)*fpr0) / (tpr1-tpr0)
    return fpr95


def calc_auroc(scores, trues):
    #calculate the AUROC
    result = roc_auc_score(trues, scores)

    return result

if __name__ == '__main__':
    ood_detect(args)