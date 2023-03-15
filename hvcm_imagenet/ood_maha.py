import argparse
import os

import numpy as np
import torch
from progress.bar import Bar
from sklearn.metrics import roc_auc_score, roc_curve
from torchvision import datasets
from torchvision import models as torchvision_models
from torchvision import transforms

import utils
from dataset.imagenet import Imagenet
use_cuda = torch.cuda.is_available()


def ood_maha(args):
    print('==> Preparing model..')
    model = torchvision_models.__dict__[args.arch]()
    embed_dim = model.fc.weight.shape[1]
    model = utils.MultiCropWrapper(
        model,
        utils.DINOHead(embed_dim, args.out_dim, False, num_kernel=args.num_kernel),
    )
    print('==> Preparing GMMs..')
    _, gmm_weights = load_pretrained_weights(model, args.pretrained_weights, 'teacher')
    means, covs_inv = get_gaussian(args.pretrained_weights)

    if use_cuda:
        model.cuda()
        means = means.cuda()
        covs_inv = covs_inv.cuda()

    model.eval()

    transform = transforms.Compose([
        transforms.Resize((280, 280)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    print('==> Preparing InD dataset..')

    in_data = Imagenet('val', args.ind_path, args.num_labels, transform)
    in_loader = torch.utils.data.DataLoader(
        in_data,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    results_in = get_scores(model, in_loader, means, covs_inv, gmm_weights, args.out_dim, args.num_kernel)
    aurocs, fpr95s = [], []

    # Texture dataset
    print("==> Preparing Texture..")
    ood_data = datasets.ImageFolder(os.path.join(args.ood_path, 'dtd', 'images'), transform=transform)
    ood_loader = torch.utils.data.DataLoader(ood_data, 
                        batch_size=args.batch_size_per_gpu, 
                        shuffle=True,
                        num_workers=args.num_workers,
                        pin_memory=True
                        )

    results_out = get_scores(model, ood_loader, means, covs_inv, gmm_weights, args.out_dim, args.num_kernel)
    auroc, fpr95 = get_results(results_in, results_out, 'dtd')
    aurocs.append(auroc)
    fpr95s.append(fpr95)
    print(f"AUROC:{auroc * 100:.2f}, FPR95:{fpr95 * 100:.2f}")
    print()

    # iNaturalist dataset
    print("==> Preparing iNaturalist..")
    ood_data = datasets.ImageFolder(os.path.join(args.ood_path, 'iNaturalist'), transform=transform)
    ood_loader = torch.utils.data.DataLoader(ood_data, 
                        batch_size=args.batch_size_per_gpu, 
                        shuffle=True,
                        num_workers=args.num_workers,
                        pin_memory=True
                        )
    results_out = get_scores(model, ood_loader, means, covs_inv, gmm_weights, args.out_dim, args.num_kernel)
    auroc, fpr95 = get_results(results_in, results_out, 'inat')
    aurocs.append(auroc)
    fpr95s.append(fpr95)
    print(f"AUROC:{auroc * 100:.2f}, FPR95:{fpr95 * 100:.2f}")
    print()

    # Places dataset
    print("==> Preparing Places..")
    ood_data = datasets.ImageFolder(os.path.join(args.ood_path, 'Places'), transform=transform)
    ood_loader = torch.utils.data.DataLoader(ood_data, 
                        batch_size=args.batch_size_per_gpu, 
                        shuffle=True,
                        num_workers=args.num_workers,
                        pin_memory=True
                        )

    results_out = get_scores(model, ood_loader, means, covs_inv, gmm_weights, args.out_dim, args.num_kernel)
    auroc, fpr95 = get_results(results_in, results_out, 'place')
    aurocs.append(auroc)
    fpr95s.append(fpr95)
    print(f"AUROC:{auroc * 100:.2f}, FPR95:{fpr95 * 100:.2f}")
    print()

    # SUN dataset
    print("==> Preparing SUN..")
    ood_data = datasets.ImageFolder(os.path.join(args.ood_path, 'SUN'), transform=transform)
    ood_loader = torch.utils.data.DataLoader(ood_data, 
                        batch_size=args.batch_size_per_gpu, 
                        shuffle=True,
                        num_workers=args.num_workers,
                        pin_memory=True
                        )

    results_out = get_scores(model, ood_loader, means, covs_inv, gmm_weights, args.out_dim, args.num_kernel)
    auroc, fpr95 = get_results(results_in, results_out, 'sun')
    aurocs.append(auroc)
    fpr95s.append(fpr95)
    print(f"AUROC:{auroc * 100:.2f}, FPR95:{fpr95 * 100:.2f}")
    print()
    print('AVERAGE')
    print(f"AUROC:{np.mean(aurocs) * 100:.2f}, FPR95:{np.mean(fpr95s) * 100:.2f}")
    print()

    # ImageNet-O dataset
    print('==> Preparing ImageNet-O')
    ood_data = datasets.ImageFolder(os.path.join(args.ood_path, 'imagenet-o'), transform=transform)
    ood_loader = torch.utils.data.DataLoader(ood_data, 
                        batch_size=args.batch_size_per_gpu, 
                        shuffle=True,
                        num_workers=args.num_workers,
                        pin_memory=True
                        )

    results_out = get_scores(model, ood_loader, means, covs_inv, gmm_weights, args.out_dim, args.num_kernel)
    auroc, fpr95 = get_results(results_in, results_out, 'im-o')
    print(f"AUROC:{auroc * 100:.2f}, FPR95:{fpr95 * 100:.2f}")
    print()

    # OpenImage-O dataset
    print("Preparing OpenImage-O..")
    ood_data = datasets.ImageFolder(os.path.join(args.ood_path, 'OpenImagesDataset/Images'), transform=transform)
    ood_loader = torch.utils.data.DataLoader(ood_data, 
                        batch_size=args.batch_size_per_gpu, 
                        shuffle=True,
                        num_workers=args.num_workers,
                        pin_memory=True
                        )
    results_out = get_scores(model, ood_loader, means, covs_inv, gmm_weights, args.out_dim, args.num_kernel)
    auroc, fpr95 = get_results(results_in, results_out, 'openimage')
    print(f"AUROC:{auroc * 100:.2f}, FPR95:{fpr95 * 100:.2f}")
    print()


def get_results(res_in, res_out, data_name):
    tar_in, tar_out = np.zeros(len(res_in)), np.ones(len(res_out))
    res, tar = [], []
    res.extend(res_in)
    res.extend(res_out)
    tar.extend(tar_in.tolist())
    tar.extend(tar_out.tolist())
    
    auroc = calc_auroc(res, tar)
    fpr95 = calc_fpr(res, tar, data_name)
    return auroc, fpr95
    

def get_scores(model, loader, means, covs_inv, gmm_weights, out_dim, num_kernel):
    num_iter = len(loader)
    bar = Bar('Getting results:', max=num_iter)
    results = []

    for idx, (inp, _) in enumerate(loader):
        # move to gpu
        if use_cuda:
            inp = inp.cuda(non_blocking=True)
        # forward
        with torch.no_grad():
            _, q = model(inp)
        maha = get_maha_score(means, covs_inv, gmm_weights, q.reshape(-1, num_kernel, out_dim // num_kernel))
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
        mu: centers of gmm of all classes with shape (classes, kernels, dimensions)
        cov_inv: The inverse matrix of cov which has shape (classes, kernels, dimensions, dimensions)
        gmm_weights: weights of gmm with shape (classes, kernels)
        x: features of input with shape (num_samples, kernels, dimensions)
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

    
def get_gaussian(pretrained_weights):
    gau_path = (os.path.join(*pretrained_weights.split('/')[:-1]))
    means = torch.load(os.path.join(gau_path, 'means.pt'), map_location='cpu')
    covs_inv = torch.load(os.path.join(gau_path, 'covs_inv.pt'), map_location='cpu')
    return means, covs_inv


def load_pretrained_weights(model, pretrained_weights, checkpoint_key):
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if 'center_loss' in state_dict.keys():
            centers = state_dict['center_loss']['centers']
            gmm_weights = state_dict['center_loss']['gmm_weights']
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
    return centers, gmm_weights
    

def calc_fpr(scores, trues, data_name):
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
    result = roc_auc_score(trues, scores)
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser('OOD detection on ImageNet')
    parser.add_argument('--arch', default='resnet50', type=str, help='Architecture')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--num_labels', default=1000, type=int, help='Number of labels for linear classifier')
    parser.add_argument('--ind_path', default='/path/to/imagenet/', type=str)
    parser.add_argument('--ood_path', default='/path/to/ood_datasets/', type=str)
    parser.add_argument('--out_dim', default=8192, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--num_kernel', default=32, type=int, help="Number of Gaussian components of GMM")
    args = parser.parse_args()
    ood_maha(args)