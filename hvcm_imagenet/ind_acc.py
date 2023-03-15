import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from progress.bar import Bar
from torch import nn
from torchvision import models as torchvision_models
from torchvision import transforms

import utils
from dataset.imagenet import Imagenet


def calculate_ind_acc(args):
    utils.init_distributed_mode(args)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    print('==> Preparing val dataset..')
    dataset_val = Imagenet('val', args.data_path, args.num_labels, transform)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset_val)
    dataloader = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print('==> Preparing model..')
    if args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch]()
        embed_dim = model.fc.weight.shape[1]
        model.fc = nn.Identity()
    else:
        print(f"Unknow architecture: {args.arch}")
        sys.exit(1)

    model = utils.MultiCropWrapper(
        model,
        utils.DINOHead(embed_dim, args.out_dim, False, num_kernel=args.num_kernel),
    )
    model.cuda()
    model.eval()

    # load weights to evaluate
    print("==> Preparing group centers..")
    centers, _ = load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch)
    clses, kers, dims = centers.shape
    acc = utils.Accuracy()

    num_batch = len(dataloader)
    bar = Bar('Calculating ind_acc:', max=num_batch)

    for idx, (x, cls) in enumerate(dataloader):
        x, cls = x.cuda(), cls.cuda()
        with torch.no_grad():
            _, q = model(x)
        bs = q.shape[0]

        centers_ = centers.unsqueeze(0).expand(bs, clses, kers, dims)
        q = q.unsqueeze(1).expand(bs, clses, kers * dims).reshape(-1, dims)

        centers_ = centers_.reshape(-1, clses, kers * dims).cuda()
        q = q.reshape(-1, clses, kers * dims)
        results = torch.cosine_similarity(q, centers_, dim=-1)
        results = torch.argmax(results, 1)

        acc.update(results, cls)
        bar.suffix = f'{idx+1}/{num_batch}, acc:{(acc.get_top1()*100):.3f}'
        bar.next()
    bar.finish()


def load_pretrained_weights(model, pretrained_weights, checkpoint_key, model_name):
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
        # remove `backbone.` prefix induced by multicrop wrapper
        # state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
        return centers, gmm_weights


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with cosine similarity on ImageNet')
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument('--data_path', default='/path/to/imagenet/', type=str)
    parser.add_argument('--num_workers', default=40, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--num_labels', default=1000, type=int, help='Number of labels for linear classifier')

    parser.add_argument('--out_dim', default=8192, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--num_kernel', default=32, type=int, help="Number of Gaussian components of GMM")
    args = parser.parse_args()
    calculate_ind_acc(args)