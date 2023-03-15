import os
import argparse
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import models as torchvision_models
from torchvision import transforms
import utils

parser = argparse.ArgumentParser('GMM')
parser.add_argument('--arch', default='resnet50', type=str, help='Architecture')
parser.add_argument('--output_dir', default='', type=str, help="Path to pretrained weights to evaluate.")
parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
parser.add_argument('--num_labels', default=1000, type=int, help='Number of labels for linear classifier')
parser.add_argument('--data_path', default='/path/to/imagenet/', type=str)
parser.add_argument('--out_dim', default=8192, type=int, help="""Dimensionality of
    the DINO head output. For complex and large datasets large values (like 65k) work well.""")
parser.add_argument('--num_kernel', default=32, type=int, help="Number of Gaussian components of GMM")
args = parser.parse_args()


class Imagenet(Dataset):
    def __init__(self, mode, data_path, cls_idx, transform=None) -> None:
        super().__init__()
        assert mode in ['train', 'val']

        self.mode = mode
        self.transform = transform
        self.imagenet_path = os.path.join(data_path, mode)
        self.classes, self.img_list = {}, []

        with open(f'dataset/ind_imagenet_{args.num_labels}cls.txt', 'r') as f:
            for idx, line in enumerate(f):
                if idx != cls_idx:
                    continue
                if idx > cls_idx:
                    break
                cls_name = line.strip()
                self.classes[cls_name] = idx

                cls_img_list = os.listdir(os.path.join(self.imagenet_path, cls_name))
                cls_img_list = [os.path.join(cls_name, k) for k in cls_img_list]
                self.img_list = self.img_list + cls_img_list

    def __getitem__(self, idx):

        img_name = self.img_list[idx]

        cls_name = img_name.split('/')[0]
        cls_label = self.classes[cls_name]

        img = Image.open(os.path.join(self.imagenet_path, img_name)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        else:
            img = self.train_transforms(img) if self.mode == 'train' else \
                self.val_transforms(img)

        return img, cls_label

    def __len__(self):
        return len(self.img_list)

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


print('==> Preparing model..')
model = torchvision_models.__dict__[args.arch]()
embed_dim = model.fc.weight.shape[1]
model = utils.MultiCropWrapper(
        model,
        utils.DINOHead(embed_dim, args.out_dim, False, num_kernel=args.num_kernel),
    )

pretrained_weights = os.path.join(args.output_dir, 'checkpoint.pth')
_, _ = load_pretrained_weights(model, pretrained_weights, 'teacher')

use_cuda = torch.cuda.is_available()
if use_cuda:
    model.cuda()

model.eval()

'''
Class-by-class calculation of the Mean and Cov.
'''
transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

for i in range(args.num_labels):
    print(f'==> Preparing dataset {i}..')
    dataset = Imagenet('train', args.data_path, i, transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
    )
    for idx, (img, _) in enumerate(dataloader):
        img = img.cuda()

        with torch.no_grad():
            _, q = model(img)
            if len(q.shape) > 2:
                q = F.adaptive_avg_pool2d(q, 1)
            q = q.reshape(q.shape[0], args.num_kernel, -1)
            dim = q.shape[-1]
        if idx == 0:
            x = q
        else:
            x = torch.cat([x, q], dim=0)

    for gau in range(args.num_kernel):
        samples = x[:, gau:gau + 1]
        mean = samples.mean(0)

        # Whitening
        samples = (samples - mean).reshape(-1, dim, 1).double()

        # Dot product
        cov = torch.bmm(samples, samples.permute(0, 2, 1))

        # Unbiased estimation
        cov = cov.reshape(-1, 1, dim, dim).sum(0)
        cov = (cov / (x.shape[0] - 1))

        if gau == 0:
            cls_mean = mean
            cls_cov = cov
        else:
            cls_mean = torch.cat([cls_mean, mean], dim=0)
            cls_cov = torch.cat([cls_cov, cov], dim=0)
    
    if i == 0:
        ind_means = cls_mean.unsqueeze(0)
        ind_covs_inv = torch.linalg.inv(cls_cov.unsqueeze(0))
    else:
        ind_means = torch.cat([ind_means, cls_mean.unsqueeze(0)], dim=0)
        ind_covs_inv = torch.cat([ind_covs_inv, torch.linalg.inv(cls_cov.unsqueeze(0))], dim=0)
        
    print(f'Mean and cov_inverse of dataset {i} calculation complete.')

torch.save(ind_means, os.path.join(args.output_dir, 'means.pt'))
torch.save(ind_covs_inv, os.path.join(args.output_dir, 'covs_inv.pt'))