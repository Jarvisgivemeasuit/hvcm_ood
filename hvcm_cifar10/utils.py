import os, logging
import sys
import time
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.datasets as datasets
from torchvision import transforms


def set_logging_defaults(logdir, args):
    if os.path.isdir(logdir):
        res = input('"{}" exists. Overwrite [Y/n]? '.format(logdir))
        if res != 'Y':
            raise Exception('"{}" exists.'.format(logdir))
    else:
        os.makedirs(logdir)

    # set basic configuration for logging
    logging.basicConfig(format="[%(asctime)s] [%(name)s] %(message)s",
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(logdir, 'log.txt')),
                                  logging.StreamHandler(os.sys.stdout)])

    # log cmdline argumetns
    logger = logging.getLogger('main')
    logger.info(' '.join(os.sys.argv))
    logger.info(args)

_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 86.
last_time = time.time()
begin_time = last_time

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


# --- HVCM ---
class HVCMLoss(nn.Module):
    """HVCM loss.
    Reference:
        Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=64, feat_dim=[32, 256], use_gpu=True, decom=True):
        super(HVCMLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim

        assert len(feat_dim) == 2
        self.centers_shape = feat_dim.copy()
        self.centers_shape.insert(0, num_classes)

        if not decom:
            self.centers_shape = [num_classes, feat_dim[0] * feat_dim[1]]

        if use_gpu:
            self.centers = nn.Parameter(torch.randn(self.centers_shape).cuda())
            self.gmm_weights = nn.Parameter(torch.softmax(torch.randn(num_classes, feat_dim[0]), dim=-1).cuda(), requires_grad=False)
        else:
            self.centers = nn.Parameter(torch.randn(self.centers_shape))
            self.gmm_weights = nn.Parameter(torch.softmax(torch.randn(num_classes, feat_dim[0]), dim=-1), requires_grad=False)

    def forward(self, gmm_weights, x, labels, epoch, args):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
        """
        x = x.reshape(-1, self.feat_dim[0], self.feat_dim[1])
        gmm_weights = gmm_weights.reshape(-1, self.feat_dim[0])

        assert x.size(0) == labels.size(0), "features.size(0) is not equal to labels.size(0)"
        total_loss = 0

        center = self.centers[labels]

        '''loss for center'''
        kl_x = F.kl_div(F.log_softmax(x,dim=-1), F.softmax(center.clone().detach(), dim=-1), reduction='none').sum(-1) * args.alpha
        kl_ce = F.kl_div(F.log_softmax(center, dim=-1), F.softmax(x.clone().detach(),dim=-1), reduction='none').sum(-1) * args.beta

        '''loss for gmm weights'''
        loss = (kl_x.clone().detach() * F.softmax(gmm_weights, dim=-1)).mean()
        total_loss += loss

        js = (kl_ce + kl_x)

        loss = torch.clamp(js, min=1e-5, max=1e+5).mean()
        total_loss += loss

        for cls in labels.unique():
            self.gmm_weights[cls] = F.softmax(gmm_weights[labels==cls], dim=-1).detach().mean(0) * args.gamma + \
                                    self.gmm_weights[cls] * (1 - args.gamma)
        return total_loss


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor



def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class HVCMhead(nn.Module):
    def __init__(self, in_dim, out_dim, 
                norm_last_layer=True, 
                bottleneck_dim=256, num_kernel=0):
        super().__init__()
        self.mlp = nn.Linear(in_dim, bottleneck_dim)
        self.apply(self._init_weights)

        # --- Attributes ---
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        # --- Weights for Gaussian compoents of GMM ---
        self.weight_layer = nn.Linear(bottleneck_dim, num_kernel, bias=False)

        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False
        

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)

        attri = self.last_layer(x)
        weight = self.weight_layer(x)
        return weight, attri


class NetworkWrapper(nn.Module):
    def __init__(self, backbone, head) -> None:
        super().__init__()
        self.backbone = backbone
        try:
            self.fc = backbone.linear
        except:
            self.fc = backbone.fc

        self.backbone.fc, self.backbone.linear, self.backbone.head = nn.Identity(), nn.Identity(), nn.Identity()
        self.head = head

    def forward(self, x):
        x = self.backbone(x)

        prob = self.fc(x)
        output = self.head(x)
        return prob, output


class Accuracy:
    def __init__(self, eps=1e-7):
        self.num_correct = 0
        self.num_instance = 0

        self.eps = eps

    def update(self, pred, target):

        self.num_correct += (pred == target).sum().item()
        self.num_instance += target.shape[0]

    def get_top1(self):
        return self.num_correct / (self.num_instance + self.eps)

    def reset(self):
        self.num_correct = 0
        self.num_instance = 0


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule