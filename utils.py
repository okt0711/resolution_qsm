import numpy as np
import torch
from torch.nn import init
from torch.optim import lr_scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


### For Pytorch version < 1.8
# def roll(x, shift, dim):
#     """
#     Similar to np.roll but applies to PyTorch Tensors
#     """
#     if isinstance(shift, (tuple, list)):
#         assert len(shift) == len(dim)
#         for s, d in zip(shift, dim):
#             x = roll(x, s, d)
#         return x
#     shift = shift % x.size(dim)
#     if shift == 0:
#         return x
#     left = x.narrow(dim, 0, x.size(dim) - shift)
#     right = x.narrow(dim, x.size(dim) - shift, shift)
#     return torch.cat((right, left), dim=dim)
#
#
# def fftshift(x, dim=None):
#     """
#     Similar to np.fft.fftshift but applies to PyTorch Tensors
#     """
#     if dim is None:
#         dim = tuple(range(x.dim()))
#         shift = [dim // 2 for dim in x.shape]
#     elif isinstance(dim, int):
#         shift = x.shape[dim] // 2
#     else:
#         shift = [x.shape[i] // 2 for i in dim]
#     return roll(x, shift, dim)
#
#
# def ifftshift(x, dim=None):
#     """
#     Similar to np.fft.ifftshift but applies to PyTorch Tensors
#     """
#     if dim is None:
#         dim = tuple(range(x.dim()))
#         shift = [(dim + 1) // 2 for dim in x.shape]
#     elif isinstance(dim, int):
#         shift = (x.shape[dim] + 1) // 2
#     else:
#         shift = [(x.shape[i] + 1) // 2 for i in dim]
#     return roll(x, shift, dim)
#
#
# def fft3(data):
#     assert data.size(-1) == 2
#     data = torch.fft(data, 3, normalized=False)
#     data = fftshift(data, dim=(-4, -3, -2))
#     return data
#
#
# def ifft3(data):
#     assert data.size(-1) == 2
#     data = ifftshift(data, dim=(-4, -3, -2))
#     data = torch.ifft(data, 3, normalized=False)
#     return data
#
#
# def to_complex(data, mode='img'):
#     data = data.unsqueeze(5)
#     if mode == 'img':
#         out = torch.cat([data, torch.zeros_like(data)], dim=5)
#     elif mode == 'dk':
#         out = torch.cat([data, data], dim=5)
#     else:
#         raise NotImplementedError('to_complex [%s] is not implemented', mode)
#     return out


def fft3(data):
    data = torch.fft.fftshift(torch.fft.fftn(data))
    return data


def ifft3(data):
    data = torch.fft.ifftn(torch.fft.ifftshift(data))
    return data


# Function for tensorboard visualization
def vol2slice(vol, slice_min=None, slice_max=None):
    hnY = int(vol.shape[2] / 2)
    hnX = int(vol.shape[3] / 2)
    hnZ = int(vol.shape[4] / 2)
    slice1 = vol[0, :, hnY, :, :]
    slice2 = vol[0, :, :, hnX, :]
    slice3 = vol[0, :, :, :, hnZ]
    slice = torch.cat([slice1, slice2, slice3], dim=2)
    if slice_min == None:
        slice_min = torch.min(slice)
    if slice_max == None:
        slice_max = torch.max(slice)

    slice = torch.clamp(slice, slice_min, slice_max)
    slice = (slice - slice_min) / (slice_max - slice_min) * 255
    slice = slice.type(torch.ByteTensor)
    return slice


class Mean:
    def __init__(self):
        self.numel = 0
        self.mean = 0
        self.val = 0

    def __call__(self, val):
        self.mean = self.mean * (self.numel / (self.numel + 1)) + val / (self.numel + 1)
        self.numel += 1
        self.val = val

    def step(self):
        return self.val

    def epoch(self):
        return self.mean

    def reset(self):
        self.numel = 0
        self.mean = 0
        self.val = 0


# Function for patch-wise inference
def get_patch_weight(size, stride):
    patch_weight_y = np.ones(size[0])
    overlap_size_y = size[0] - stride
    patch_weight_y[:overlap_size_y] = patch_weight_y[:overlap_size_y] * (np.cos(np.linspace(-np.pi, 0, overlap_size_y)) + 1) / 2
    patch_weight_y[-overlap_size_y:] = patch_weight_y[-overlap_size_y:] * (np.cos(np.linspace(0, np.pi, overlap_size_y)) + 1) / 2
    patch_weight_y = np.tile(patch_weight_y[:, np.newaxis, np.newaxis], [1, size[0], size[0]])

    patch_weight_x = np.ones(size[1])
    overlap_size_x = size[1] - stride
    patch_weight_x[:overlap_size_x] = patch_weight_x[:overlap_size_x] * (np.cos(np.linspace(-np.pi, 0, overlap_size_x)) + 1) / 2
    patch_weight_x[-overlap_size_x:] = patch_weight_x[-overlap_size_x:] * (np.cos(np.linspace(0, np.pi, overlap_size_x)) + 1) / 2
    patch_weight_x = np.tile(patch_weight_x[np.newaxis, :, np.newaxis], [size[1], 1, size[1]])

    patch_weight_z = np.ones(size[2])
    overlap_size_z = size[2] - stride
    patch_weight_z[:overlap_size_z] = patch_weight_z[:overlap_size_z] * (np.cos(np.linspace(-np.pi, 0, overlap_size_z)) + 1) / 2
    patch_weight_z[-overlap_size_z:] = patch_weight_z[-overlap_size_z:] * (np.cos(np.linspace(0, np.pi, overlap_size_z)) + 1) / 2
    patch_weight_z = np.tile(patch_weight_z[np.newaxis, np.newaxis, :], [size[2], size[2], 1])

    patch_weight = patch_weight_y * patch_weight_x * patch_weight_z
    return patch_weight
