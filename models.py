import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from os import makedirs
from os.path import join, isdir, isfile
from scipy import io as sio
from math import ceil, floor
from networks import Unet_AdaIN, forward_model
from losses import GRADLoss, TVLoss
from utils import get_scheduler, init_net, vol2slice, Mean, get_patch_weight
from tqdm import tqdm


class ResolutionQSM:
    def __init__(self, opt):
        self.device = opt.device
        self.disp_step = opt.disp_step
        self.nY = opt.nY
        self.nX = opt.nX
        self.nZ = opt.nZ
        self.lambda_cycle = opt.lambda_cycle
        self.lambda_grad = opt.lambda_grad
        self.lambda_tv = opt.lambda_tv
        self.beta1 = opt.beta1
        self.beta2 = opt.beta2
        self.n_epochs = opt.n_epochs
        self.n_epochs_decay = opt.n_epochs_decay
        self.continue_epoch = opt.continue_epoch
        self.test_epoch = opt.test_epoch
        self.lr = opt.lr
        self.init_type = opt.init_type
        self.init_gain = opt.init_gain
        self.save_epoch = opt.save_epoch
        self.save_path = opt.save_path
        self.experiment_name = opt.experiment_name
        self.stride = opt.stride
        self.opt = opt
        self.ckpt_dir = opt.ckpt_dir
        self.log_dir = opt.log_dir
        self.writer = SummaryWriter(self.log_dir)

        self.cycle_loss = nn.L1Loss()
        self.grad_loss = GRADLoss()
        self.loss_name = ['total_loss',
                          'cycle_loss',
                          'grad_loss',
                          'tv_loss']

        self.net = Unet_AdaIN(1, 1, self.opt).to(self.device)
        self.forward_model = forward_model()

        self.optim = torch.optim.Adam(self.net.parameters(), self.lr, betas=(self.beta1, self.beta2))
        self.scheduler = get_scheduler(self.optim, self.opt)

    def train(self, dataloader):
        if isfile(join(self.ckpt_dir, str(self.continue_epoch) + '.pth')):
            checkpoint = torch.load(join(self.ckpt_dir, str(self.continue_epoch) + '.pth'))
            self.net.load_state_dict(checkpoint['state_dict'])
            self.optim.load_state_dict(checkpoint['optim_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            trained_epoch = checkpoint['epoch']
            print('Start from saved model - ' + str(trained_epoch))
        else:
            init_net(self.net, self.init_type, self.init_gain)
            trained_epoch = 0
            print('Start initially')

        losses = {name: Mean() for name in self.loss_name}
        dataset_len = len(dataloader)

        for epoch in tqdm(range(trained_epoch, self.n_epochs + self.n_epochs_decay), desc='Epoch', total=self.n_epochs + self.n_epochs_decay, initial=trained_epoch):
            for name in self.loss_name:
                losses[name].reset()
            disp_cnt = 0
            for step, (phase, DK, mask, voxel) in enumerate(tqdm(dataloader)):
                phase = phase.to(self.device)
                DK = DK.to(self.device)
                mask = mask.to(self.device)
                voxel = voxel.to(self.device)

                QSM = self.net(phase, voxel) * mask
                phase_recon = self.forward_model(QSM, DK) * mask

                cycle_loss = self.cycle_loss(phase_recon, phase)
                grad_loss = self.grad_loss(phase_recon, phase)
                tv_loss = TVLoss(QSM)
                total_loss = self.lambda_cycle * cycle_loss + self.lambda_grad * grad_loss + self.lambda_tv * tv_loss

                self.optim.zero_grad()
                total_loss.backward()
                self.optim.step()

                losses['cycle_loss'](cycle_loss.detach())
                losses['grad_loss'](grad_loss.detach())
                losses['tv_loss'](tv_loss.detach())
                losses['total_loss'](total_loss.detach())

                if step % self.disp_step == 0:
                    for name in self.loss_name:
                        self.writer.add_scalar('Step/' + name, losses[name].step(), epoch * ceil(dataset_len / self.disp_step) + disp_cnt)
                    self.writer.add_image('Image/1_phase', vol2slice(phase, -0.1, 0.1), epoch * ceil(dataset_len / self.disp_step) + disp_cnt)
                    self.writer.add_image('Image/2_QSM', vol2slice(QSM, -0.1, 0.1), epoch * ceil(dataset_len / self.disp_step) + disp_cnt)
                    self.writer.add_image('Image/3_phase_recon', vol2slice(phase_recon, -0.1, 0.1), epoch * ceil(dataset_len / self.disp_step) + disp_cnt)
                    disp_cnt += 1

            for name in self.loss_name:
                self.writer.add_scalar('Epoch/' + name, losses[name].epoch(), epoch + 1)
            self.writer.add_scalar('Epoch/lr', self.scheduler.get_last_lr()[0], epoch + 1)

            self.scheduler.step()
            if (epoch + 1) % self.save_epoch == 0:
                torch.save({'epoch': epoch + 1, 'state_dict': self.net.state_dict(),
                            'optim_state_dict': self.optim.state_dict(),
                            'scheduler_state_dict': self.scheduler.state_dict()},
                           join(self.ckpt_dir, '{}'.format(epoch + 1) + '.pth'))

    def test(self, dataloader):
        with torch.no_grad():
            checkpoint = torch.load(join(self.ckpt_dir, str(self.test_epoch) + '.pth'))
            self.net.load_state_dict(checkpoint['G_P2Q_state_dict'])
            self.net.eval()

            save_path = join(self.save_path, self.experiment_name, 'test_{}_{}'.format(self.test_epoch, self.stride))
            if not isdir(save_path):
                makedirs(save_path)

            print('Start test.')
            for step, [phase, _, mask, voxel] in enumerate(tqdm(dataloader)):
                overlap_y = self.nY - self.stride
                overlap_x = self.nX - self.stride
                overlap_z = self.nZ - self.stride

                pad_y = (ceil((phase.size(2) + overlap_y - self.nY) / self.stride) * self.stride + self.nY) - (phase.size(2) + overlap_y)
                pad_x = (ceil((phase.size(3) + overlap_x - self.nX) / self.stride) * self.stride + self.nX) - (phase.size(3) + overlap_x)
                pad_z = (ceil((phase.size(4) + overlap_z - self.nZ) / self.stride) * self.stride + self.nZ) - (phase.size(4) + overlap_z)

                pad_y1 = overlap_y + floor(pad_y / 2)
                pad_y2 = overlap_y + (pad_y - floor(pad_y / 2))
                pad_x1 = overlap_x + floor(pad_x / 2)
                pad_x2 = overlap_x + (pad_x - floor(pad_x / 2))
                pad_z1 = overlap_z + floor(pad_z / 2)
                pad_z2 = overlap_z + (pad_z - floor(pad_z / 2))

                phase_pad = F.pad(phase, [pad_z1, pad_z2, pad_x1, pad_x2, pad_y1, pad_y2], mode='constant')
                mask_pad = F.pad(mask, [pad_z1, pad_z2, pad_x1, pad_x2, pad_y1, pad_y2], mode='constant')
                voxel = voxel.to(self.device)
                tmp_output = np.zeros((phase_pad.size(2), phase_pad.size(3), phase_pad.size(4)), dtype=np.float32)

                # Patch-wise inference (reduce patch artifact)
                patch_weight = get_patch_weight((self.nY, self.nX, self.nZ), self.stride)
                for i in range(ceil((phase_pad.size(2) - self.nY) / self.stride) + 1):
                    for j in range(ceil((phase_pad.size(3) - self.nX) / self.stride) + 1):
                        for k in range(ceil((phase_pad.size(4) - self.nZ) / self.stride) + 1):
                            ys = self.stride * i
                            xs = self.stride * j
                            zs = self.stride * k

                            ys = min(ys, phase_pad.size(2) - self.nY)
                            xs = min(xs, phase_pad.size(3) - self.nX)
                            zs = min(zs, phase_pad.size(4) - self.nZ)

                            patch = phase_pad[:, :, ys:ys + self.nY, xs:xs + self.nX, zs:zs + self.nZ].to(self.device)
                            patch_mask = mask_pad[:, :, ys:ys + self.nY, xs:xs + self.nX, zs:zs + self.nZ].to(self.device)

                            patch_output = np.squeeze((self.net(patch, voxel) * patch_mask).to('cpu:0').detach().numpy())
                            tmp_output[ys:ys + self.nY, xs:xs + self.nX, zs:zs + self.nZ] += patch_output * patch_weight

                QSM = tmp_output[pad_y1:-pad_y2, pad_x1:-pad_x2, pad_z1:-pad_z2]

                subpath = dataloader.flist_P[step].split('test/')[1]
                subname = subpath.split('/')[0]
                fname = subpath.split('/')[1]

                test_output = {'QSM': QSM}

                sub_save_path = join(save_path, subname)
                if not isdir(sub_save_path):
                    makedirs(sub_save_path)

                sio.savemat(join(sub_save_path, fname), test_output)
