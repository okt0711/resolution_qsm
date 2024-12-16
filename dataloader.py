import numpy as np
import torch
import random
from os import listdir
from os.path import join
from scipy import io as sio
from torch.utils.data import Dataset

dtype = np.float32


class QSMDataset(Dataset):
    def __init__(self, opt, mode):
        # Data root
        self.data_root = join(opt.data_root, 'phase', mode)

        # List of root of files
        self.flist = []
        for aFolder in sorted(listdir(self.data_root)):
            folder_root = join(self.data_root, aFolder)
            for aImg in sorted(listdir(folder_root)):
                self.flist.append(join(folder_root, aImg))

        # Size of patch
        self.nY = opt.nY
        self.nX = opt.nX
        self.nZ = opt.nZ

        # The number of patch for one epoch
        self.num_patch = opt.num_patch

        # Train or test
        self.mode = mode

    def __len__(self):
        if self.mode == 'train':
            return self.num_patch
        else:
            return len(self.flist)

    def __getitem__(self, idx):
        if self.mode == 'train':
            idx_P = random.randrange(len(self.flist))
        else:
            idx_P = idx

        # Read phase, brain mask, voxel size, and B0 direction
        phase, mask, voxel, B0 = self.read_mat(self.flist[idx_P], 'phase')
        phase = phase * mask

        # Generate dipole kernel with the size of (nY, nX, nZ)
        DK = self.make_dipole_kernel([self.nX, self.nY, self.nX], voxel, B0)

        # X, Y, Z -> Y, X, Z
        voxel = np.squeeze(voxel).astype(dtype)
        voxel = voxel[[1, 0, 2]]

        if self.mode == 'train':
            # Generate the patch by random croping
            [phase, mask] = self.random_crop([phase, mask])

            # Random scaling to cover various range of values (refer to 4.3. Implementation details)
            phase = self.scale_data(phase)

            # Augmentation with random flipping/rotation
            [phase, DK, mask], voxel = self.augment_data([phase, DK, mask], voxel)

        phase = torch.from_numpy(phase.copy()).unsqueeze(0)
        DK = torch.from_numpy(DK.copy()).unsqueeze(0)
        mask = torch.from_numpy(mask.copy()).unsqueeze(0)
        voxel = torch.from_numpy(voxel)

        return phase, DK, mask, voxel

    @staticmethod
    def make_dipole_kernel(matrix_size, voxel_size, B0_dir):
        Y, X, Z = np.meshgrid(np.linspace(-matrix_size[1] / 2, matrix_size[1] / 2 - 1, matrix_size[1]),
                              np.linspace(-matrix_size[0] / 2, matrix_size[0] / 2 - 1, matrix_size[0]),
                              np.linspace(-matrix_size[2] / 2, matrix_size[2] / 2 - 1, matrix_size[2]))
        X = X / (matrix_size[0] * voxel_size[0])
        Y = Y / (matrix_size[1] * voxel_size[1])
        Z = Z / (matrix_size[2] * voxel_size[2])

        np.seterr(divide='ignore', invalid='ignore')
        D = 1 / 3 - np.divide(np.square(X * B0_dir[0] + Y * B0_dir[1] + Z * B0_dir[2]), np.square(X) + np.square(Y) + np.square(Z))
        D = np.where(np.isnan(D), 0, D)

        return D.astype(dtype)

    @staticmethod
    def scale_data(img):
        p_scale = random.uniform(0.7, 4)
        img = img * p_scale

        p_invert = random.random()
        if p_invert > 0.5:
            img = -img
        return img

    @staticmethod
    def augment_data(imgs, voxel_size):
        p_flip = random.random()
        p_rot = random.random()
        if p_flip < 0.25:
            rotated_imgs = [np.flip(img, 0) for img in imgs]
        elif p_flip < 0.5:
            rotated_imgs = [np.flip(img, 1) for img in imgs]
        elif p_flip < 0.75:
            rotated_imgs = [np.flip(img, 2) for img in imgs]
        else:
            rotated_imgs = imgs

        if p_rot < 0.5:
            augmented_imgs = [np.rot90(img, axes=(0, 1)) for img in rotated_imgs]
            voxel_size = voxel_size[[1, 0, 2]]
        else:
            augmented_imgs = rotated_imgs

        return augmented_imgs, voxel_size

    def read_mat(self, filename, keyword):
        mat = sio.loadmat(filename, verify_compressed_data_integrity=False)
        data = mat[keyword]
        mask = mat['mask']
        voxel_size = np.squeeze(mat['voxel_size'])
        B0_dir = np.squeeze(mat['B0_dir'])

        size = np.shape(data)
        padY = max(0, self.nY - size[-3])
        padY1 = int(padY / 2)
        padX = max(0, self.nX - size[-2])
        padX1 = int(padX / 2)
        padZ = max(0, self.nZ - size[-1])
        padZ1 = int(padZ / 2)

        data = np.pad(data, ((padY1, padY - padY1), (padX1, padX - padX1), (padZ1, padZ - padZ1)))
        mask = np.pad(mask, ((padY1, padY - padY1), (padX1, padX - padX1), (padZ1, padZ - padZ1)))

        return data, mask, voxel_size, B0_dir

    def random_crop(self, imgs):
        y, x, z = np.shape(imgs[0])

        ys = random.randrange(y - self.nY + 1)
        xs = random.randrange(x - self.nX + 1)
        zs = random.randrange(z - self.nZ + 1)

        patches = [img[ys:ys + self.nY, xs:xs + self.nX, zs:zs + self.nZ] for img in imgs]
        return patches

