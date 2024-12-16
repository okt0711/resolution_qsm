import torch
import torch.nn as nn
import numpy as np


class GRADLoss(nn.Module):
    def __init__(self):
        super(GRADLoss, self).__init__()
        self.loss = nn.L1Loss()

    def __call__(self, x, y):
        x_cen = x[:, :, 1:-1, 1:-1, 1:-1]
        x_shape = x.shape
        grad_x = torch.zeros_like(x_cen)
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    x_slice = x[:, :, i + 1:i + x_shape[2] - 1, j + 1:j + x_shape[3] - 1, k + 1:k + x_shape[4] - 1]
                    if i * i + j * j + k * k == 0:
                        temp = torch.zeros_like(x_cen)
                    else:
                        temp = (1.0 / np.sqrt(i * i + j * j + k * k)) * (x_slice - x_cen)
                    grad_x = grad_x + temp

        y_cen = y[:, :, 1:-1, 1:-1, 1:-1]
        y_shape = y.shape
        grad_y = torch.zeros_like(y_cen)
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    y_slice = y[:, :, i + 1:i + y_shape[2] - 1, j + 1:j + y_shape[3] - 1, k + 1:k + y_shape[4] - 1]
                    if i * i + j * j + k * k == 0:
                        temp = torch.zeros_like(y_cen)
                    else:
                        temp = (1.0 / np.sqrt(i * i + j * j + k * k)) * (y_slice - y_cen)
                    grad_y = grad_y + temp

        loss = self.loss(grad_x, grad_y)
        return loss


def TVLoss(x):
    x_cen = x[:, :, 1:-1, 1:-1, 1:-1]
    x_shape = x.shape
    grad_x = torch.zeros_like(x_cen)
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                x_slice = x[:, :, i + 1:i + x_shape[2] - 1, j + 1:j + x_shape[3] - 1, k + 1:k + x_shape[4] - 1]
                if i * i + j * j + k * k == 0:
                    temp = torch.zeros_like(x_cen)
                else:
                    temp = (1.0 / np.sqrt(i * i + j * j + k * k)) * (x_slice - x_cen)
                grad_x = grad_x + temp

    return torch.mean(torch.abs(grad_x))
