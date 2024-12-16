import torch
import torch.nn as nn
from layers import ConvAdaInBlock, DownAdaInBlock, UpAdaInBlock, CommonCode, IndividualCode
from utils import fft3, ifft3#, to_complex


# 3D Unet architecture with AdaIN layers
class Unet_AdaIN(nn.Module):
    def __init__(self, in_channels, out_channels, opt):
        super(Unet_AdaIN, self).__init__()
        self.commoncode = CommonCode(3, 128)
        self.d1code = IndividualCode(128, opt.ngf)
        self.d2code = IndividualCode(128, opt.ngf * 2)
        self.d3code = IndividualCode(128, opt.ngf * 4)
        self.brcode = IndividualCode(128, opt.ngf * 8)
        self.u3code = IndividualCode(128, opt.ngf * 4)
        self.u2code = IndividualCode(128, opt.ngf * 2)
        self.u1code = IndividualCode(128, opt.ngf)

        self.down1 = ConvAdaInBlock(in_channels, opt.ngf, opt.act_type)
        self.down2 = DownAdaInBlock(opt.ngf, opt.ngf * 2, opt.act_type)
        self.down3 = DownAdaInBlock(opt.ngf * 2, opt.ngf * 4, opt.act_type)
        self.bridge = DownAdaInBlock(opt.ngf * 4, opt.ngf * 8, opt.act_type)
        self.up3 = UpAdaInBlock(opt.ngf * 8, opt.ngf * 4, opt.act_type)
        self.up2 = UpAdaInBlock(opt.ngf * 4, opt.ngf * 2, opt.act_type)
        self.up1 = UpAdaInBlock(opt.ngf * 2, opt.ngf, opt.act_type)
        self.last_conv = nn.Conv3d(opt.ngf, out_channels, kernel_size=1, stride=1)

    def forward(self, x, voxel, alpha=1.0):
        commoncode = self.commoncode(voxel)
        m_d1, s_d1 = self.d1code(commoncode)
        m_d2, s_d2 = self.d2code(commoncode)
        m_d3, s_d3 = self.d3code(commoncode)
        m_br, s_br = self.brcode(commoncode)
        m_u3, s_u3 = self.u3code(commoncode)
        m_u2, s_u2 = self.u2code(commoncode)
        m_u1, s_u1 = self.u1code(commoncode)

        d1 = self.down1(x, m_d1, s_d1, alpha)
        d2 = self.down2(d1, m_d2, s_d2, alpha)
        d3 = self.down3(d2, m_d3, s_d3, alpha)
        br = self.bridge(d3, m_br, s_br, alpha)
        u3 = self.up3(br, d3, m_u3, s_u3, alpha)
        u2 = self.up2(u3, d2, m_u2, s_u2, alpha)
        u1 = self.up1(u2, d1, m_u1, s_u1, alpha)
        out = self.last_conv(u1)
        return out


### For Pytorch version < 1.8
# class forward_model(nn.Module):
#     def __init__(self):
#         super(forward_model, self).__init__()
#
#     def forward(self, x, dipole_kernel):
#         qsm_k = fft3(to_complex(x, mode='img'))
#         phase_k = qsm_k * to_complex(dipole_kernel, mode='dk')
#         out = ifft3(phase_k)
#         out = out[:, :, :, :, :, 0]
#         return out


# Forward model (b = F^(-1)dFX)
class forward_model(nn.Module):
    def __init__(self):
        super(forward_model, self).__init__()

    def forward(self, x, dipole_kernel):
        qsm_k = fft3(x)
        phase_k = qsm_k * dipole_kernel
        out = torch.real(ifft3(phase_k))
        return out
