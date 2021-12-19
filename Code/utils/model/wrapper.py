import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.model.edcnn import EDCNN
from utils.common.layer_utils import *

fft  = lambda x, ax : np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x, axes=ax), axes=ax, norm='ortho'), axes=ax) 
ifft = lambda X, ax : np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(X, axes=ax), axes=ax, norm='ortho'), axes=ax) 
torch_fft  = lambda x : torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x), norm='ortho')) 
torch_ifft = lambda X : torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(X) , norm='ortho'))

class KI(nn.Module):
    def __init__(self, model_abs, model_config):
        super(KI, self).__init__()
        
        self.k_model1 = model_abs(**model_config['k_space'])
        self.conv1 = nn.Conv2d(model_config['k_space']['output'], 2, kernel_size=1)
        self.conv2 = nn.Conv2d(2, 1, kernel_size=1)
        self.i_model1 = model_abs(**model_config['image'])
        self.relu = nn.LeakyReLU()

    def forward(self, img, kspace=None):
        if not kspace==None:
            out1 = self.k_model1(kspace)
            out1 = self.relu(self.conv1(out1))
            ## torch ver.
            # r_out1 = out1[:,0,...]
            # i_out1 = out1[:,1,...]
            # out = r_out1 + i_out1*1.0j
            # out = abs(torch_ifft(out))
            # out = out.unsqueeze(1)
            # out = torch.cat([out, img], dim=-3)
            # out = self.relu(self.conv2(out))
            ## numpy ver.
            out1 = out1.cpu().detach().numpy()
            r_out1 = out1[:,0,...]
            i_out1 = out1[:,1,...]
            out = r_out1 + i_out1*1.0j
            out = abs(ifft(out, range(out.ndim)))
            out = out[:,np.newaxis, ...]
            out = torch.from_numpy(out).cuda()
            out = torch.cat([out, img], dim=-3)
            out = out.type(torch.cuda.HalfTensor)
            out = self.relu(self.conv2(out))
        else:
            out = img

        out2 = self.i_model1(out)

        return out2

class IK(nn.Module):
    def __init__(self, model_abs, model_config):
        super(IK, self).__init__()
        
        self.i_model1 = model_abs(**model_config['image'])
        self.k_model1 = model_abs(**model_config['k_space'])
        # self.conv1 = nn.Conv2d(model_config['k_space']['output'], 2, kernel_size=1)
        # self.conv2 = nn.Conv2d(2, 1, kernel_size=1)
        self.relu = nn.LeakyReLU()

    def forward(self, img, kspace=None):
        out1 = self.i_model1(img)
        if kspace==None:
            return out1
        else:
            out1 = torch_fft(out1)
            r_out1 = out1.real
            i_out1 = out1.imag
            out = torch.cat([r_out1, i_out1, kspace], dim=-3)
            out1 = self.k_model1(out) # [l, 2, H, W]
            ## torch ver.
            r_out1 = out1[:,0,...]
            i_out1 = out1[:,1,...]
            out = r_out1 + i_out1*1.0j
            out = abs(torch_ifft(out))
            return out

        