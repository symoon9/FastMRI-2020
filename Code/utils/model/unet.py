import torch
from torch import nn
from torch.nn import functional as F


class Unet(nn.Module):

    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans

        self.first_block = ConvBlock(in_chans, 2)
        self.down1 = Down(2,3,4)
        self.up1 = Up(4, 2)
        self.last_block = nn.Conv2d(2, out_chans, kernel_size=1)

    def norm(self, x):
        b, h, w = x.shape
        x = x.view(b, h * w)
        mean = x.mean(dim=1).view(b, 1, 1)
        std = x.std(dim=1).view(b, 1, 1)
        x = x.view(b, h, w)
        return (x - mean) / std, mean, std

    def unnorm(self, x, mean, std):
        return x * std + mean

    def forward(self, input):
        input, mean, std = self.norm(input)  
        input = input.unsqueeze(1) # 차원 1 생성
        d1 = self.first_block(input) # 
        m0 = self.down1(d1)
        u1 = self.up1(m0, d1)
        output = self.last_block(u1)
        output = output.squeeze(1)
        output = self.unnorm(output, mean, std)

        return output


class ConvBlock(nn.Module):

    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class Down(nn.Module):

    def __init__(self, in_chans, mid_chans, out_chans, end = False):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.mid_chans = mid_chans
        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, mid_chans, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(mid_chans),
            nn.ReLU(inplace = True),
            nn.Conv2d(mid_chans, out_chans, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace = True)
        )
        if not end: self.layers.add_module("pool",nn.MaxPool2d(2))
        

    def forward(self, x):
        return self.layers(x)


class Up(nn.Module):

    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.up = nn.ConvTranspose2d(in_chans, in_chans // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_chans, out_chans)

    def forward(self, x, concat_input):
        x = self.up(x)
        concat_output = torch.cat([concat_input, x], dim=1)
        return self.conv(concat_output)