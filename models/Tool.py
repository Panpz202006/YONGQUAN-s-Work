import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=4):
        super().__init__()

        self.channel=nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            ConvNormAct(in_channels,in_channels,1),
            ConvNormAct(in_channels,in_channels,1,act=nn.Sigmoid)
        )

    def forward(self, x):
        weight=self.channel(x)#B,C,1,1
        x=weight*x+x
        return x



class JA(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.ee=EdgeEnhance(in_channels)
        self.space=SpatialAttention(in_channels)
        self.channel=ChannelAttention(in_channels)

    def forward(self, x):
        x=self.ee(x)
        x=self.space(x)
        x=self.channel(x)
        return x



class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.space=nn.Sequential(
            ConvNormAct(in_channels,in_channels,7,padding=3,groups=in_channels,act=None),
            ConvNormAct(in_channels,in_channels,3,padding=1,groups=in_channels),
            ConvNormAct(in_channels,1,1,act=nn.Sigmoid)
        )
    

    def forward(self, x):
        weight=self.space(x)#B,1,H,W
        x=weight*x+x
        return x


class EdgeEnhance(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.context=ConvNormAct(in_channels,in_channels,7,padding=3,act=None,groups=in_channels)
        self.AP=nn.AvgPool2d(3,1,1)
        self.conv=ConvNormAct(in_channels,in_channels,1,act=nn.Sigmoid)
        
    def forward(self,x):
        x_ap=self.AP(self.context(x))
        weight=self.conv(x-x_ap)#B,C,H,W
        x_ee=weight*x+x
        return x_ee


class ConvNormAct(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 norm=nn.BatchNorm2d,
                 act=nn.ReLU):
        super().__init__()
        self.conv=nn.Conv2d(in_channels, out_channels, kernel_size,stride,padding,dilation,groups)
        self.in_channels=in_channels
        self.out_channels=out_channels
        
        if norm!=None:
            if norm==nn.BatchNorm2d:
                self.reshape=False
                self.norm=norm(out_channels)
            elif norm==nn.LayerNorm:
                self.reshape=True
                self.norm=norm(out_channels)
        else:
            self.norm=None
        if act!=None:
            self.act=act()
        else:
            self.act=None
    
    def forward(self,x:torch.Tensor):
        x=self.conv(x)
        if self.norm!=None:
            if self.reshape==True:
                B,C,H,W=x.shape
                x=x.permute(0,2,3,1).contiguous().view(B,H*W,C)
                x=self.norm(x)
                x=x.permute(0,2,1).contiguous().view(B,C,H,W)
            else:
                x=self.norm(x)
        if self.act!=None:
            x=self.act(x)
        return x


