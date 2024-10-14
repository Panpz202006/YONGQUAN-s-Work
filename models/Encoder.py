import torch
from torch import nn
from mamba_ssm import Mamba
from models.Tool import ConvNormAct

class Block(nn.Module):
    
    def __init__(self, input_channels=3, out_channels=3,pool=True,m_factor=[1.0,1.0],p_factor=[1.0,1.0]):
        super().__init__()
        if pool:
            self.pool=nn.MaxPool2d(2, stride=2)
        else:
            self.pool=None
        self.conv1=ConvNormAct(input_channels,out_channels,7,1,3)
        self.conv2=ConvNormAct(out_channels,out_channels,3,1,1)
        self.conv3=ConvNormAct(out_channels,out_channels,1)
        self.mamba=Mamba(
                d_model=out_channels, # Model dimension d_model
                d_state=16,  # SSM state expansion factor
                d_conv=4,    # Local convolution width
                expand=2,    # Block expansion factor
        )
        self.conv4=ConvNormAct(out_channels,out_channels,1)
        self.prompt1=ConvNormAct(out_channels,1,1,norm=nn.LayerNorm,act=None)
        self.prompt2=ConvNormAct(out_channels,1,1,norm=nn.LayerNorm,act=None)
        self.p_factor=p_factor
        self.m_factor=m_factor

    
    def forward(self, x):
        if self.pool!=None:
            x=self.pool(x)
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        B,C,H,W=x.shape
        p1=self.p_factor[0]*self.prompt1(x)
        x=self.m_factor[0]*self.mamba(x.permute(0,2,3,1).contiguous().view(B,H*W,C)).permute(0,2,1).contiguous().view(B,C,H,W)+self.m_factor[1]*x
        p2=self.p_factor[1]*self.prompt2(x)
        x=self.conv4(x)
        return x,p1,p2


        

class CPE(nn.Module):
    
    def __init__(self, input_channels=3, out_channels=[8,16,24,32,40,48],m_factor=[1.0,1.0],p_factor=[1.0,1.0]):
        super().__init__()
        self.block1=Block(input_channels,out_channels[0],pool=False,m_factor=m_factor,p_factor=p_factor)
        self.block2=Block(out_channels[0],out_channels[1],pool=True,m_factor=m_factor,p_factor=p_factor)
        self.block3=Block(out_channels[1],out_channels[2],pool=True,m_factor=m_factor,p_factor=p_factor)
        self.block4=Block(out_channels[2],out_channels[3],pool=True,m_factor=m_factor,p_factor=p_factor)
        self.block5=Block(out_channels[3],out_channels[4],pool=True,m_factor=m_factor,p_factor=p_factor)

    def forward(self, x):
        x_list=[]
        p1_list=[]
        p2_list=[]
        x,p1,p2=self.block1(x)
        x_list.append(x)
        p1_list.append(p1)
        p2_list.append(p2)
        x,p1,p2=self.block2(x)
        x_list.append(x)
        p1_list.append(p1)
        p2_list.append(p2)
        x,p1,p2=self.block3(x)
        x_list.append(x)
        p1_list.append(p1)
        p2_list.append(p2)
        x,p1,p2=self.block4(x)
        x_list.append(x)
        p1_list.append(p1)
        p2_list.append(p2)
        x,p1,p2=self.block5(x)
        x_list.append(x)
        p1_list.append(p1)
        p2_list.append(p2)
        return x_list,p1_list,p2_list
        