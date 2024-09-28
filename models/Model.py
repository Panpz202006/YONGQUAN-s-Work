import torch
import torch.nn as nn

from models.Decoder import Decoder,PredictionHead
from models.Encoder import CPE






class Model(nn.Module):
    def __init__(self,input_channels=3, out_channels=[8,16,24,32,40],scale_factor=[1,2,4,8,16],m_factor=[1.0,1.0],p_factor=[1.0,1.0]) -> None:
        super().__init__()
        self.encoder=CPE(input_channels,out_channels,m_factor=[1.0,1.0],p_factor=[1.0,1.0])
        self.decoder=Decoder(out_channels)
        self.ph=PredictionHead(out_channels,scale_factor)

    def forward(self,x):
        x,p1,p2=self.encoder(x)
        x=self.decoder(x,p1,p2)
        x=self.ph(x)
        return x