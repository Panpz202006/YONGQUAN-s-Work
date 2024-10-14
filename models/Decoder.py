import torch
import torch.nn as nn

from models.Tool import JA
from models.Tool import ConvNormAct

class CFBlock(nn.Module):
    def __init__(self,in_channels,kernel,sample1=None,sample2=None):
        super().__init__()
        self.sample1=sample1
        self.sample2=sample2
        self.extract=ConvNormAct(in_channels,in_channels//4,kernel,padding=kernel//2,groups=in_channels//4)
        self.fusion=ConvNormAct(in_channels//4,in_channels//4,1,act=None)

    def forward(self,x):
        if self.sample1!=None:
            x=self.sample1(x)
        x=self.extract(x)
        x=self.fusion(x)
        if self.sample2!=None:
            x=self.sample2(x)
        return x

class IG(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.extract=ConvNormAct(in_channels,in_channels,3,padding=1,groups=in_channels)
        self.local=ConvNormAct(in_channels,in_channels,1,act=None)
        self.attn=JA(in_channels)

    def forward(self,x):
        x=self.extract(x)
        x=self.local(x)
        x=self.attn(x)
        return x

class CF(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.c=in_channels
        self.cf1=CFBlock(in_channels,3)
        self.cf2=CFBlock(in_channels,5)
        self.cf3=CFBlock(in_channels,7)
        self.cf4=CFBlock(in_channels,9)
        

    def forward(self,x):
        x1=self.cf1(x)
        x2=self.cf2(x)
        x3=self.cf3(x)
        x4=self.cf4(x)
        out=torch.cat([x1,x2,x3,x4],dim=1)
        return out
    

class CE(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.cf=CF(in_channels)
        self.ig=IG(in_channels)

    def forward(self,x):
        short_cut=x
        x=self.cf(x)
        x=self.ig(x)+short_cut
        return x


class Prompt(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.conv=nn.Sequential(
            ConvNormAct(in_channels,in_channels,1,act=None)
        )
    def forward(self,x,p1,p2):
        x=x+p1+p2
        x=self.conv(x)
        return x


class CAF(nn.Module):
    def __init__(self,in_channels1,in_channels2):
        super().__init__()
        self.pro=nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvNormAct(in_channels1,in_channels2,1,act=None),
        )
        self.prompt=Prompt(in_channels2)
        self.norm=nn.BatchNorm2d(in_channels2)
        self.soft=nn.Softmax(-1)
        self.local=nn.Sequential(
            ConvNormAct(in_channels2,in_channels2,1)
        )
    
    
    
    def forward(self,saliency,details,p1,p2):
        B,C,H,W=details.shape
        N=H*W
        saliency_q=self.pro(saliency)
        details_k=self.prompt(details,p1,p2)
        saliency_v=saliency_q.view(B,C,H*W)#B,C,N
        details_v=details.view(B,C,H*W)#B,C,N
        saliency_q=saliency_q.view(B,C,H*W)#B,C,N
        details_k=details_k.view(B,C,H*W)#B,C,N
        attn=saliency_q@details_k.transpose(1,2)*(N**-0.5)#B,C,C
        attn_score=self.soft(attn.view(B,C*C)).view(B,C,C)#B,C,C
        attn_s=attn_score@saliency_v+saliency_v#B,C,N
        attn_d=attn_score@details_v+details_v#B,C,N
        x=attn_s+attn_d#B,C,N
        x=x.view(B,C,H,W)#B,C,H,W
        x=self.local(x)
        return x



class Decoder(nn.Module):
    def __init__(self,in_channels=[64,128,256,512,512]):
        super().__init__()
        self.num_layer=len(in_channels)
        self.ce=nn.ModuleList()
        for i_layer in range(self.num_layer):
            self.ce.append(CE(in_channels[i_layer]))
        self.caf=nn.ModuleList()
        for i_layer in range(self.num_layer-1):
            self.caf.append(CAF(in_channels[i_layer+1],in_channels[i_layer]))
        
    def forward(self,x,p1,p2):
        x_list=[]
        input=x[-1]
        for i in range(-1, -len(self.ce)-1, -1):
            x_d=self.ce[i](input)
            x_list.append(x_d)
            if i!=-self.num_layer:
                input=self.caf[i](x_d,x[i-1],p1[i-1],p2[i-1])
        return x_list



class PH_Block(nn.Module):
    def __init__(self,in_channels,scale_factor=1):
        super().__init__()
        if scale_factor>1:
            self.upsample=nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        else:
            self.upsample=None
        self.pro=nn.Sequential(
            nn.Conv2d(in_channels,1,1),
            nn.Sigmoid()
        )

    def forward(self,x:torch.Tensor):
        if self.upsample!=None:
            x=self.upsample(x)
        x=self.pro(x)
        return x

class PredictionHead(nn.Module):
    def __init__(self,in_channels=[64,128,256,512,512],scale_factor=[1,2,4,8,16]):
        super().__init__()
        self.final=nn.ModuleList()
        self.num_layer=len(in_channels)
        for i_layer in range(self.num_layer):
            self.final.append(PH_Block(in_channels[i_layer],scale_factor[i_layer]))

    def forward(self,x):
        x=x[::-1]
        x_list=[]
        for i in range(self.num_layer):
            x_list.append(self.final[i](x[i]))
        return x_list
