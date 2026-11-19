"""DeepConvLSTM paper/Lasagne model port with explicit peephole connections.

Source: Sensors 2016, Sec. 3.1/3.3, Eq.4-8, Table1; author notebook
ea25e34da3a48b151a97bd3502802db547523b7f. This is a port, not author PyTorch code.
"""
from typing import List
import torch
from torch import nn


class LasagneRMSprop(torch.optim.Optimizer):
    """Lasagne uses sqrt(accumulator + epsilon), unlike torch.optim.RMSprop."""
    def __init__(self,params,lr=.001,rho=.9,eps=1e-6):
        super().__init__(params,dict(lr=lr,rho=rho,eps=eps))
    @torch.no_grad()
    def step(self,closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:continue
                state=self.state[p]
                if 'accumulator' not in state:state['accumulator']=torch.zeros_like(p)
                accu=state['accumulator']
                accu.mul_(group['rho']).addcmul_(p.grad,p.grad,value=1-group['rho'])
                p.addcdiv_(p.grad,(accu+group['eps']).sqrt(),value=-group['lr'])


@torch.jit.script
def recurrent(projected:torch.Tensor,recurrent_weight:torch.Tensor,
              pi:torch.Tensor,pf:torch.Tensor,po:torch.Tensor):
    hidden=recurrent_weight.size(0)
    h=projected.new_zeros((projected.size(0),hidden));c=torch.zeros_like(h)
    outputs=torch.jit.annotate(List[torch.Tensor],[])
    for t in range(projected.size(1)):
        gates=projected[:,t]+h@recurrent_weight
        i,f,g,o=gates.chunk(4,dim=1)
        i=torch.sigmoid(i+c*pi);f=torch.sigmoid(f+c*pf)
        c=f*c+i*torch.tanh(g)
        o=torch.sigmoid(o+c*po)
        h=o*torch.tanh(c)
        outputs.append(h)
    return torch.stack(outputs,dim=1)


class PeepholeLSTM(nn.Module):
    def __init__(self,inputs,hidden=128):
        super().__init__()
        self.input=nn.Linear(inputs,4*hidden)
        self.recurrent=nn.Parameter(torch.empty(hidden,4*hidden))
        self.pi=nn.Parameter(torch.empty(hidden));self.pf=nn.Parameter(torch.empty(hidden));self.po=nn.Parameter(torch.empty(hidden))
        for weight in self.input.weight.chunk(4,dim=0):nn.init.orthogonal_(weight)
        for weight in self.recurrent.chunk(4,dim=1):nn.init.orthogonal_(weight)
        nn.init.zeros_(self.input.bias)
        with torch.no_grad():self.input.bias[hidden:2*hidden].fill_(1.)
        for weight in [self.pi,self.pf,self.po]:nn.init.normal_(weight,0.,.1)

    def forward(self,x):return recurrent(self.input(x),self.recurrent,self.pi,self.pf,self.po)


class DeepConvLSTM(nn.Module):
    def __init__(self,channels,classes):
        super().__init__()
        layers=[]
        for i in range(4):
            conv=nn.Conv2d(1 if i==0 else 64,64,(5,1))
            nn.init.orthogonal_(conv.weight);nn.init.zeros_(conv.bias)
            layers.extend([conv,nn.ReLU()])
        self.conv=nn.Sequential(*layers)
        self.first=PeepholeLSTM(64*channels)
        self.second=PeepholeLSTM(128)
        self.dropout=nn.Dropout(.5)
        self.output=nn.Linear(128,classes)
        nn.init.orthogonal_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,x):
        x=self.conv(x).permute(0,2,1,3).flatten(2)
        x=self.first(x);x=self.second(x)
        # Sec.3.3 places dropout before dense layers. The released inference
        # notebook omits dropout operators; evaluation here disables dropout too.
        return self.output(self.dropout(x[:,-1]))
