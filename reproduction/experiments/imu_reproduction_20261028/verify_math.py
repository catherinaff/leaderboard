"""Numerical regression for the source-framework differences and paper losses."""
import numpy as np
import torch
from deepsense import TFGRU
from deepconvlstm import PeepholeLSTM,LasagneRMSprop
from onehar_paper_components import Reverse,PaperTrainingLoss,position_selection,proportional_counts,quality_score
from train import require_finite

try:require_finite(torch.tensor([float('nan')]),'test logits');raise AssertionError('NaN logits were accepted')
except FloatingPointError:pass

torch.set_num_threads(4);torch.manual_seed(42)
cell=PeepholeLSTM(5,3).double()
with torch.no_grad():cell.pi.zero_();cell.pf.zero_();cell.po.zero_()
reference=torch.nn.LSTM(5,3,batch_first=True).double()
with torch.no_grad():
    reference.weight_ih_l0.copy_(cell.input.weight);reference.weight_hh_l0.copy_(cell.recurrent.T)
    reference.bias_ih_l0.copy_(cell.input.bias);reference.bias_hh_l0.zero_()
x=torch.randn(2,8,5,dtype=torch.double,requires_grad=True)
actual=cell(x);expected,_=reference(x)
assert torch.allclose(actual,expected,atol=1e-12)
assert torch.allclose(torch.autograd.grad(actual.sum(),x,retain_graph=True)[0],torch.autograd.grad(expected.sum(),x)[0],atol=1e-12)
p=torch.nn.Parameter(torch.tensor([1.],dtype=torch.double));p.grad=torch.tensor([2.],dtype=torch.double)
LasagneRMSprop([p],lr=.001).step()
assert abs(p.item()-(1-.001*2/np.sqrt(.1*4+1e-6)))<1e-12

gru=TFGRU(2,3).double();v=torch.randn(2,4,2,dtype=torch.double)
got=gru(v).detach().numpy();h=np.zeros((2,3));outputs=[]
wg=np.concatenate([gru.gate.weight.detach().numpy().T,gru.wg.detach().numpy()])
wc=np.concatenate([gru.candidate.weight.detach().numpy().T,gru.wc.detach().numpy()])
for t in range(4):
    gates=np.concatenate([v[:,t].numpy(),h],1)@wg+gru.gate.bias.detach().numpy()
    r,u=np.split(1/(1+np.exp(-gates)),2,1)
    candidate=np.tanh(np.concatenate([v[:,t].numpy(),r*h],1)@wc+gru.candidate.bias.detach().numpy())
    h=u*h+(1-u)*candidate;outputs.append(h.copy())
assert np.allclose(got,np.stack(outputs,1),atol=1e-12)
wrong=torch.nn.GRU(2,3,batch_first=True).double()
with torch.no_grad():
    wrong.weight_ih_l0.copy_(torch.cat([gru.gate.weight,gru.candidate.weight]))
    wrong.weight_hh_l0.copy_(torch.cat([gru.wg.T,gru.wc.T]))
    wrong.bias_ih_l0.copy_(torch.cat([gru.gate.bias,gru.candidate.bias]));wrong.bias_hh_l0.zero_()
assert not torch.allclose(gru(v),wrong(v)[0],atol=1e-6),'Reset-after PyTorch GRU must not silently replace TF1 GRU'

v=torch.tensor([2.],requires_grad=True);Reverse.apply(v,.5).sum().backward();assert v.grad.item()==-.5
assert proportional_counts([0,0,1])=={0:0,1:1}
assert position_selection([.8,.2],[[1.,.8],[.8,1.]],contribution_weight=.5,correlation_weight=.5)==[0]
assert position_selection([.8,.2],[[1.,.8],[.8,1.]],contribution_weight=.5,correlation_weight=.5,sort_all=True)==[0,1]
q=quality_score(np.ones((3,2)),np.ones((3,2)),np.zeros((2,)),np.full(2,-1.),np.ones(2),dtw_weight=0.,cosine_weight=1.,biomechanical_weight=0.)
assert abs(q['score']-1)<1e-12 and q['joint_penalty']==0
loss=PaperTrainingLoss(torch.nn.Linear(4,2),temperature=.07,category_weight=0.,domain_weight=0.,negative_weight=.5)
a=torch.randn(3,4,requires_grad=True);b=torch.randn(3,4)
total,parts=loss(a,b,torch.tensor([0,0,1]),torch.tensor([0,1,0]))
assert torch.allclose(total,parts['alignment']);total.backward()
print('PASS peephole forward/gradients, Lasagne RMSprop, TF1 GRU semantics, GRL and oneHAR loss/selection/quality components.')
