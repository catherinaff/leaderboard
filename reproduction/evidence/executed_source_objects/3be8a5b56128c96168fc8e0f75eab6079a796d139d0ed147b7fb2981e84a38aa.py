"""Missing training components reconstructed from oneHAR, IMWUT 2025.

Equations 18–29 and Algorithm 1. These are NOT author training code and do not
fabricate GPT-4o responses, motion sequences or synthetic IMU observations.
See ONEHAR_AUDIT.md before enabling a full training run.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx,value,coefficient):
        ctx.coefficient=coefficient
        return value.view_as(value)
    @staticmethod
    def backward(ctx,gradient):return -ctx.coefficient*gradient,None


class PaperTrainingLoss(nn.Module):
    def __init__(self,domain_discriminator,*,temperature,category_weight,
                 domain_weight,negative_weight,gradient_reversal=1.):
        super().__init__()
        self.domain_discriminator=domain_discriminator
        self.temperature=temperature;self.category_weight=category_weight
        self.domain_weight=domain_weight;self.negative_weight=negative_weight
        self.gradient_reversal=gradient_reversal

    def forward(self,imu_features,text_features,category_ids,domain_ids):
        if (category_ids<0).any():
            raise ValueError('Paper coarse categories do not cover these labels; an explicit extension protocol is required')
        imu=F.normalize(imu_features,dim=-1);text=F.normalize(text_features,dim=-1)
        similarity=imu@text.T/self.temperature
        alignment=F.cross_entropy(similarity,torch.arange(len(imu),device=imu.device))
        same=(category_ids[:,None]==category_ids[None,:]).to(imu.dtype)
        s=imu@imu.T
        category=-(same*s-self.negative_weight*(1-same)*s).sum()/len(imu)
        reversed_features=Reverse.apply(imu_features,self.gradient_reversal)
        # Discriminator must return logits, rather than applying Softmax twice.
        domain=F.cross_entropy(self.domain_discriminator(reversed_features),domain_ids)
        total=alignment+self.category_weight*category+self.domain_weight*domain
        return total,{'alignment':alignment,'category':category,'domain':domain}


def proportional_counts(training_labels):
    values,counts=np.unique(training_labels,return_counts=True)
    return {int(label):int(counts.max()-count) for label,count in zip(values,counts)}


def position_selection(contribution,correlation,*,contribution_weight,correlation_weight,
                       contribution_threshold=.5,correlation_threshold=.5,sort_all=False):
    """Algorithm 1; use sort_all for the inference-time complete ordering."""
    contribution=np.asarray(contribution,np.float64)
    correlation=np.asarray(correlation,np.float64)
    if correlation.shape!=(len(contribution),len(contribution)) or not np.isfinite(correlation).all():
        raise ValueError('A complete, finite sensor correlation matrix is required')
    if not np.isfinite(contribution).all() or (contribution<0).any() or contribution.sum()<=0:
        raise ValueError('Invalid LLM contribution scores')
    contribution=contribution/contribution.sum()  # Eq.27
    first=int(contribution.argmax());selected=[first]
    remaining=set(range(len(contribution)))-{first}
    while remaining:
        candidate=max(sorted(remaining),key=lambda j:contribution_weight*contribution[j]+correlation_weight*correlation[first,j])
        if not sort_all and (contribution[candidate]<contribution_threshold or correlation[first,candidate]<correlation_threshold):break
        selected.append(candidate);remaining.remove(candidate)
    return selected


def dtw_distance(virtual,real):
    virtual=np.asarray(virtual,np.float64);real=np.asarray(real,np.float64)
    if virtual.ndim!=2 or real.ndim!=2 or virtual.shape[1]!=real.shape[1]:raise ValueError('Expect time × channels with matching channels')
    previous=np.full(len(real)+1,np.inf);previous[0]=0.
    for x in virtual:
        row=np.full_like(previous,np.inf)
        for j,y in enumerate(real,1):row[j]=np.linalg.norm(x-y)+min(row[j-1],previous[j],previous[j-1])
        previous=row
    return float(previous[-1])


def quality_score(virtual,real,joint_angles,joint_min,joint_max,*,dtw_weight,
                  cosine_weight,biomechanical_weight):
    """Literal Eq.26. Metric normalization/sign ambiguities require calibration."""
    virtual=np.asarray(virtual,np.float64);real=np.asarray(real,np.float64)
    if virtual.shape!=real.shape:raise ValueError('Aligned synthetic/real observations are required for cosine similarity')
    a,b=virtual.ravel(),real.ravel()
    cosine=float(a@b/max(np.linalg.norm(a)*np.linalg.norm(b),1e-12))
    theta=np.asarray(joint_angles,np.float64)
    penalty=float((np.maximum(0,theta-joint_max)+np.maximum(0,joint_min-theta)).sum())
    distance=dtw_distance(virtual,real)
    score=dtw_weight*distance+cosine_weight*cosine+biomechanical_weight*penalty
    return {'score':float(score),'dtw_distance':distance,'cosine_similarity':cosine,'joint_penalty':penalty,
            'literal_paper_threshold_pass':bool(score>=.7)}
