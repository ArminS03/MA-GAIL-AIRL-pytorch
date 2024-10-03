import torch
import torch.nn as nn
import numpy as np

def sample(logits, device, axis=1):
    noise = torch.rand(logits.shape).to(device)
    return torch.argmax(logits - torch.log(-torch.log(noise)), dim=axis)

def fc(ns, nh, init_scale=1.0):
    linear = nn.Linear(ns, nh)
    nn.init.orthogonal_(linear.weight, gain=init_scale)
    nn.init.constant_(linear.bias, 0.0)
    return linear

def constant(p):
    return 1

def linear(p):
    return 1-p


def middle_drop(p):
    eps = 0.75
    if 1-p<eps:
        return eps*0.1
    return 1-p

def double_linear_con(p):
    p *= 2
    eps = 0.125
    if 1-p<eps:
        return eps
    return 1-p

def double_middle_drop(p):
    eps1 = 0.75
    eps2 = 0.25
    if 1-p<eps1:
        if 1-p<eps2:
            return eps2*0.5
        return eps1*0.1
    return 1-p

schedules = {
    'linear':linear,
    'constant':constant,
    'double_linear_con':double_linear_con,
    'middle_drop':middle_drop,
    'double_middle_drop':double_middle_drop
}

class Scheduler(object):
    def __init__(self, v, nvalues, schedule):
        self.n = 0.
        self.v = v
        self.nvalues = nvalues
        self.schedule = schedules[schedule]

    def value(self):
        current_value = self.v*self.schedule(self.n/self.nvalues)
        self.n += 1.
        return current_value

    def value_steps(self, steps):
        return self.v*self.schedule(steps/self.nvalues)


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r*(1.-done) # fixed off by one bug
        discounted.append(r)
    return discounted[::-1]


def cat_entropy(logits):
    a0 = logits - torch.max(logits, dim=1, keepdim=True).values
    ea0 = torch.exp(a0)
    z0 = torch.sum(ea0, dim=1, keepdim=True)
    p0 = ea0 / z0
    return torch.sum(p0 * (torch.log(z0) - a0), dim=1)


def mse(pred, target):
    return torch.square(pred-target)/2


def onehot(value, depth):
    a = np.zeros([depth])
    a[value] = 1
    return a


def multionehot(values, depth):
    a = np.zeros([values.shape[0], depth])
    for i in range(values.shape[0]):
        a[i, int(values[i])] = 1
    return a