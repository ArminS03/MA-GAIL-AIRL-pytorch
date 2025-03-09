import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rl.acktr.utils_torch import fc, sample


class CategoricalPolicy(nn.Module):
    def __init__(self, ob_space, ac_space, ob_spaces, ac_spaces, nstack, device):
        super(CategoricalPolicy, self).__init__()
        self.device = device
        ob_shape = ob_space.shape[0] * nstack
        all_ob_shape = sum([obs.shape[0] for obs in ob_spaces]) * nstack
        nact = ac_space.n
        all_ac_shape = (sum([ac.n for ac in ac_spaces]) - nact) * nstack

        self.pi = nn.Sequential(
            fc(ob_shape, nh=128, init_scale=np.sqrt(2)),
            nn.ReLU(),
            fc(128, nh=128, init_scale=np.sqrt(2)),
            nn.ReLU(),
            fc(128, nact),
        )

        if len(ob_spaces) > 1:
            in_space = all_ob_shape + all_ac_shape
        else:
            in_space = all_ob_shape

        self.vf = nn.Sequential(
            fc(in_space, nh=256, init_scale=np.sqrt(2)),
            nn.ReLU(),
            fc(256, nh=256, init_scale=np.sqrt(2)),
            nn.ReLU(),
            fc(256, 1),
        )

        self.initial_state = []  # not stateful

    def forward(self, ob, obs, a_v):
        pi = self.compute_pi(ob)
        vf = self.compute_vf(obs, a_v)
        return pi, vf

    def compute_pi(self, ob):
        ob = torch.from_numpy(ob).to(self.device).float()
        pi = self.pi(ob)
        return pi

    def compute_vf(self, obs, a_v):
        obs = torch.from_numpy(obs).to(self.device)
        if a_v is not None:
            a_v = torch.from_numpy(a_v).to(self.device)
            Y = torch.cat([obs, a_v], dim=1)
        else:
            Y = obs
        Y = Y.float()
        vf = self.vf(Y)
        return vf

    def step_log_prob(self, ob, acts):
        pi = self.compute_pi(ob)
        acts = torch.from_numpy(acts).to(self.device)
        loss = -1 * nn.CrossEntropyLoss()(pi, acts)
        return loss.reshape((-1, 1))

    def step(self, ob, obs, a_v):
        pi, v = self.forward(ob, obs, a_v)
        a = sample(pi, self.device)
        return a, v[:, 0], []  # dummy state

    def best_step(self, ob, obs, a_v):
        pi, v = self.forward(ob, obs, a_v)
        a = torch.argmax(pi, dim=1)
        return a, v[:, 0], []  # dummy state

    def value(self, ob, a_v):
        vf = self.compute_vf(ob, a_v)
        return vf[:, 0]

    def policy_params(self):
        return self.pi.parameters()


class GaussianPolicy(nn.Module):
    def __init__(self, ob_space, ac_space, ob_spaces, ac_spaces, nstack, device):
        super(GaussianPolicy, self).__init__()
        self.device = device

        ob_shape = ob_space.shape[0] * nstack
        all_ob_shape = sum([obs.shape[0] for obs in ob_spaces]) * nstack
        nact = ac_space.shape[0]
        self.nact = nact
        all_ac_shape = (sum([ac.shape[0] for ac in ac_spaces]) - nact) * nstack

        self.pi = nn.Sequential(
            fc(ob_shape, nh=64, init_scale=np.sqrt(2)),
            nn.Tanh(),
            fc(64, nh=64, init_scale=np.sqrt(2)),
            nn.Tanh(),
            fc(64, nact, init_scale=0.01),
        )

        self.logstd = nn.Parameter(torch.zeros(nact))

        if len(ob_spaces) > 1:
            in_space = all_ob_shape + all_ac_shape
        else:
            in_space = all_ob_shape
        self.vf = nn.Sequential(
            fc(in_space, nh=64, init_scale=np.sqrt(2)),
            nn.Tanh(),
            fc(64, nh=64, init_scale=np.sqrt(2)),
            nn.Tanh(),
            fc(64, 1),
        )

        self.initial_state = []

    def forward(self, ob, obs, a_v):
        pi = self.compute_pi(ob)
        vf = self.compute_vf(obs, a_v)
        return pi, vf

    def compute_pi(self, ob):
        ob = torch.from_numpy(ob).to(self.device).float()
        pi = self.pi(ob)
        return pi

    def compute_vf(self, obs, a_v):
        obs = torch.from_numpy(obs).to(self.device).float()
        if a_v is not None:
            a_v = torch.from_numpy(a_v).to(self.device).float()
            Y = torch.cat([obs, a_v], dim=1)
        else:
            Y = obs
        vf = self.vf(Y)
        return vf

    def step(self, ob, obs, a_v):
        pi, v = self.forward(ob, obs, a_v)
        std = self.logstd.exp().unsqueeze(0).expand_as(pi)
        noise = torch.randn_like(pi)
        a = pi + noise * std
        return a, v[:, 0], []  # dummy state

    def value(self, ob, a_v):
        vf = self.compute_vf(ob, a_v)
        return vf[:, 0]

    def mean_std(self, ob):
        pi = self.compute_pi(ob)
        std = self.logstd.exp().unsqueeze(0).expand_as(pi)
        return torch.cat([pi, std], dim=1)

    # def best_step(self, ob, obs, a_v):
    #     pi, v = self.forward(ob, obs, a_v)
    #     a = torch.argmax(pi, dim=1)
    #     return a, v[:, 0], []  # dummy state

    def policy_params(self):
        return self.pi.parameters()
