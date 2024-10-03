import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from rl.acktr.utils_torch import fc
from rl.acktr.utils_torch import Scheduler
from irl.mack.kfac_torch import KFACOptimizer

disc_types = ['decentralized', 'decentralized-all']


class Discriminator(nn.Module):
    def __init__(self, ob_spaces, ac_spaces, state_only, discount,
                 nstack, index, device, disc_type='decentralized', hidden_size=128,
                 lr_rate=0.01, total_steps=50000, kfac_clip=0.001, max_grad_norm=0.5, l2=0.1):
        super(Discriminator, self).__init__()
        self.lr = Scheduler(v=lr_rate, nvalues=total_steps, schedule='linear')
        self.disc_type = disc_type
        if disc_type not in disc_types:
            assert False
        self.state_only = state_only
        self.gamma = discount
        self.device = device
        self.max_grad_norm = max_grad_norm

        ob_space = ob_spaces[index]
        ac_space = ac_spaces[index]
        self.ob_shape = ob_space.shape[0] * nstack
        self.ac_shape = ac_space.n * nstack
        self.all_ob_shape = sum([obs.shape[0] for obs in ob_spaces]) * nstack
        self.all_ac_shape = sum([ac.n for ac in ac_spaces]) * nstack    
        
        if disc_type == 'decentralized':
            self.obs = self.ob_shape
            self.act = self.ac_shape     
        elif disc_type == 'decentralized-all':
            self.obs = self.all_ob_shape
            self.act = self.all_ac_shape     
        else:
            assert False
        
        rew_input = self.obs
        if not self.state_only:
            rew_input = self.obs, self.act
            
        
        self.reward_fn = self.relu_net(rew_input, dout=1, hidden_size=hidden_size).to(device)
        self.value_fn = self.relu_net(self.obs, dout=1, hidden_size=hidden_size).to(device)
    
        
        # self.optimizer = KFACOptimizer(self, lr=lr_rate, kl_clip=kfac_clip)
        self.optimizer = optim.Adam(self.parameters(), lr=lr_rate, weight_decay=l2)

    def relu_net(self, din, layers=2, dout=1, hidden_size=128):
        seq_layers = []
        for i in range(layers):
            if i == 0:
                seq_layers.append(nn.Linear(din, hidden_size))
                seq_layers.append(nn.ReLU())
            else:
                seq_layers.append(nn.Linear(hidden_size, hidden_size))
                seq_layers.append(nn.ReLU())
        seq_layers.append(nn.Linear(hidden_size, dout))
        seq = nn.Sequential(*seq_layers)
        return seq
    
    def get_reward(self, obs, acs, obs_next, path_probs, discrim_score=False):               
        obs = self.to_torch(obs)
        acs = self.to_torch(acs)
        obs_next = self.to_torch(obs_next)
        
        raw_input = obs
        if not self.state_only:
            raw_input = torch.cat([obs, acs], axis=1)
        
        if discrim_score:
            log_q_tau = path_probs
            log_p_tau = self.reward_fn(raw_input) + self.gamma * self.value_fn(obs_next) - self.value_fn(obs)
            
            log_pq = torch.logsumexp(torch.stack([log_p_tau, log_q_tau]), dim=0)
            scores = torch.exp(log_p_tau - log_pq)
            eps=1e-20        
            score = torch.logit(scores, eps)
        else:
            score = self.reward_fn(raw_input)
        return score.detach().cpu().numpy()
            
    def calculate_loss(self, obs, act, nobs, lprobs, labels):
        raw_input = obs
        if not self.state_only:
            raw_input = torch.cat([obs, act], axis=1)
        
        log_q_tau = lprobs
        log_p_tau = self.reward_fn(raw_input) + self.gamma * self.value_fn(nobs) - self.value_fn(obs)
        log_pq = torch.logsumexp(torch.stack([log_p_tau, log_q_tau]), dim=0)

        loss = -torch.mean(labels * (log_p_tau - log_pq) + (1 - labels) * (log_q_tau - log_pq))
        return loss

    def train(self, g_obs, g_acs, g_nobs, g_probs, e_obs, e_acs, e_nobs, e_probs):
        labels_pi = torch.zeros((g_obs.shape[0], 1), device=self.device)
        labels_e = torch.ones((e_obs.shape[0], 1), device=self.device)
        labels = torch.cat([labels_pi, labels_e], dim=0)
        
        cur_lr = self.lr.value()
        for g in self.optimizer.param_groups:
                g['lr'] = cur_lr
        self.optimizer.lr = cur_lr
        
        obs = self.to_torch(np.concatenate([g_obs, e_obs], axis=0))
        act = self.to_torch(np.concatenate([g_acs, e_acs], axis=0))
        nobs = self.to_torch(np.concatenate([g_nobs, e_nobs], axis=0))
        lprobs = self.to_torch(np.concatenate([g_probs, e_probs], axis=0))
        
        loss = self.calculate_loss(obs, act, nobs, lprobs, labels)
        
        # if self.optimizer.steps % self.optimizer.Ts == 0:
        #     self.zero_grad()
        #     fisher_loss = -loss
        #     self.optimizer.acc_stats = True
        #     fisher_loss.backward(retain_graph=True)
        #     self.optimizer.acc_stats = False
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        return loss.detach().cpu()

    def save(self, save_path):
        torch.save(self.state_dict(), save_path) # TODO save optimizer parameters
        
    def load(self, load_path):
        self.load_state_dict(torch.load(load_path)) # TODO load optimizer parameters
        
    def to_torch(self, param):
        return torch.from_numpy(param).to(self.device)