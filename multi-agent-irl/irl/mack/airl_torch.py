import os.path as osp
import random
import time

import numpy as np
import torch
from irl.dataset import Dset
from irl.mack.kfac_discriminator_airl_torch import Discriminator
from rl import logger
from rl.acktr.utils_torch import (Scheduler, cat_entropy, discount_with_dones,
                                  multionehot, onehot)
from rl.common.torch_util import explained_variance, set_global_seeds
from scipy.stats import pearsonr, spearmanr
from irl.mack.kfac_torch import KFACOptimizer


class GeneralModel(object):
    def __init__(self, policy, ob_space, ac_space, nenvs, total_timesteps, device, nprocs=2, nsteps=200,
                 nstack=1, ent_coef=0.00, vf_coef=1, vf_fisher_coef=1.0, lr=0.25, max_grad_norm=0.5,
                 kfac_clip=0.001, lrschedule='linear', identical=None, weight_decay=1e-4):
        
        self.vf_fisher_coef = vf_fisher_coef
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.num_agents = num_agents = len(ob_space)
        self.n_actions = [ac_space[k].n for k in range(self.num_agents)]
        self.identical = identical
        if self.identical is None:
            self.identical = [False for _ in range(self.num_agents)]
            
        self.scale = [1 for _ in range(num_agents)]
        self.pointer = [i for i in range(num_agents)]

        h = 0
        for k in range(num_agents):
            if self.identical[k]:
                self.scale[h] += 1
            else:
                self.pointer[h] = k
                h = k
        self.pointer[h] = num_agents
        
        
        
        self.step_model = []

        
        self.step_model = []
        self.train_model = []
        for k in range(num_agents):
            if self.identical[k]:
                self.train_model.append(self.train_model[-1])
            else:      
                self.train_model.append(policy(ob_space[k], ac_space[k], ob_space, ac_space,
                                         nstack, self.device).to(self.device))

        self.optim = []
        self.clones = []
        # optimizers
        for k in range(num_agents):
            if self.identical[k]:
                self.optim.append(self.optim[-1])
                self.clones.append(self.clones[-1])
            else:
                self.optim.append(
                    KFACOptimizer(self.train_model[k], lr=lr, kl_clip=kfac_clip, weight_decay=weight_decay)
                )
                self.clones.append(
                    KFACOptimizer(self.train_model[k].pi, lr=lr, kl_clip=kfac_clip, weight_decay=weight_decay)
                )

        self.lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)
        self.clone_lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)
        self.initial_state = [self.train_model[k].initial_state for k in range(num_agents)]

    def train(self, obs, states, rewards, masks, actions, values):
        advs = [rewards[k] - values[k] for k in range(self.num_agents)]
        cur_lr = self.clone_lr.value()
    
        ob = np.concatenate(obs, axis=1)
            
        policy_loss = []
        value_loss = []
        policy_entropy = []
        
        for k in range(self.num_agents):
            if self.identical[k]:
                policy_loss.append(policy_loss[-1])
                value_loss.append(value_loss[-1])
                policy_entropy.append(policy_entropy[-1])
                continue
            A_v = None
                
            if self.num_agents > 1:
                action_v = []
                for _ in range(k, self.pointer[k]):
                        action_v.append(np.concatenate([multionehot(actions[i], self.n_actions[i])
                                                       for i in range(self.num_agents) if i != k], axis=1))
                action_v = np.concatenate(action_v, axis=0)
                A_v = action_v
            
            X = np.concatenate([obs[j] for j in range(k, self.pointer[k])], axis=0)
            X_v = np.concatenate([ob.copy() for _ in range(k, self.pointer[k])], axis=0)
            
            A = torch.from_numpy(np.concatenate([actions[j] for j in range(k, self.pointer[k])], axis=0)).to(self.device)
            ADV =  torch.from_numpy(np.concatenate([advs[j] for j in range(k, self.pointer[k])], axis=0)).to(self.device)
            R =  torch.from_numpy(np.concatenate([rewards[j] for j in range(k, self.pointer[k])], axis=0)).to(self.device)
            # calculations
            pi, vf = self.train_model[k](X, X_v, A_v)
            # print(f'X: {X.shape}')
            # print(f'X_v: {X_v.shape}')
            # print(f'A: {A.shape}')
            # print(f'ADV: {ADV.shape}')
            # print(f'R: {R.shape}')
            # print(f'pi: {pi.shape}')
            # print(f'vf: {vf.shape}')
            logpac = torch.nn.CrossEntropyLoss(reduction="none")(pi, A.long())
            entropy = torch.mean(cat_entropy(pi))
            pg_loss = torch.mean(ADV * logpac)
            pg_loss = pg_loss - self.ent_coef * entropy
            vf = torch.squeeze(vf)
            vf_loss = torch.mean(torch.nn.MSELoss()(vf, R))
            loss = pg_loss + self.vf_coef * vf_loss
            # print(f'logpac: {logpac.shape}')
            # print(f'pg_loss: {pg_loss.shape}')
            # print(f'vf_loss: {vf_loss.shape}')
            # print(f'loss: {loss.shape}')
            
            
            # optimizer loss
            if self.optim[k].steps % self.optim[k].Ts == 0:
                self.train_model[k].zero_grad()
                pg_fisher_loss = -torch.mean(logpac)
                
                value_noise = torch.randn(vf.size())
                if vf.is_cuda:
                    value_noise = value_noise.cuda()

                sample_values = vf + value_noise
                vf_fisher_loss = -(vf - sample_values.detach()).pow(2).mean()

                fisher_loss = pg_fisher_loss + vf_fisher_loss * self.vf_fisher_coef
                self.optim[k].acc_stats = True
                fisher_loss.backward(retain_graph=True)
                self.optim[k].acc_stats = False
    
            for g in self.optim[k].param_groups:
                g['lr'] = cur_lr / float(self.scale[k])
            self.optim[k].lr = cur_lr
            
            self.optim[k].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.train_model[k].parameters(), self.max_grad_norm)
            self.optim[k].step()
            policy_loss.append(pg_loss.clone().detach().cpu().numpy())
            value_loss.append(vf_loss.clone().detach().cpu().numpy())
            policy_entropy.append(entropy.clone().detach().cpu().numpy())
        return policy_loss, value_loss, policy_entropy
    
    def clone(self, obs, actions):
        cur_lr = self.clone_lr.value()
        lld_loss = []
        for k in range(self.num_agents):
            if self.identical[k]:
                continue

            X = np.concatenate([obs[j] for j in range(k, self.pointer[k])], axis=0)
            A = torch.from_numpy(np.concatenate([actions[j] for j in range(k, self.pointer[k])], axis=0)).to(self.device)
            pi = self.train_model[k].compute_pi(X)
            logpac = torch.nn.CrossEntropyLoss(reduction="none")(pi, A)
            lld = torch.mean(logpac)
            
            if self.clones[k].steps % self.clones[k].Ts == 0:
                self.train_model[k].zero_grad()
                pg_fisher_loss = -lld

                fisher_loss = pg_fisher_loss
                self.clones[k].acc_stats = True
                fisher_loss.backward(retain_graph=True)
                self.clones[k].acc_stats = False
            
            for g in self.clones[k].param_groups:
                g['lr'] = cur_lr / float(self.scale[k])
            self.optim[k].lr = cur_lr

            self.clones[k].zero_grad()
            lld.backward()
            torch.nn.utils.clip_grad_norm_(self.train_model[k].policy_params(), self.max_grad_norm)
            self.clones[k].step()
            lld_loss.append(lld.detach().cpu().numpy())
            
        return lld_loss

    def get_log_action_prob(self, obs, actions):
        action_prob = []
        for k in range(self.num_agents):
            if self.identical[k]:
                continue
            
            X = np.concatenate([obs[j] for j in range(k, self.pointer[k])], axis=0)
            A = torch.from_numpy(np.concatenate([actions[j] for j in range(k, self.pointer[k])], axis=0)).to(self.device)
            pi = self.train_model[k].compute_pi(X)
            
            log_pac = -1 * torch.nn.CrossEntropyLoss(reduction="none")(pi, A).detach().cpu().numpy()
            
            if self.scale[k] == 1:
                action_prob.append(log_pac)
            else:
                log_pac = np.split(log_pac, self.scale[k], axis=0)
                action_prob.append(log_pac)

        action_prob = np.array(action_prob)
        if action_prob.shape[0] == 1:
            action_prob = action_prob.squeeze(axis=0)
        return action_prob

    def get_log_action_prob_step(self, obs, actions):
        action_prob = []
        for k in range(self.num_agents):
            action_prob.append(self.train_model[k].step_log_prob(obs[k], actions[k]))
        return action_prob

    def save(self, save_path):
        models_dict = {}
        for k in range(self.num_agents):
            key1 = f'train_model{k+1}'
            models_dict[key1] = self.train_model[k].state_dict()
        torch.save(models_dict, save_path)

    def load(self, load_path):
        models_dict = torch.load(load_path)
        for k in range(self.num_agents):
            key1 = f'train_model{k+1}'
            self.train_model[k].load_state_dict(models_dict[key1])
        
    def step(self, ob, av):
        a, v, s = [], [], []
        obs = np.concatenate(ob, axis=1)
        for k in range(self.num_agents):
            a_v = np.concatenate([multionehot(av[i], self.n_actions[i])
                                  for i in range(self.num_agents) if i != k], axis=1)
            a_, v_, s_ = self.train_model[k].step(ob[k], obs, a_v)
            a.append(a_.detach().cpu().numpy())
            v.append(v_.detach().cpu().numpy())
            s.append(s_)
        return a, v, s

    def best_step(self, ob, av):
        a, v, s = [], [], []
        obs = np.concatenate(ob, axis=1)
        for k in range(self.num_agents):
            a_v = np.concatenate([multionehot(av[i], self.n_actions[i])
                                  for i in range(self.num_agents) if i != k], axis=1)
            a_, v_, s_ = self.train_model[k].best_step(ob[k], obs, a_v)
            a.append(a_.detach().cpu().numpy())
            v.append(v_.detach().cpu().numpy())
            s.append(s_)
        return a, v, s
             
    def value(self, obs, av):
        v = []
        ob = np.concatenate(obs, axis=1)
        for k in range(self.num_agents):
            a_v = np.concatenate([multionehot(av[i], self.n_actions[i])
                                  for i in range(self.num_agents) if i != k], axis=1)
            with torch.no_grad():
                v_ = self.train_model[k].value(ob, a_v)
            v.append(v_.detach().cpu().numpy())
        return v


class Runner(object):
    def __init__(self, env, model, discriminator, nsteps, nstack, gamma, lam, disc_type, nobs_flag=False):
        self.env = env
        self.model = model
        self.discriminator = discriminator
        self.disc_type = disc_type
        self.nobs_flag = nobs_flag
        self.num_agents = len(env.observation_space)
        self.nenv = nenv = env.num_envs
        self.batch_ob_shape = [
            (nenv * nsteps, nstack * env.observation_space[k].shape[0]) for k in range(self.num_agents)
        ]
        self.obs = [
            np.zeros((nenv, nstack * env.observation_space[k].shape[0])) for k in range(self.num_agents)
        ]
        self.actions = [np.zeros((nenv, )) for _ in range(self.num_agents)]
        obs = env.reset()
        self.update_obs(obs)
        self.gamma = gamma
        self.lam = lam
        self.nsteps = nsteps
        self.states = model.initial_state
        self.n_actions = [env.action_space[k].n for k in range(self.num_agents)]
        self.dones = [np.array([False for _ in range(nenv)]) for k in range(self.num_agents)]

    def update_obs(self, obs):
        # TODO: Potentially useful for stacking.
        self.obs = obs
        # for k in range(self.num_agents):
        #     ob = np.roll(self.obs[k], shift=-1, axis=1)
        #     ob[:, -1] = obs[:, 0]
        #     self.obs[k] = ob

        # self.obs = [np.roll(ob, shift=-1, axis=3) for ob in self.obs]
        # self.obs[:, :, :, -1] = obs[:, :, :, 0]

    def run(self):
        mb_obs = [[] for _ in range(self.num_agents)]
        mb_obs_next = [[] for _ in range(self.num_agents)]
        mb_true_rewards = [[] for _ in range(self.num_agents)]
        mb_rewards = [[] for _ in range(self.num_agents)]
        mb_report_rewards = [[] for _ in range(self.num_agents)]
        mb_actions = [[] for _ in range(self.num_agents)]
        mb_values = [[] for _ in range(self.num_agents)]
        mb_dones = [[] for _ in range(self.num_agents)]
        mb_masks = [[] for _ in range(self.num_agents)]
        mb_states = self.states
        for n in range(self.nsteps):
            actions, values, states = self.model.step(self.obs, self.actions)

            self.actions = actions
            for k in range(self.num_agents):
                mb_obs[k].append(np.copy(self.obs[k]))
                mb_actions[k].append(actions[k])
                mb_values[k].append(values[k])
                mb_dones[k].append(self.dones[k])
            actions_list = []
            for i in range(self.nenv):
                actions_list.append([onehot(actions[k][i], self.n_actions[k]) for k in range(self.num_agents)])
            obs, true_rewards, dones, _ = self.env.step(actions_list)

            for k in range(self.num_agents):
                for ni, done in enumerate(dones[k]):
                    if done:
                        obs[k][ni] = obs[k][ni] * 0.0
            for k in range(self.num_agents):
                mb_obs_next[k].append(np.copy(obs[k]))

            re_obs = self.obs
            re_actions = self.actions
            re_obs_next = obs
            re_path_prob = self.model.get_log_action_prob_step(re_obs, re_actions)  # [num_agent, nenv, 1]
            re_actions_onehot = [multionehot(re_actions[k], self.n_actions[k]) for k in range(self.num_agents)]

            # get reward from discriminator
            if self.disc_type == 'decentralized':
                rewards = []
                report_rewards = []
                for k in range(self.num_agents):
                    rewards.append(np.squeeze(self.discriminator[k].get_reward(re_obs[k],
                                                                               multionehot(re_actions[k], self.n_actions[k]),
                                                                               re_obs_next[k],
                                                                               re_path_prob[k],
                                                                               discrim_score=False))) # For competitive tasks, log(D) - log(1-D) empirically works better (discrim_score=True)
                    report_rewards.append(np.squeeze(self.discriminator[k].get_reward(re_obs[k],
                                                                               multionehot(re_actions[k], self.n_actions[k]),
                                                                               re_obs_next[k],
                                                                               re_path_prob[k],
                                                                               discrim_score=False)))
            elif self.disc_type == 'decentralized-all':
                rewards = []
                report_rewards = []
                for k in range(self.num_agents):
                    rewards.append(np.squeeze(self.discriminator[k].get_reward(np.concatenate(re_obs, axis=1),
                                                                               np.concatenate(re_actions_onehot, axis=1),
                                                                               np.concatenate(re_obs_next, axis=1),
                                                                               re_path_prob[k],
                                                                               discrim_score=False))) # For competitive tasks, log(D) - log(1-D) empirically works better (discrim_score=True)
                    report_rewards.append(np.squeeze(self.discriminator[k].get_reward(np.concatenate(re_obs, axis=1),
                                                                               np.concatenate(re_actions_onehot, axis=1),
                                                                               np.concatenate(re_obs_next, axis=1),
                                                                               re_path_prob[k],
                                                                               discrim_score=False)))
            else:
                assert False

            for k in range(self.num_agents):
                mb_rewards[k].append(rewards[k])
                mb_report_rewards[k].append(report_rewards[k])

            self.states = states
            self.dones = dones
            self.update_obs(obs)

            for k in range(self.num_agents):
                mb_true_rewards[k].append(true_rewards[k])
        for k in range(self.num_agents):
            mb_dones[k].append(self.dones[k])

        # batch of steps to batch of rollouts
        for k in range(self.num_agents):
            mb_obs[k] = np.asarray(mb_obs[k], dtype=np.float32).swapaxes(1, 0).reshape(self.batch_ob_shape[k])
            mb_obs_next[k] = np.asarray(mb_obs_next[k], dtype=np.float32).swapaxes(1, 0).reshape(self.batch_ob_shape[k])
            mb_true_rewards[k] = np.asarray(mb_true_rewards[k], dtype=np.float32).swapaxes(1, 0)
            mb_rewards[k] = np.asarray(mb_rewards[k], dtype=np.float32).swapaxes(1, 0)
            mb_report_rewards[k] = np.asarray(mb_report_rewards[k], dtype=np.float32).swapaxes(1, 0)
            mb_actions[k] = np.asarray(mb_actions[k], dtype=np.int32).swapaxes(1, 0)
            mb_values[k] = np.asarray(mb_values[k], dtype=np.float32).swapaxes(1, 0)
            mb_dones[k] = np.asarray(mb_dones[k], dtype=np.bool).swapaxes(1, 0)
            mb_masks[k] = mb_dones[k][:, :-1]
            mb_dones[k] = mb_dones[k][:, 1:]

        mb_returns = [np.zeros_like(mb_rewards[k]) for k in range(self.num_agents)]
        mb_report_returns = [np.zeros_like(mb_rewards[k]) for k in range(self.num_agents)]
        mb_true_returns = [np.zeros_like(mb_rewards[k]) for k in range(self.num_agents)]
        last_values = self.model.value(self.obs, self.actions)
        # discount/bootstrap off value fn
        for k in range(self.num_agents):
            for n, (rewards, report_rewards, true_rewards, dones, value) in enumerate(zip(mb_rewards[k], mb_report_rewards[k], mb_true_rewards[k], mb_dones[k], last_values[k].tolist())):
                rewards = rewards.tolist()
                report_rewards = report_rewards.tolist()
                dones = dones.tolist()
                true_rewards = true_rewards.tolist()
                if dones[-1] == 0:
                    rewards = discount_with_dones(rewards + [value], dones + [0], self.gamma)[:-1]
                    report_rewards = discount_with_dones(report_rewards + [value], dones + [0], self.gamma)[:-1]
                    true_rewards = discount_with_dones(true_rewards + [value], dones + [0], self.gamma)[:-1]
                else:
                    rewards = discount_with_dones(rewards, dones, self.gamma)
                    report_rewards = discount_with_dones(report_rewards, dones, self.gamma)
                    true_rewards = discount_with_dones(true_rewards, dones, self.gamma)
                mb_returns[k][n] = rewards
                mb_report_returns[k][n] = report_rewards
                mb_true_returns[k][n] = true_rewards

        for k in range(self.num_agents):
            mb_returns[k] = mb_returns[k].flatten()
            mb_report_returns[k] = mb_report_returns[k].flatten()
            mb_masks[k] = mb_masks[k].flatten()
            mb_values[k] = mb_values[k].flatten()
            mb_actions[k] = mb_actions[k].flatten()

        mh_actions = [multionehot(mb_actions[k], self.n_actions[k]) for k in range(self.num_agents)]
        mb_all_obs = np.concatenate(mb_obs, axis=1)
        mb_all_nobs = np.concatenate(mb_obs_next, axis=1)
        mh_all_actions = np.concatenate(mh_actions, axis=1)
        if self.nobs_flag:
            return mb_obs, mb_obs_next, mb_states, mb_returns, mb_report_returns, mb_masks, mb_actions, \
                   mb_values, mb_all_obs, mb_all_nobs, mh_actions, mh_all_actions, mb_rewards, mb_true_rewards, mb_true_returns
        else:
            return mb_obs, mb_states, mb_returns, mb_report_returns, mb_masks, mb_actions,\
                   mb_values, mb_all_obs, mh_actions, mh_all_actions, mb_rewards, mb_true_rewards, mb_true_returns


def learn(policy, expert, env, env_id, seed, total_timesteps=int(40e6), gamma=0.99, lam=0.95, log_interval=1, nprocs=32,
          nsteps=20, nstack=1, ent_coef=0.01, vf_coef=1, vf_fisher_coef=1.0, lr=0.25, max_grad_norm=0.5,
          kfac_clip=0.001, save_interval=100, lrschedule='linear', dis_lr=0.001, disc_type='decentralized',
          bc_iters=500, identical=None, d_iters=1, rew_scale=0.1, weight_decay=1e-4, l2=0.1):
    print(f"learning rate: {lr}")
    print(f"discriminator lr: {dis_lr}")
    print(f"weight decay: {weight_decay}")
    print(f"bc iters: {bc_iters}")
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    set_global_seeds(seed)
    buffer = None
    update_policy_until = 10

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    num_agents = (len(ob_space))
    make_model = lambda: GeneralModel(policy, ob_space, ac_space, nenvs, total_timesteps, device, nprocs=nprocs, nsteps=nsteps,
                               nstack=nstack, ent_coef=ent_coef, vf_coef=vf_coef, vf_fisher_coef=vf_fisher_coef,
                               lr=lr, max_grad_norm=max_grad_norm, kfac_clip=kfac_clip,
                               lrschedule=lrschedule, identical=identical, weight_decay=weight_decay)
    
    if save_interval and logger.get_dir():
        import cloudpickle
        with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
            fh.write(cloudpickle.dumps(make_model))
    model = make_model()
    if disc_type == 'decentralized' or disc_type == 'decentralized-all':
        discriminator = [
            Discriminator(ob_space, ac_space,
                          state_only=True, discount=gamma, nstack=nstack, index=k, device=device,
                          disc_type=disc_type,
                          total_steps=total_timesteps // (nprocs * nsteps),
                          lr_rate=dis_lr, l2=l2).double() for k in range(num_agents)
        ]
    else:
        assert False

    # add reward regularization
    # if env_id == 'simple_tag':
    #     reward_reg_loss = torch.mean(
    #         torch.square(discriminator[0].reward + discriminator[3].reward) +
    #         torch.square(discriminator[1].reward + discriminator[3].reward) +
    #         torch.square(discriminator[2].reward + discriminator[3].reward)
    #     ) + rew_scale * torch.mean(
    #         torch.maximum(0.0, 1 - discriminator[0].reward) +
    #         torch.maximum(0.0, 1 - discriminator[1].reward) +
    #         torch.maximum(0.0, 1 - discriminator[2].reward) +
    #         torch.maximum(0.0, discriminator[3].reward + 1)
    #     )
    #     reward_reg_lr = tf.placeholder(tf.float32, ())
    #     reward_reg_optim = tf.train.AdamOptimizer(learning_rate=reward_reg_lr)
    #     reward_reg_train_op = reward_reg_optim.minimize(reward_reg_loss)

    runner = Runner(env, model, discriminator, nsteps=nsteps, nstack=nstack, gamma=gamma, lam=lam, disc_type=disc_type,
                    nobs_flag=True)
    nbatch = nenvs * nsteps
    tstart = time.time()

    for _ in range(bc_iters):
        e_obs, e_actions, e_nobs, _, _ = expert.get_next_batch(nenvs * nsteps)
        e_a = [np.argmax(e_actions[k], axis=1) for k in range(len(e_actions))]
        lld_loss = model.clone(e_obs, e_a)
        # print(lld_loss)


    for update in range(1, total_timesteps // nbatch + 1):
        obs, obs_next, states, rewards, report_rewards, masks, actions, values, all_obs, all_nobs,\
        mh_actions, mh_all_actions, mh_rewards, mh_true_rewards, mh_true_returns = runner.run()

        total_loss = np.zeros((num_agents, d_iters))

        idx = 0
        idxs = np.arange(len(all_obs))
        random.shuffle(idxs)
        all_obs = all_obs[idxs]
        mh_actions = [mh_actions[k][idxs] for k in range(num_agents)]
        mh_obs = [obs[k][idxs] for k in range(num_agents)]
        mh_obs_next = [obs_next[k][idxs] for k in range(num_agents)]
        mh_values = [values[k][idxs] for k in range(num_agents)]

        if buffer:
            buffer.update(mh_obs, mh_actions, mh_obs_next, all_obs, mh_values)
        else:
            buffer = Dset(mh_obs, mh_actions, mh_obs_next, all_obs, mh_values, randomize=True, num_agents=num_agents,
                          nobs_flag=True)

        d_minibatch = nenvs * nsteps

        for d_iter in range(d_iters):
            e_obs, e_actions, e_nobs, e_all_obs, _ = expert.get_next_batch(d_minibatch)
            g_obs, g_actions, g_nobs, g_all_obs, _ = buffer.get_next_batch(batch_size=d_minibatch)

            e_a = [np.argmax(e_actions[k], axis=1) for k in range(len(e_actions))]
            g_a = [np.argmax(g_actions[k], axis=1) for k in range(len(g_actions))]

            g_log_prob = model.get_log_action_prob(g_obs, g_a)
            e_log_prob = model.get_log_action_prob(e_obs, e_a)
            if disc_type == 'decentralized':
                for k in range(num_agents):
                    total_loss[k, d_iter] = discriminator[k].train(
                        g_obs[k],
                        g_actions[k],
                        g_nobs[k],
                        g_log_prob[k].reshape([-1, 1]),
                        e_obs[k],
                        e_actions[k],
                        e_nobs[k],
                        e_log_prob[k].reshape([-1, 1])
                    )
            elif disc_type == 'decentralized-all':
                g_obs_all = np.concatenate(g_obs, axis=1)
                g_actions_all = np.concatenate(g_actions, axis=1)
                g_nobs_all = np.concatenate(g_nobs, axis=1)
                e_obs_all = np.concatenate(e_obs, axis=1)
                e_actions_all = np.concatenate(e_actions, axis=1)
                e_nobs_all = np.concatenate(e_nobs, axis=1)
                for k in range(num_agents):
                    total_loss[k, d_iter] = discriminator[k].train(
                        g_obs_all,
                        g_actions_all,
                        g_nobs_all,
                        g_log_prob[k].reshape([-1, 1]),
                        e_obs_all,
                        e_actions_all,
                        e_nobs_all,
                        e_log_prob[k].reshape([-1, 1])
                    )
            else:
                assert False

            # if env_id == 'simple_tag':
            #     if disc_type == 'decentralized':
            #         feed_dict = {discriminator[k].obs: np.concatenate([g_obs[k], e_obs[k]], axis=0)
            #                      for k in range(num_agents)}
            #     elif disc_type == 'decentralized-all':
            #         feed_dict = {discriminator[k].obs: np.concatenate([g_obs_all, e_obs_all], axis=0)
            #                      for k in range(num_agents)}
            #     else:
            #         assert False
            #     feed_dict[reward_reg_lr] = discriminator[0].lr.value()
            #     ## TODO
            #     model.sess.run(reward_reg_train_op, feed_dict=feed_dict)

            idx += 1

        if update > update_policy_until:  # 10
            policy_loss, value_loss, policy_entropy = model.train(obs, states, rewards, masks, actions, values)
        model.old_obs = obs
        nseconds = time.time() - tstart
        fps = int((update * nbatch) / nseconds)
        if update % log_interval == 0 or update == 1:
            ev = [explained_variance(values[k], rewards[k]) for k in range(model.num_agents)]
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update * nbatch)
            logger.record_tabular("fps", fps)

            for k in range(model.num_agents):
                logger.record_tabular("explained_variance %d" % k, float(ev[k]))
                if update > update_policy_until:
                    logger.record_tabular("policy_entropy %d" % k, float(policy_entropy[k]))
                    logger.record_tabular("policy_loss %d" % k, float(policy_loss[k]))
                    logger.record_tabular("value_loss %d" % k, float(value_loss[k]))
                    try:
                        logger.record_tabular('pearson %d' % k, float(
                            pearsonr(report_rewards[k].flatten(), mh_true_returns[k].flatten())[0]))
                        logger.record_tabular('spearman %d' % k, float(
                            spearmanr(report_rewards[k].flatten(), mh_true_returns[k].flatten())[0]))
                        logger.record_tabular('reward %d' % k, float(np.mean(mh_rewards[k])))
                        logger.record_tabular('true reward %d' % k, float(np.mean(mh_true_rewards[k])))
                    except:
                        pass

            total_loss_m = np.mean(total_loss, axis=1)
            for k in range(num_agents):
                logger.record_tabular("total_loss %d" % k, total_loss_m[k])
            logger.dump_tabular()

        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
            savepath = osp.join(logger.get_dir(), 'm_%.5i' % update)
            print('Saving to', savepath)
            model.save(savepath)
            if disc_type == 'decentralized' or disc_type == 'decentralized-all':
                for k in range(num_agents):
                    savepath = osp.join(logger.get_dir(), 'd_%d_%.5i' % (k, update))
                    discriminator[k].save(savepath)
            else:
                assert False
    env.close()
