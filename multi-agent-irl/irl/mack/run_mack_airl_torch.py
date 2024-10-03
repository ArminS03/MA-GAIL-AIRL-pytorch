#!/usr/bin/env python
import logging
import os
import itertools
import click
import gym
import sys
relative_path = sys.path[0]

relative_path = relative_path.replace('multi-agent-irl', 'multi-agent-particle-envs')
sys.path.append(relative_path)

import make_env
from rl import bench
from rl import logger
from rl.common.torch_util import set_global_seeds
from rl.common.vec_env.subproc_vec_env import SubprocVecEnv
from irl.dataset import MADataSet
from irl.mack.airl_torch import learn
from sandbox.mack.policies_torch import CategoricalPolicy


def train(logdir, env_id, num_timesteps, lr, timesteps_per_batch, seed, num_cpu, expert_path,
          traj_limitation, ret_threshold, dis_lr, disc_type='decentralized', bc_iters=500, d_iters=1,
          rew_scale=0.1, weight_decay=1e-4, l2=0.1):
    def create_env(rank):
        def _thunk():
            env = make_env.make_env(env_id)
            env.seed(seed + rank)
            env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)),
                                allow_early_resets=True)
            gym.logger.setLevel(logging.WARN)
            return env
        return _thunk

    logger.configure(logdir, format_strs=['stdout', 'log', 'json', 'tensorboard'])

    set_global_seeds(seed)
    env = SubprocVecEnv([create_env(i) for i in range(num_cpu)], is_multi_agent=True)
    print(num_cpu)
    policy_fn = CategoricalPolicy
    expert = MADataSet(expert_path, ret_threshold=ret_threshold, traj_limitation=traj_limitation, nobs_flag=True)
    learn(policy_fn, expert, env, env_id, seed, total_timesteps=int(num_timesteps * 1.1), nprocs=num_cpu,
          nsteps=timesteps_per_batch // num_cpu, lr=lr, ent_coef=0.0, dis_lr=dis_lr,
          disc_type=disc_type, bc_iters=bc_iters, identical=make_env.get_identical(env_id), d_iters=d_iters,
          rew_scale=rew_scale, weight_decay=weight_decay, save_interval=500, l2=l2)
    env.close()


@click.command()
@click.option('--logdir', type=click.STRING, default='./results/torch_trained')
@click.option('--env', type=click.STRING, default='simple_speaker_listener')
@click.option('--expert_path', type=click.STRING,
    default='./results/target_model/simple_speaker_listener/l-0.1-b-1000/seed-1/checkpoint20000-10000tra.pkl')
@click.option('--seed', type=click.INT, default=1)
@click.option('--traj_limitation', type=click.INT, default=200)
@click.option('--ret_threshold', type=click.FLOAT, default=-10)
@click.option('--dis_lr', type=click.FLOAT, default=0.1)
@click.option('--disc_type', type=click.Choice(['decentralized', 'decentralized-all']),
              default='decentralized')
@click.option('--bc_iters', type=click.INT, default=500)
@click.option('--l2', type=click.FLOAT, default=0.1)
@click.option('--d_iters', type=click.INT, default=1)
@click.option('--rew_scale', type=click.FLOAT, default=0)
def main(logdir, env, expert_path, seed, traj_limitation, ret_threshold, dis_lr, disc_type, bc_iters, l2, d_iters,
         rew_scale):
    env_ids = [env]
    lrs = [0.01]
    l2 = 0.1
    seeds = [1]
    batch_sizes = [1000]
    weight_decay = 0
    bc_iters = 500
    d_iters = 1

    for env_id, seed, lr, batch_size in itertools.product(env_ids, seeds, lrs, batch_sizes):
        train(logdir + '/airl/' + env_id + '/' + disc_type + '/s-{}/l-{}-d-{}-b-{}-c-{}-iter-{}-l2-{}/seed-{}'.format(
              traj_limitation, lr, lr, batch_size, bc_iters, d_iters, lr, seed),
              env_id, 4e7, lr, batch_size, seed, batch_size // 250, expert_path,
              traj_limitation, ret_threshold, lr, disc_type=disc_type, bc_iters=bc_iters, d_iters=d_iters,
              rew_scale=rew_scale, weight_decay=weight_decay, l2=lr)


if __name__ == "__main__":
    main()
