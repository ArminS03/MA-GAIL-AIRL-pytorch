#!/usr/bin/env python
import itertools
import os

import click

# relative_path = sys.path[0]
# relative_path = relative_path.replace('multi-agent-irl', 'multi-agent-particle-envs')
# sys.path.append(relative_path)
from rl import bench, logger
from rl.common import set_global_seeds
from sandbox.mack.acktr_cont_vmas import learn
from sandbox.mack.policies import GaussianPolicy
from vmas import make_env
from rl.common.vec_env.subproc_vec_env_vasm import SubprocVecEnv
from gym.spaces import Box, Tuple
import gym
import numpy as np


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class ActionClipWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(ActionClipWrapper, self).__init__(env)
        assert isinstance(env.action_space, Tuple), "Action space must be of type Tuple."
        # Ensure all spaces within the tuple are Box spaces
        for space in env.action_space.spaces:
            assert isinstance(space, Box), "All spaces in Tuple action space must be of type Box."
    
    def action(self, action):
        # Iterate over each action space and corresponding action, clipping within the bounds of each Box
        clipped_action = [
            np.clip(act, box_space.low, box_space.high)
            for act, box_space in zip(action, self.action_space.spaces)
        ]
        return clipped_action
    
    

def train(logdir, env_id, num_timesteps, lr, timesteps_per_batch, seed, num_cpu):


    logger.configure(logdir, format_strs=['stdout', 'log', 'json', 'tensorboard'])
    set_global_seeds(seed)
    def create_env(rank):
        def _thunk():
            env = make_env(
                    scenario=env_id,
                    num_envs=1,
                    device='cpu',
                    continuous_actions=True,
                    seed=(seed+rank),
                    max_steps=200,
                    n_agents=5,
            )
            env = ActionClipWrapper(env)
            return env
        return _thunk
    
    env = SubprocVecEnv([create_env(i) for i in range(num_cpu)], is_multi_agent=True)
    policy_fn = GaussianPolicy
    learn(policy_fn, env, seed, total_timesteps=int(num_timesteps * 1.1), nprocs=num_cpu,
          nsteps=timesteps_per_batch // num_cpu, lr=lr, ent_coef=0.00, identical=None, save_interval=1000)
    env.close()


@click.command()
@click.option('--logdir', type=click.STRING, default='./results/target_model')
# @click.option('--env', type=click.STRING, default='simple_world_comm')
@click.option('--env', type=click.STRING, default='passage')
@click.option('--lr', type=click.FLOAT, default=0.1)
@click.option('--seed', type=click.INT, default=1)
@click.option('--batch_size', type=click.INT, default=1000)
def main(logdir, env, lr, seed, batch_size):
    env_ids = [env]
    lrs = [lr]
    seeds = [seed]
    batch_sizes = [batch_size]

    print('logging to: ' + logdir)

    for env_id, seed, lr, batch_size in itertools.product(env_ids, seeds, lrs, batch_sizes):
        train(logdir + '/' + env_id + '/l-{}-b-{}/seed-{}'.format(lr, batch_size, seed),
              env_id, 5e7, lr, batch_size, seed, batch_size // 500)


if __name__ == "__main__":
    main()
