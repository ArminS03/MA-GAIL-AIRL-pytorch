import gym
import click
import sys
relative_path = sys.path[0]
relative_path = relative_path.replace('multi-agent-irl', 'multi-agent-particle-envs')
sys.path.append(relative_path)
import multiagent
import time
import torch
import make_env
import numpy as np
from rl.common.torch_util import set_global_seeds
from sandbox.mack.acktr_disc import onehot
from irl.mack.gail_torch import GeneralModel as Model
from sandbox.mack.policies_torch import CategoricalPolicy
from rl import bench
import imageio
import pickle as pkl
from tqdm import tqdm

@click.command()
@click.option('--env', type=click.Choice(['simple', 'simple_speaker_listener',
                                          'simple_crypto', 'simple_push',
                                          'simple_tag', 'simple_spread', 'simple_adversary']), 
                                            default='simple_speaker_listener')
@click.option('--image', is_flag=True, flag_value=True)
def render(env, image):
    
    env_id = env

    def create_env():
        env = make_env.make_env(env_id)
        env.seed(10)
        # env = bench.Monitor(env, '/tmp/',  allow_early_resets=True)
        set_global_seeds(10)
        return env

    env = create_env()
    paths = [
                '/home/arisa/Documents/MGAIL_Original/checkpoints/airl/simple_speaker_listener/decentralized/s-200/l-0.01-d-0.01-b-1000-c-500-iter-1-l2-0.01/seed-1/m_25000',
                '/home/arisa/Documents/MGAIL_Original/checkpoints/airl/simple_speaker_listener/decentralized/s-200/l-0.01-d-0.01-b-1000-c-500-iter-1-l2-0.01/seed-1/m_30000',
                '/home/arisa/Documents/MGAIL_Original/checkpoints/airl/simple_speaker_listener/decentralized/s-200/l-0.01-d-0.01-b-1000-c-500-iter-1-l2-0.01/seed-1/m_35000',
                '/home/arisa/Documents/MGAIL_Original/checkpoints/airl/simple_speaker_listener/decentralized/s-200/l-0.01-d-0.01-b-1000-c-500-iter-1-l2-0.01/seed-1/m_40000',
    ]
    for path in paths:
        test(path, env, env_id, image)
    

def test(path, env, env_id, image):
        print(path)
        n_agents = len(env.action_space)

        ob_space = env.observation_space
        ac_space = env.action_space

        print('observation space')
        print(ob_space)
        print('action space')
        print(ac_space)

        n_actions = [action.n for action in ac_space]
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = 'cpu'
        make_model = lambda: Model(
            CategoricalPolicy, ob_space, ac_space, 1, total_timesteps=1e7, device=device, nprocs=2, nsteps=500,
            nstack=1, ent_coef=0.01, vf_coef=0.5, vf_fisher_coef=1.0, lr=0.01, max_grad_norm=0.5, kfac_clip=0.001,
            lrschedule='linear', identical=make_env.get_identical(env_id))

        model = make_model()
        print("load model from", path)
        model.load(path)

        images = []
        sample_trajs = []
        num_trajs = 500
        max_steps = 50
        avg_ret = [[] for _ in range(n_agents)]

        for i in tqdm(range(num_trajs)):
            all_ob, all_agent_ob, all_ac, all_rew, ep_ret = [], [], [], [], [0 for k in range(n_agents)]
            for k in range(n_agents):
                all_ob.append([])
                all_ac.append([])
                all_rew.append([])
                
            obs = env.reset()
            obs = [ob[None, :] for ob in obs]
            action = [np.zeros([1]) for _ in range(n_agents)]
            step = 0
            done = False
            
            while not done:
                with torch.no_grad():
                    action, _, _ = model.step(obs, action)
                actions_list = [onehot(action[k][0], n_actions[k]) for k in range(n_agents)]
                # actions_list = [onehot(np.random.randint(n_actions[k]), n_actions[k]) for k in range(n_agents)]
                for k in range(n_agents):
                    all_ob[k].append(obs[k])
                    all_ac[k].append(actions_list[k])
                all_agent_ob.append(np.concatenate(obs, axis=1))
                obs, rew, done, _ = env.step(actions_list)
                for k in range(n_agents):
                    all_rew[k].append(rew[k])
                    ep_ret[k] += rew[k]
                obs = [ob[None, :] for ob in obs]
                step += 1
                
                if image and (i % int(num_trajs // 10) == 0):
                    img = env.render(mode='rgb_array')
                    images.append(img[0])
                    time.sleep(0.02)
                    
                if step == max_steps or True in done:
                    done = True
                    step = 0
                else:
                    done = False
                    
            for k in range(n_agents):
                all_ob[k] = np.squeeze(all_ob[k])
            all_agent_ob = np.squeeze(all_agent_ob)
            traj_data = {
                "ob": all_ob, "ac": all_ac, "rew": all_rew,
                "ep_ret": ep_ret, "all_ob": all_agent_ob
            }
            
            sample_trajs.append(traj_data)
            # if i % 1000 == 0:
            #     print(f'traj_num {i} expected_return {ep_ret}')
                
            for k in range(n_agents):
                avg_ret[k].append(ep_ret[k])

        print(path)
        for k in range(n_agents):
            print('agent', k, np.mean(avg_ret[k]), np.std(avg_ret[k]))

        images = np.array(images)
        # pkl.dump(sample_trajs, open(path + '-%dtra.pkl' % num_trajs, 'wb'))
        if image:
            print(images.shape)
            imageio.mimsave(path + '.gif', images, duration=int(len(images[0]) / 20), loop=1)


if __name__ == '__main__':
    render()
