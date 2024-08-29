import sys

import numpy as np
sys.path.append('./')
from simulation.envs.TWLRobot import TWLRobot
 

import gymnasium as gym

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

def make_env(rank: int, seed: int = 0):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the initial seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = TWLRobot("./models/scene.xml", 5, termination_time = 5, time_first_reward = 5)
        env.goal = np.array([2, 0, 0])
        env._healthy_z_range = (0.4, 1)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

if __name__ == "__main__":

    termination_time = 5
    num_cpu = 4
    vec_env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    vec_env_eval = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    obs = vec_env.reset()
    eval_log_dir = "./ppo_eval/"
    model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log="./ppo_train/")

    eval_callback = EvalCallback(vec_env_eval,
                              log_path=eval_log_dir, eval_freq=max(1000 // 4, 1),
                              n_eval_episodes=2, deterministic=True,
                              render=False)
    
    model.learn(total_timesteps=500_000, progress_bar=True, tb_log_name="first_run", callback=eval_callback)
    #obs = vec_env.reset()


    # one_env = TWLRobot("./models/scene.xml", 5, termination_time = 5, time_first_reward = 5)
    # one_env.render_mode = "human"
    # one_env.camera_id = 0
    # one_env.goal = np.array([0.5, 0, 0])
    # one_env._healthy_z_range = (0.4, 1)
    
    # # evaluate_policy(model, one_env, 2, render=True)
    # one_env.reset()
    # for _ in range(1000):
    #     action = one_env.action_space.sample()
    #     for i in range(500):
    #         #print(action)
    #         #action = 0
    #         obs, reward, terminated, truncated, info = one_env.step(action)
    #         #print(terminated)
    #         print(info["reward_position"])
    #         #print(obs.round(3))
    #         if terminated:
    #             one_env.reset()
