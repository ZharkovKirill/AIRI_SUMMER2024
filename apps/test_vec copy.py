import sys

import numpy as np
sys.path.append('./')
from simulation.envs.TWLRobot import TWLRobot, TWLRobotDisc
 

import gymnasium as gym

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

from gymnasium.wrappers import RecordVideo, HumanRendering

def make_env(rank: int, seed: int = 0):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the initial seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = TWLRobotDisc("./models/scene.xml", 5, termination_time = 5, time_first_reward = 5)
        env.goal = np.array([2, 0, 0])
        env._healthy_z_range = (0.4, 1)
        env.reset(seed=seed + rank)
        return env
    #set_random_seed(seed)
    return _init


from gymnasium.wrappers import RecordVideo
if __name__ == "__main__":

    # termination_time = 10
    # num_cpu = 8
    # vec_env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    # vec_env_eval = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    # obs = vec_env.reset()
    # eval_log_dir = "./ppo_eval/"
    # model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log="./ppo_train/", learning_rate=7e-4, n_epochs= 30, seed=83)

    # eval_callback = EvalCallback(vec_env_eval,
    #                           log_path=eval_log_dir, eval_freq=max(1000 // 4, 1),
    #                           n_eval_episodes=2, deterministic=True,
    #                           render=False)
    
    # model.learn(total_timesteps=350_000, progress_bar=True, tb_log_name="ppo_disc_multidisc1_speed", callback=eval_callback, )
    # model.save("../models/ppo_disc_multidisc_speed10")


    model = PPO.load("../models/ppo_disc_multidisc_speed9")
    one_env = TWLRobotDisc("./models/scene.xml", 5, termination_time = 10, time_first_reward = 5)

    one_env.render_mode = "human"
    one_env.camera_id = 0
    one_env.speeds = [0.0, 0.3]

    obs, info = one_env.reset(1)
    for i in range(2000):
        action, _states = model.predict(obs)
        # action = [4, 4, 4, 4, 0, 0]
        
        obs, rewards, terminated, truncated, info = one_env.step(action)


