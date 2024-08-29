import sys
import time

import numpy as np
sys.path.append('./')
from simulation.envs.TWLRobot import TWLRReduccedObsDiscete, TWLRobot, TWLRReduccedObs
 

import gymnasium as gym

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy


if __name__ == "__main__":

    one_env = TWLRReduccedObsDiscete("./models/scene.xml", 5, termination_time = 5, time_first_reward = 5)
    one_env.render_mode = "human"
    one_env.camera_id = 0
    one_env.goal = np.array([1.0, 0, 0.5])
    one_env._healthy_z_range = (0.3, 1.5)
    all_obbs = []
    check_env(one_env)
    one_env.reset()
    for _ in range(1000):
        action = one_env.action_space.sample()
        for i in range(50):
            #print(action)
            #time.sleep(0.3)
            # action[0] = 0
            # action[1] = 0
            # action[2] = 0
            obs, reward, terminated, truncated, info = one_env.step(action)
            #print(terminated)
            print(reward)
            #print(obs.round(3))
            if terminated:
                one_env.reset()
            all_obbs.append(obs)
    