import sys
sys.path.append('./')
from simulation.envs.TWLRobot import TWLRobot
 

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_checker import check_env



def make_env(rank: int, seed: int = 0):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the initial seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = TWLRobot("./models/scene.xml", 1, termination_time, 2)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

if __name__ == "__main__":

    termination_time = 5
    num_cpu = 4
    vec_env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])


    model = PPO("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=100)

    obs = vec_env.reset()

    vec_env.render_mode = "human"
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render("human")