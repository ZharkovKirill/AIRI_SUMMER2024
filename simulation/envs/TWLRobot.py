
from copy import copy
from math import exp
import pathlib
import sys
import os

sys.path.append("./")
sys.path.append("./../")
sys.path.append("./../../")

from dataclasses import dataclass, field

from enum import Enum
from typing import Optional, Union
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete, MultiDiscrete
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv

from jax import numpy as jp
import mujoco
from mujoco import mjx
from brax.envs.base import Env, PipelineEnv, State
from brax.mjx.base import State as MjxState

from scipy.spatial.transform import Rotation as R

# from dm_control.utils import containers

# SUITE = containers.TaggedTasks()

from simulation.context import RobotContext

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 1.0)),
    "elevation": -20.0,
}

SEED = 0xEBAB


def standardize_vector(vector, mean, std):
    return (vector - mean) / std

def exp_func(x):
    return np.exp(-np.linalg.norm(x)**2)
@dataclass
class ControlConfig:
    stiffness: np.ndarray = field(default_factory=np.ndarray)#np.array([80, 80, 80, 80, 20, 20], dtype=np.float64)
    damping: np.ndarray = field(default_factory=np.ndarray)#np.array([2, 2, 2, 2, 0.5, 0.5], dtype=np.float64)
    default_position: np.ndarray = field(default_factory=np.ndarray)#np.array(
    #     [
    #         -0.7853981633974483,
    #         1.5707963267948966,
    #         0.7853981633974483,
    #         -1.5707963267948966,
    #         0.0,
    #         0.0,
    #     ]
    # )
    # action_scale: tuple = field(default_factory=tuple)#(0.5, 0.5, 0.5, 0.5, 0.2, 0.2)
    # control_types: tuple = field(default_factory=tuple) #("P", "P", "P", "P", "V", "V")
    action_scale: tuple[float] = (0.5, 0.5, 0.5, 0.5, 0.01, 0.01)
    control_types: tuple[str] = ("P", "P", "P", "P", "V", "V")
    scale: tuple[float] = (-1.0, -1.0, 1.0, 1.0, -1.0, 1.0)
    def __init__(self,
                # stiffness = np.array([80, 80, 80, 80, 1, 1], dtype=np.float64),
                stiffness = np.array([80, 80, 80, 80, 0.01, 0.01], dtype=np.float64),
                damping: np.ndarray = np.array([6, 6, 6, 6, 0.02, 0.02], dtype=np.float64),
                # damping: np.ndarray = np.zeros(6),
                default_position: np.ndarray = np.array(
                    [
                        -0.7853981633974483,
                        1.5707963267948966,
                        0.7853981633974483,
                        -1.5707963267948966,
                        0.0,
                        0.0,
                    ]
                ), 
                action_scale: tuple[float] = (0.5, 0.5, 0.5, 0.5, 0.1, 0.1),
                control_types: tuple[str] = ("P", "P", "P", "P", "V", "V")):
        self.stiffness = stiffness
        self.damping = damping
        self.default_position = default_position
        self.action_scale = action_scale
        self.control_types = control_types


@dataclass
class ControlDiscConfig:
    stiffness: np.ndarray = field(default_factory=np.ndarray)#np.array([80, 80, 80, 80, 20, 20], dtype=np.float64)
    damping: np.ndarray = field(default_factory=np.ndarray)#np.array([2, 2, 2, 2, 0.5, 0.5], dtype=np.float64)
    default_position: np.ndarray = field(default_factory=np.ndarray)#np.array(
    action_scale: tuple[float] = (0.5, 0.5, 0.5, 0.5, 1, 1)
    control_types: tuple[str] = ("P", "P", "P", "P", "V", "V")
    scale: tuple[float] = (-1.0, -1.0, 1.0, 1.0, -1.0, 1.0)

    def __init__(self,
                # stiffness = np.array([80, 80, 80, 80, 1, 1], dtype=np.float64),
                stiffness = np.array([80, 80, 80, 80, 0.01, 0.01], dtype=np.float64),
                damping: np.ndarray = np.array([6, 6, 6, 6, 0.02, 0.02], dtype=np.float64),
                # damping: np.ndarray = np.zeros(6),
                default_position: np.ndarray = np.array(
                    [
                        -0.7853981633974483,
                        1.5707963267948966,
                        0.7853981633974483,
                        -1.5707963267948966,
                        0.0,
                        0.0,
                    ]
                ), 
                # action_scale: tuple[float] = (0.5, 0.5, 0.5, 0.5, 0.01, 0.01),
                action_scale: tuple[float] = (1, 1, 1, 1, 1, 1),
                control_types: tuple[str] = ("P", "P", "P", "P", "V", "V")):
        self.stiffness = stiffness
        self.damping = damping
        self.default_position = default_position
        self.action_scale = action_scale
        self.control_types = control_types


INIT_POS = np.array(
    [  # Chassis
        0.0,  # x
        0.0,  # y
        0.6481854249492381,  # z
        # 2,
        # quat
        1.0,
        0.0,
        0.0,
        0.0,
        # right upper j
        0.7853981633974483,
        # right lower j
        -1.5707963267948966,
        # right wheel
        0.0,
        # left upper j
        -0.7853981633974483,
        # left lower j
        1.5707963267948966,
        # left wheel
        0.0,
    ]
)


class BodyNames(Enum):
    chassis = "body_chassis_"
    right_upper_link = "right_upper_link/body_link_right_upper"
    right_lower_link = "right_upper_link/right_down_link/body_link_right_down"
    left_upper_link = "left_upper_link/body_link_left_upper"
    left_lower_link = "left_upper_link/left_down_link/body_link_left_down"
    right_wheel = "right_upper_link/right_down_link/right_tire/body_tire_right"
    left_wheel = "left_upper_link/left_down_link/left_tire/body_tire_left"


class JointNames(Enum):
    """All joint connetcts from chassis to wheel

    Args:
        Enum (_type_): _description_
    """

    free_floating_chassis = "free_floating_chassis_"
    right_upper_joint = "right_upper_link/joint_link_right_upper"
    right_lower_joint = "right_upper_link/right_down_link/joint_link_right_down"
    left_upper_joint = "left_upper_link/joint_link_left_upper"
    left_lower_joint = "left_upper_link/left_down_link/joint_link_left_down"
    right_wheel = "right_upper_link/right_down_link/right_tire/joint_tire_right"
    left_wheel = "left_upper_link/left_down_link/left_tire/joint_tire_left"


class TWLRobot(MujocoEnv):
    path_to_model: str
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(
        self,
        path_to_model: str,
        frame_skip: int,
        termination_time: float,
        time_first_reward: float,
        control_cfg: ControlConfig = ControlConfig(),
        ctrl_cost_weight=1,
        contact_cost_weight=1,
        contact_cost_range=(-1, 1),
        healthy_z_range=(0, 3.0),
        healthu_pitch_range=(-np.pi / 2, np.pi / 2),
        obs_noise = 0.05,
        seed = SEED,
        default_camera_config: dict[str, dict[float, int]] = DEFAULT_CAMERA_CONFIG,
        **kwargs
    ):
        """
        Initializes the TWLRobot environment.
        Args:
            path_to_model (str): The path to the robot model.
            frame_skip (int): The number of simulation steps to skip between each action.
            termination_time (float): The maximum time allowed for each episode.
            time_first_reward (float): The time at which the first reward is given.
            ctrl_cost_weight (float, optional): The weight of the control cost term in the reward function. Defaults to 1.
            contact_cost_weight (float, optional): The weight of the contact cost term in the reward function. Defaults to 1.
            contact_cost_range (tuple, optional): The range of contact cost values. Defaults to (-1, 1).
            healthy_z_range (tuple, optional): The range of healthy z values. Defaults to (0, 1.0).
            healthu_pitch_range (tuple, optional): The range of healthy pitch values. Defaults to (-np.pi / 2, np.pi / 2).
            default_camera_config (dict, optional): The default camera configuration. Defaults to DEFAULT_CAMERA_CONFIG.
            **kwargs: Additional keyword arguments.
        Raises:
            None
        Returns:
            None
        """
        super().__init__(
            path_to_model,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            **kwargs,
        )
        
        self.data.qpos[:] = copy(INIT_POS)
        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }
        self.frame_skip = frame_skip
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight
        self._contact_cost_range = contact_cost_range

        self._healthy_z_range = healthy_z_range
        self._healthy_pitch_range = healthu_pitch_range

        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                4  # qpos
                + 6  # qvel
                + 3  # ang_vel_in_base
                + 2  # direction_to_goal
                + 1  # height_command
                + 3  # gravity projection
                + 6,  # last_action
            ),
            dtype=np.float32,
        )

        self.observation_structure = {
            "qpos": 4,
            "qvel": 6,
            "ang_vel_in_base": 3,
            "direction_to_goal": 2,
            "height_command": 1,
            "gravity_projection": 1,
            "last_action": 6,
        }
        self.action_space = Box(
            low=np.array([-np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2, -50, -50]),
            high=np.array([np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2, 50, 50]),
            shape=(6,),
            dtype=np.float64,
        )

        self.action_structure = {
            "left_configuration": 2,
            "right_configuration": 2,
            "vel_left_wheel": 1,
            "vel_right_wheel": 1,
        }

        # self.robot_context = RobotContext("./models/robot.xml")
        path = pathlib.PurePath(self.fullpath)
        path_to_robot = path.parts[:-1] + ("robot.xml",)
        robot_path = pathlib.PurePath(*path_to_robot)
        self.robot_context = RobotContext(str(robot_path))

        self.goal = np.zeros(3)
        self.goal_orientation = R.identity().as_matrix()
        
        self.last_actions = np.zeros(6)
        self.last_qvel = np.zeros(6)
        
        self.termination_time = termination_time
        self.time_first_reward = time_first_reward
        self.control_cfg = control_cfg
        
        self._obs_noise = obs_noise
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.goal  = np.array([1, 0, 0])
        self.steps_done = 0 

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def control_cost(self):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(self.data.ctrl))
        return control_cost

    @property
    def contact_cost(self):
        contact_forces = self.data.cfrc_ext
        contact_cost = self._contact_cost_weight * np.sum(np.square(contact_forces))
        min_cost, max_cost = self._contact_cost_range
        contact_cost = np.clip(contact_cost, min_cost, max_cost)
        return contact_cost

    @property
    def is_healthy(self):
        """
        Check if the robot is in a healthy state.
        Returns:
            bool: True if the robot is healthy, False otherwise.
        """

        min_z, max_z = self._healthy_z_range
        min_pitch, max_pitch = self._healthy_pitch_range
        z_health = (
            min_z
            <= self.data.xpos[self.model.body(BodyNames.chassis.value).id][2]
            <= max_z
        )
        #print(self.data.xpos[self.model.body(BodyNames.chassis.value).id][2])
        pitch = R.from_quat(self.data.xquat[self.model.body(BodyNames.chassis.value).id]).as_euler("xyz")[1]
        pitch_health = min_pitch <= pitch <= max_pitch
        return pitch_health and z_health

    def _get_obs(self):
        """
        Returns the observation vector for the environment.

        Returns:
            np.ndarray: The observation vector, which includes the joint positions and velocities,
            angular velocity in the base, direction to the goal, height command, projected gravity,
            and last actions.
        """
        # Rest of the code...

        left_upper_qpos = self.data.joint(JointNames.left_upper_joint.value).qpos
        left_upper_qvel = self.data.joint(JointNames.left_upper_joint.value).qvel
        left_lower_qpos = self.data.joint(JointNames.left_lower_joint.value).qpos
        left_lower_qvel = self.data.joint(JointNames.left_lower_joint.value).qvel

        right_upper_qpos = self.data.joint(JointNames.right_upper_joint.value).qpos
        right_upper_qvel = self.data.joint(JointNames.right_upper_joint.value).qvel
        right_lower_qpos = self.data.joint(JointNames.right_lower_joint.value).qpos
        right_lower_qvel = self.data.joint(JointNames.right_lower_joint.value).qvel

        r_wheel_qvel = self.data.joint(JointNames.right_wheel.value).qvel
        l_wheel_qvel = self.data.joint(JointNames.left_wheel.value).qvel

        qpos = np.hstack(
            (left_upper_qpos, left_lower_qpos, right_upper_qpos, right_lower_qpos)
        )
        qvel = np.hstack(
            (
                left_upper_qvel,
                left_lower_qvel,
                right_upper_qvel,
                right_lower_qvel,
                l_wheel_qvel,
                r_wheel_qvel,
            )
        )

        ang_vel_in_base = self.data.sensor("_gyro").data

        xy_pos_chassis = self.data.xpos[self.model.body(BodyNames.chassis.value).id][:2]
        direction_to_goal = self.goal[:2] - xy_pos_chassis
        direction_to_goal /= np.linalg.norm(direction_to_goal)

        height_command = self.goal[2]

        gravity_ground = np.array([0, 0, 1])

        R_wb = R.from_quat(
            self.data.xquat[self.model.body(BodyNames.chassis.value).id]
        ).as_matrix()

        projected_gravity = R_wb.T @ gravity_ground
        obs_vec = np.concatenate(
            (
                qpos,
                qvel,
                ang_vel_in_base,
                direction_to_goal,
                np.array([height_command]),
                projected_gravity,
                self.last_actions,
            )
        )
        obs_vec += self.rng.normal(0,self._obs_noise, obs_vec.size)

        return np.array(obs_vec, dtype=np.float32)

    def _get_priveleged_info(self):
        """
        Retrieves privileged information about the robot's state and environment.
        Returns:
            tuple: A tuple containing two elements:
                - numpy.ndarray: An array containing the following information:
                    - linear_velocity (numpy.ndarray): The robot's linear velocity in the local coordinate system.
                    - base_height (float): The height of the robot's chassis.
                    - relative_position_goal (numpy.ndarray): The desired position relative to the robot's position in the local coordinate system.
                    - wheels_contact_force (numpy.ndarray): The contact force with the wheels.
                    - friction_coefficient (float): The coefficient of friction.
                - dict: A dictionary containing the following information:
                    - linear_velocity (numpy.ndarray): The robot's linear velocity in the local coordinate system.
                    - base_height (float): The height of the robot's chassis.
                    - relative_position_goal (numpy.ndarray): The desired position relative to the robot's position in the local coordinate system.
                    - wheels_contact_force (numpy.ndarray): The contact force with the wheels.
                    - friction_coefficient (float): The coefficient of friction.
        """

        # WARN: Info don't have info about terrain height

        # Должно выдавать скорость в локальной с.к.
        ff_chassis_joint_id = self.model.joint(
            JointNames.free_floating_chassis.value
        ).id
        robot_vel = self.data.qvel[ff_chassis_joint_id : ff_chassis_joint_id + 3]

        # Позиция
        robot_pos = self.data.xpos[self.model.body(BodyNames.chassis.value).id]
        base_height = robot_pos[2]

        # Таргет в локальной с.к.
        rel_goal_pos = self.goal - robot_pos

        # geom_id2forces = {id: [] for id in self.data.contact.geom2}
        # for i in range(self.data.ncon):
        #     m = self.data.contact.geom2[i]
        #     # f = np.zeros(6)
        #     # mujoco.mj_contactForce(self.model, self.data, i, f)
        #     geom_id2forces[m].append(f[:3])
        # f_wheels = [np.mean(fs, axis=0) for id, fs in geom_id2forces.items()]

        # Сила контака с колесами
        l_whl_frc = self.data.cfrc_ext[self.model.body(BodyNames.left_wheel.value).id]
        r_whl_frc = self.data.cfrc_ext[self.model.body(BodyNames.right_wheel.value).id]

        f_wheels = np.concatenate((l_whl_frc[3:], r_whl_frc[3:]))
        # Коэффициент трения
        frc_coeff = 0.95#self.data.contact.mu[0]

        priveleged_info = {
            "linear_velocity": robot_vel,  # Скорость в локальной с.к.
            "base_height": base_height,  # Высота корпуса
            "relative_position_goal": rel_goal_pos,  # Желаемая позиция в л.с.к.
            "wheels_contact_force": f_wheels,  # Контактная сила колес
            "friction_coefficient": frc_coeff,  # Коэффициент трения
        }

        return (
            np.concatenate(
                (
                    robot_vel,
                    np.array([base_height]),
                    rel_goal_pos,
                    f_wheels,
                    np.array([frc_coeff]),
                )
            ),
            priveleged_info,
        )

    def step(self, action):
        self.steps_done += 1
        torques = self._compute_torques(action)
        left_conf = torques[:2]
        right_conf = torques[2:4]
        dr_wheel = torques[4]
        dl_wheel = torques[5]

        torques = self.robot_context.get_control_vector(
            left_conf, dl_wheel, right_conf, dr_wheel
        )
        self.do_simulation(torques, self.frame_skip)
        # self.data.qpos[:7] = INIT_POS[:7]
        # self.data.qvel[:6] = np.zeros(6)
        # self.data.qacc[:6] = np.zeros(6)
        # self.data.xpos[1] = INIT_POS[:3]
        # self.data.xquat[1] = INIT_POS[3:7]
        

        observation = self._get_obs()
        # self.last_qvel = observation[4:10]
        self.last_qvel = np.zeros(6)

        reward, reward_info = self.calculate_reward()

        priveleged_state, priveleged_info = self._get_priveleged_info()
        info = {
            # "qpos": observation[0:4],
            # "qvel": observation[4:10],
            # "ang_vel_in_base": observation[10:13],
            # "direction_to_goal":observation[13:15],
            # "height_command": observation[15],
            # "projected_gravity": observation[15:19],
            # "last_actions": observation[19:],
            **priveleged_info,
            **reward_info,
        }

        time_cond = self.termination_time >= self.data.time
        terminated = not time_cond or not self.is_healthy

        if terminated:
            self.reset()
        
        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info

    def calculate_reward(self):
        T_r = self.time_first_reward

        robot_pos = self.data.xpos[self.model.body(BodyNames.chassis.value).id]

        ff_chassis_joint_id = self.model.joint(
            JointNames.free_floating_chassis.value
        ).id

        robot_vel = self.data.qvel[ff_chassis_joint_id : ff_chassis_joint_id + 3]

        robot_quat = self.data.xquat[self.model.body(BodyNames.chassis.value).id]
        robot_orient = R.from_quat(robot_quat).as_matrix()

        weights = np.array([10, 1, 1, 0.1])

        err = robot_pos - self.goal

        if self.data.time > self.termination_time - self.time_first_reward:
            r_pos = 1 / T_r * 1 / (1 + np.linalg.norm(err))
        else:
            r_pos = 0
            
        if np.linalg.norm(robot_vel) < 0.001:
            r_posbias = 0
        else:
            r_posbias = (
                np.inner(robot_vel, -err) / np.linalg.norm(robot_vel) / np.linalg.norm(-err)
            )

        if np.linalg.norm(robot_vel) < 0.1 and np.linalg.norm(err) > 0.5:
            r_stall = -1
        else:
            r_stall = 0

        if np.linalg.norm(err) > 0.5:
            r_face_goal = -np.trace(robot_orient.T @ self.goal_orientation)
        else:
            r_face_goal = 0

        rew = np.array([r_pos, r_posbias, r_stall, r_face_goal])

        rew_info = {
            "reward_position": r_pos,
            "reward_position_bias": r_posbias,
            "reward_stall": r_stall,
            "reward_face_goal": r_face_goal,
        }

        return np.sum(np.inner(weights, rew)), rew_info

    def _compute_torques(self, action):
        actions_scaled = action * np.array(self.control_cfg.action_scale)
        # obs = self._get_obs()
        
        # qpos = obs[:4]
        # qvel = obs[4:10]
        
        # qpos = np.hstack((qpos, [0,0]))
        qpos = np.zeros(6)
        qvel = np.zeros(6)
        
        torques = []
        for type_act, a, curr_pos, curr_vel, last_vel, defaul_pos, stiff, damp, scl in zip(
            self.control_cfg.control_types,
            actions_scaled,
            qpos,
            qvel,
            self.last_qvel,
            self.control_cfg.default_position,
            self.control_cfg.stiffness,
            self.control_cfg.damping,
            self.control_cfg.scale
        ):
            if type_act == "P":
                # torques.append((
                #     stiff
                #     * (scl*a + defaul_pos - curr_pos)
                #     - damp * curr_vel
                # ))
                torques.append(scl*a+ defaul_pos)
            elif type_act == "V":
                # torques.append((
                #     stiff * (scl*a - curr_vel)
                #     - damp
                #     * (curr_vel - last_vel)
                #     / self.dt
                # ))
                torques.append(scl*a)
            elif type_act == "T":
                torques.append(scl*actions_scaled)
        torques = np.clip(torques, -30*np.ones_like(torques), 30*np.ones_like(torques))
        return np.array(torques)

    def sample_goal(self):
        bound_pos_x = [0.5, 1.0]
        bound_pos_y = [-0.1, 0.1]
        bound_pos_z = [-0.1, 0.1]
        orienration_goal_z = [-0.3, 0.3]
        
        x_goal = self.rng.uniform(bound_pos_x[0], bound_pos_x[1])
        y_goal = self.rng.uniform(bound_pos_y[0], bound_pos_y[1])
        ort_goal_z = self.rng.uniform(orienration_goal_z[0], orienration_goal_z[1])
        
        height = 0.6481854249492381 + self.rng.uniform(bound_pos_z[0], bound_pos_z[1])
        
        self.goal_orientation = R.from_euler("xyz", [0, 0, ort_goal_z]).as_matrix()
        self.goal = np.array([x_goal, y_goal, height])
        
    def reset(self, seed=0xEBAB):
        self.steps_done = 0
        self._reset_simulation()
        qpos = INIT_POS
        # qpos[:2] += self.rng.uniform(-0.1, 0.1, size = 2)
        qvel = np.zeros(self.model.nv)
        # qvel[:2] = self.rng.normal(qvel[:2], 0.1, size=2)
        # qvel[6:8] = self.rng.normal(qvel[6:8], 0.01, size=2)
        # qvel[8] = self.rng.normal(qvel[8], 0.05)
        # qvel[9:11] = self.rng.normal(qvel[9:11], 0.01, size=2)
        # qvel[11] = self.rng.normal(qvel[11], 0.05)

        self.set_state(qpos, qvel)

        obs = self._get_obs()
        info = {"q_pos": qpos, 'q_vel': qvel}
        return (obs, info)


class TWLRobotDisc(TWLRobot):
    regulaze = (np.array([-0.08538231, -0.16970922,  0.14940762]),
    np.array([0.30912793, 0.4157188 , 2.33317212]))
    path_to_model: str
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(self,
        path_to_model: str,
        frame_skip: int,
        termination_time: float,
        time_first_reward: float,
        control_cfg: ControlDiscConfig = ControlDiscConfig(),
        ctrl_cost_weight=1,
        contact_cost_weight=1,
        contact_cost_range=(-1, 1),
        healthy_z_range=(0.35, 1.5),
        healthu_pitch_range=(-np.pi / 3, np.pi / 3),
        obs_noise = 0.05,
        seed = SEED,
        default_camera_config: dict[str, dict[float, int]] = DEFAULT_CAMERA_CONFIG,
        **kwargs):
        
        super().__init__(
            path_to_model,
            frame_skip,
            termination_time,
            time_first_reward,
            control_cfg,
            ctrl_cost_weight,
            contact_cost_weight,
            contact_cost_range,
            healthy_z_range,
            healthu_pitch_range,
            obs_noise,
            seed,
            default_camera_config,
        )
        
        self.observation_space = Box(
            low = np.array([-5, -5, -5, -5, -5, -5, -5]),
            high = np.array([5, 5, 5, 5, 5, 5, 5]),
            shape = (7,))
        # self.action_space = Box(
        #     low=np.array([-1, -1]),
        #     high=np.array([1, 1]),
        #     shape=(2,),
        #     dtype=np.float64,
        # )
        scl = 0.017453
        self.action_space = MultiDiscrete([17, 17, 17, 17, 5, 5])
        legs_state = {0: -15 * scl, 
                      1: -13 * scl, 
                      2: -11 * scl, 
                      3: -9 * scl, 
                      4: -7 * scl, 
                      5: -5 * scl, 
                      6: -3 * scl, 
                      7: -1 * scl, 
                      8: 0, 
                      9: 1 * scl, 
                      10: 3 * scl, 
                      11: 5 * scl, 
                      12: 7 * scl, 
                      13: 9 * scl, 
                      14: 11 * scl, 
                      15: 13 * scl, 
                      16: 15 * scl}
        self.look_table = [ 
                           legs_state, 
                           legs_state, 
                           legs_state, 
                           legs_state, 
                           {0:-5, 1:-1, 2:0, 3:1, 4:5}, 
                           {0:-5, 1:-1, 2:0, 3:1, 4:5}
                           ]
        self.last_action = np.zeros(6)

        self._x_vel_bound = np.arange(-0.5, 0.5, 0.1)
        #self._y_vel_bound = [-0.5, 0.5]
        self._yaw_vel_bound = np.arange(-0.5, 0.5, 0.1)
        #self.speeds = self._sample_speed()
        self.speeds = [0, 0]
        self.rng = np.random.default_rng(seed)

    
    def _sample_speed(self):
        x_vel = self.rng.uniform(-0.5, 0.5)
        #y_vel = self.rng.uniform(self._y_vel_bound[0], self._y_vel_bound[1])
        yaw_vel = self.rng.uniform(-0.5, 0.5)
        # print(x_vel, yaw_vel)
        return [x_vel, yaw_vel]
        
    
    def calculate_reward(self):
        des_height = 0.55 #INIT_POS[2] + 0.3
        height = self.data.qpos[2]
        joints_states = self.data.qpos[6:10]
        
        obs = self._get_obs()
        dx = obs[0]
        dyaw = obs[2]
        
        rew_1 = exp_func(height - des_height)
        rew_2 = exp_func(self.speeds[0] - dx)
        rew_3 = 0#exp_func(self.speeds[1] - dy)
        rew_4 = exp_func(self.speeds[1] - dyaw)
        rew_5 = exp_func(np.zeros(4)-joints_states)
        rew_6 = exp_func(min(self.steps_done-1000, 0))
        
        weights = np.array([1, 1.5, 1.5, 1.5, 1, 1])
        
        return np.inner(weights, [rew_1, rew_2, rew_3, rew_4, rew_5, rew_6])/2000, {"height": height} #np.inner(weights, [rew_1, rew_2]), {"height": rew_1, "x_vel": rew_2}
    
    def _get_obs(self):
        angles = R.from_quat(self.data.qpos[3:7]).as_euler("xyz")

        pitch = angles[1]
        yaw = angles[0]
        
        d_pitch = self.data.sensor("_gyro").data[1]
        d_yaw = self.data.sensor("_gyro").data[2]
        
        
        x_vel = self.data.qvel[0]
        y_vel = self.data.qvel[1]

        x_vel = np.linalg.norm([x_vel, y_vel])*np.sign(np.arctan2(y_vel, x_vel)*yaw)

        obs_vec = np.array([x_vel, yaw, d_yaw, pitch, d_pitch, self.speeds[0], self.speeds[1]])
        #print(obs_vec)
        #reg_obs_vec = standardize_vector(obs_vec, self.regulaze[0], self.regulaze[1])
        return obs_vec
    
    def _compute_torques(self, action):
        full_action = np.zeros(6)
        #print(action)
        #action = int(action)
        #scl=10
        # wheels
        
        
        # legs
        # full_action[0] = self.last_action[0] + self.look_table[2][int(action[2])]
        # full_action[1] = self.last_action[1] + self.look_table[3][int(action[3])]
        # full_action[2] = self.last_action[2] + self.look_table[4][int(action[4])]
        # full_action[3] = self.last_action[3] + self.look_table[5][int(action[5])]

        full_action[0] = self.look_table[0][int(action[0])]
        full_action[1] = self.look_table[1][int(action[1])]
        full_action[2] = self.look_table[2][int(action[2])]
        full_action[3] = self.look_table[3][int(action[3])]
        full_action[5] = self.look_table[4][int(action[4])]
        full_action[4] = self.look_table[5][int(action[5])]
        self.last_action = full_action
        # print(full_action)
        return super()._compute_torques(full_action)

    def reset(self, seed = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        # self.speeds = self._sample_speed()
        #print(self.speeds)
        self.steps_done = 0
        self._reset_simulation()
        qpos = INIT_POS
        qpos[:2] = self.rng.normal(qpos[:2], 0.02, size=2)
        qpos[7:] = self.rng.normal(qpos[7:], 0.02, size=6)
        qvel = np.zeros(self.model.nv)
        qvel[:2] = self.rng.normal(qvel[:2], 0.01, size=2)
        qvel[6:8] = self.rng.normal(qvel[6:8], 0.01, size=2)
        qvel[8] = self.rng.normal(qvel[8], 0.05)
        qvel[9:11] = self.rng.normal(qvel[9:11], 0.01, size=2)
        qvel[11] = self.rng.normal(qvel[11], 0.05)

        self.set_state(qpos, qvel)
        #print(qpos, qvel)

        obs = self._get_obs()
        info = {"q_pos": qpos, 'q_vel': qvel}
        return (obs, info)
        

    @property
    def is_healthy(self):
        """
        Check if the robot is in a healthy state.
        Returns:
            bool: True if the robot is healthy, False otherwise.
        """

        min_z, max_z = self._healthy_z_range
        min_pitch, max_pitch = self._healthy_pitch_range
        z_health = (
            min_z
            <= self.data.xpos[self.model.body(BodyNames.chassis.value).id][2]
            <= max_z
        )
        #print(self.data.xpos[self.model.body(BodyNames.chassis.value).id][2])
        pitch = R.from_quat(self.data.xquat[self.model.body(BodyNames.chassis.value).id]).as_euler("xyz")[1]
        pitch_health = min_pitch <= pitch <= max_pitch
        pose = np.abs(self.data.qpos[0])<5
        return pitch_health and z_health and pose

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    # termination_time = 5
    # env = TWLRobot("./models/scene.xml", 1, termination_time, 2)
    # env.render_mode = "human"
    # env.goal = np.array([1, 0, 0])
    # env.reset()
    # terminated = False
    # while not terminated:

    #     act = np.ones(6)*0.1
    #     act[-2:] = np.zeros(2)
    #     __,__, terminated, __, info = env.step(act)
        
    #     qpos = info["qpos"]
    #     qvel = info["qvel"]

    termination_time = 5
    env = TWLRobotDisc("./models/scene.xml", 1, termination_time, 2, obs_noise=0)
    env.render_mode = "human"
    env.reset()
    env.sample_goal()
    
    for i in range(10):
        a = env.action_space.sample()
        print(a, env._compute_torques(a))
    terminated = False
    while not terminated:
        act = env.action_space.sample()
        # act[-2:] = np.zeros(2)
        __,__, terminated, __, info = env.step(act)
        
        qpos = info["qpos"]
        qvel = info["qvel"]
    # test sampling
    # qvel = []
    # qpos = []
    # goal = []
    # for i in range(1000):
    #     env.reset()
    #     qpos.append(copy(env.data.qpos))
    #     qvel.append(copy(env.data.qvel))
    #     env.sample_goal()
    #     goal.append(env.goal)
    
    # qpos = np.array(qpos)
    # qvel = np.array(qvel)
    # goal = np.array(goal)
    
    # for q in qvel.T:
    #     plt.plot(np.arange(q.size),q)
    #     plt.show()
    
    # Test PD-controller

#     # Initialize the plot
#     fig, ax = plt.subplots(3,2, figsize=(13,8))
#     ax[0,0].set_xlim(0, termination_time)
#     ax[0,0].set_ylim(-1, 1)
#     ax[0,0].set_ylabel('pos r up, m')
    
#     ax[0,1].set_xlim(0, termination_time)
#     ax[0,1].set_ylim(-1, 1)
#     ax[0,1].set_ylabel('pos l up, m')
    
#     ax[1,0].set_xlim(0, termination_time)
#     ax[1,0].set_ylim(-1, 1)
#     ax[1,0].set_ylabel('pos r low, m')
#     ax[1,1].set_xlim(0, termination_time)
#     ax[1,1].set_ylim(-1, 1)
#     ax[1,1].set_ylabel('pos l low, m')
#     ax[2,0].set_xlim(0, termination_time)
#     ax[2,0].set_ylim(-1, 1)
#     ax[2,0].set_ylabel('pos r wheel, m')
#     ax[2,1].set_xlim(0, termination_time)
#     ax[2,1].set_ylim(-1, 1)
#     ax[2,1].set_ylabel('pos l wheel, m')
#     line_r_u, = ax[0,0].plot([], [], lw=2)
#     line_r_l, = ax[1,0].plot([], [], lw=2)
#     line_r_w, = ax[2,0].plot([], [], lw=2)
#     line_l_u, = ax[0,1].plot([], [], lw=2)
#     line_l_l, = ax[1,1].plot([], [], lw=2)
#     line_l_w, = ax[2,1].plot([], [], lw=2)
    
#     # Lists to store the data
#     xs = []
#     ys = []

#     # Function to generate new data (replace with your actual data source)
#     def generate_data():
#         global terminated
#         # Simulate receiving data in a loop
#         for i in range(5000):
#             act = np.ones(6)*0.1
#             act[-2:] = np.ones(2)*0.2
#             __,__, terminated, __, info = env.step(act)
            
#             qpos = copy(info["qpos"])
#             qvel = copy(info["qvel"])
#             time = copy(env.data.time)
#             data = {"time":time, "qpos":qpos, "qvel":qvel}
            
#             yield data
            
# # Animation function
#     def animate(i):
#         global xs, ys  # Access global lists
        
#         # Generate new data point
#         new_data = next(data_generator)
        
#         # Append new data to lists
#         xs.append(new_data["time"])
#         ys.append(np.hstack((new_data["qpos"], new_data["qvel"][-2:])))
        
#         # Limit the lists to 100 items to avoid growing indefinitely
#         # if len(xs) > 100:
#         #     xs.pop(0)
#         #     ys.pop(0)
        
#         np_ys = np.array(ys)
#         # Update the plot
#         line_l_u.set_data(xs, np_ys[:,0])
#         line_l_l.set_data(xs, np_ys[:,1])
#         line_r_u.set_data(xs, np_ys[:,2])
#         line_r_l.set_data(xs, np_ys[:,3])
#         line_l_w.set_data(xs, np_ys[:,4])
#         line_r_w.set_data(xs, np_ys[:,5])
#         ax[0,0].set_xlim(min(xs), max(xs))
#         ax[0,1].set_xlim(min(xs), max(xs))
#         ax[1,0].set_xlim(min(xs), max(xs))
#         ax[1,1].set_xlim(min(xs), max(xs))
#         ax[2,0].set_xlim(min(xs), max(xs))
#         ax[2,1].set_xlim(min(xs), max(xs))

#         ax[0,0].set_ylim(min(np_ys[:,2]) - 0.1, max(np_ys[:,2]) + 0.1)
#         ax[0,1].set_ylim(min(np_ys[:,0]) - 0.1, max(np_ys[:,0]) + 0.1)
#         ax[1,0].set_ylim(min(np_ys[:,3]) - 0.1, max(np_ys[:,3]) + 0.1)
#         ax[1,1].set_ylim(min(np_ys[:,1]) - 0.1, max(np_ys[:,1]) + 0.1)
#         ax[2,0].set_ylim(min(np_ys[:,5]) - 0.1, max(np_ys[:,5]) + 0.1)
#         ax[2,1].set_ylim(min(np_ys[:,4]) - 0.1, max(np_ys[:,4]) + 0.1)
        
#         return line_l_u, line_l_l, line_r_u, line_r_l, line_l_w, line_r_w

#     # Create a generator for the data
#     data_generator = generate_data()

#     # Create the animation
#     ani = animation.FuncAnimation(fig, animate, interval=100, blit=True)

#     # Display the animation
#     plt.show()
