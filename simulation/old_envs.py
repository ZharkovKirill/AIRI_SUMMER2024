from dataclasses import dataclass, field
import os
import time

from scipy.spatial.transform import Rotation as R
from lxml import etree
import mujoco as mj
import mujoco.viewer
import numpy as np
from sympy import pretty, pretty_print

from simulation.context import RobotContext
from utils.utils import dotdict


@dataclass
class IMU:
    gyro: np.ndarray = field(default_factory=lambda: np.zeros(3))
    accel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    mag: np.ndarray = field(default_factory=lambda: np.zeros(3))
    orientation: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0]))

@dataclass
class SensorData:
    time: float = 0.0
    left_leg: dotdict = field(default_factory=lambda: dotdict())
    right_leg: dotdict = field(default_factory=lambda: dotdict())
    chassis_imu: IMU = field(default_factory=lambda: IMU())

@dataclass
class State:
    time: float = 0.0
    q_left: np.ndarray = field(default_factory=lambda: np.zeros(3))
    q_dot_left: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    q_right: np.ndarray = field(default_factory=lambda: np.zeros(3))
    q_dot_right: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    chassis_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    chassis_quat: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0]))

class Environment:
    def __init__(self, path_to_robot, path_to_scanerio) -> None:
        self._path_to_robot = os.path.abspath(path_to_robot)
        self._path_to_scenario = os.path.abspath(path_to_scanerio)
        
        self.model, self.data = self._initilize_simulation()
        
        
    def _initilize_simulation(self) -> tuple:
        scenario_root = etree.parse(self._path_to_scenario).getroot()
        robot_root = etree.parse(self._path_to_robot).getroot()

        if scenario_root.find("include") is None:
            scenario_root.insert(0, etree.Element("include", file=self._path_to_robot))
        try:
            model = mj.MjModel.from_xml_string(etree.tostring(scenario_root))
        except ValueError:
            qpos_old = robot_root.find("keyframe")[0].attrib["qpos"]
            qpos_new = qpos_old + " 0.0"
            robot_root.find("keyframe")[0].attrib["qpos"] = qpos_new
            etree.ElementTree(robot_root).write(self._path_to_robot, pretty_print = True)
        finally:
            model = mj.MjModel.from_xml_string(etree.tostring(scenario_root))
            model.opt.timestep = 0.001
            data = mj.MjData(model)
            mj.mj_resetDataKeyframe(model, data, 0)

            return model, data

    def reset(self):
        mj.mj_resetDataKeyframe(self.model, self.data, 0)
    
    def step(self, viewer):
        mujoco.mj_step(self.model, self.data)
        viewer.sync()

    def run_simulation(self):
        start_time = time.time()
        mujoco.viewer.launch(self.model, self.data) 
        end_time = time.time()
            
    def get_state(self, robot_context: RobotContext) -> State:
        dim_q = self.model.nq 
        # chassis_pos = self.model.body("body_chassis_").pos
        # chassis_quat = self.model.body("body_chassis_").quat
        
        state = State()
        state.time = self.data.time
        if dim_q == 6:
            # state.chassis_pos = chassis_pos
            # state.chassis_quat = chassis_quat
            
            state.q_right = self.data.qpos[:3]
            state.q_left = self.data.qpos[3:]
            
            state.q_dot_right = self.data.qvel[:3]
            state.q_dot_left = self.data.qvel[3:]
        
        if dim_q == 9:
            state.chassis_pos = np.array([self.data.qpos[0], 0 , self.data.qpos[1]])
            state.chassis_quat = R.from_euler('xyz', [0, self.data.qpos[2], 0]).as_quat()

            state.q_right = self.data.qpos[3:6]
            state.q_left = self.data.qpos[6:]
            
            state.q_dot_right = self.data.qvel[3:6]
            state.q_dot_left = self.data.qvel[6:]
        
        if dim_q == 12:
            state.chassis_pos = self.data.qpos[:3]
            state.chassis_quat = self.data.qpos[3:7]
            state.q_right = self.data.qpos[7:10]
            state.q_left = self.data.qpos[10:]
            
            state.q_dot_right = self.data.qvel[6:9]
            state.q_dot_left = self.data.qvel[9:]
        
        return state
    
    def set_actuator_signal(self, name, signal):
        self.data.ctrl[name] = signal

    def get_data_sensors(self, robot_context: RobotContext) -> SensorData:
        
        sensor_data = SensorData()
        
        sensor_data.time = self.data.time
        
        sensor_data.chassis_imu.accel = self.data.sensor(robot_context.accelerometer).data.copy()
        sensor_data.chassis_imu.gyro = self.data.sensor(robot_context.gyro).data.copy()
        sensor_data.chassis_imu.mag = self.data.sensor(robot_context.magnetometer).data.copy()
        sensor_data.chassis_imu.orientation = self.data.sensor(robot_context.orientation).data.copy()
        
        for name in robot_context.left_pos + robot_context.left_vel:
            sensor_data.left_leg[name] = self.data.sensor(name).data.copy()

        for name in robot_context.right_pos + robot_context.right_vel:
            sensor_data.right_leg[name] = self.data.sensor(name).data.copy()
            
        return sensor_data
    
class EnvironmentPendulum(Environment):
    def __init__(self, path_to_robot, path_to_scanerio) -> None:
        super().__init__(path_to_robot, path_to_scanerio)
        
    def _initilize_simulation(self) -> tuple:
        scenario_root = etree.parse(self._path_to_scenario).getroot()
        robot_root = etree.parse(self._path_to_robot).getroot()

        if scenario_root.find("include") is None:
            scenario_root.insert(0, etree.Element("include", file=self._path_to_robot))
        qpos_old = robot_root.find("keyframe")[0].attrib["qpos"]
        qpos_new = qpos_old + " 1.73 0 2.2 1.0 0.0 0.0 0.0"
        robot_root.find("keyframe")[0].attrib["qpos"] = qpos_new
        etree.ElementTree(robot_root).write(self._path_to_robot, pretty_print = True)
        model = mj.MjModel.from_xml_string(etree.tostring(scenario_root))
        model.opt.timestep = 0.001
        data = mj.MjData(model)
        mj.mj_resetDataKeyframe(model, data, 0)

        return model, data