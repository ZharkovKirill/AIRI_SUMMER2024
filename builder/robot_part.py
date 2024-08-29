import os
from dataclasses import dataclass, field
from lxml import etree
from enum import Enum
from typing import List
import numpy as np


from dm_control import mjcf

from itertools import product

class DoFChassis(Enum):
    PLANE = "plane"
    FREE = "free"
    FIX = "fix"

@dataclass
class LinkParameters:
    size: list = field(default_factory=list)  # [x, y, z]
    pos_CoM: list = field(default_factory=list)
    mass: float = 0
    diaginertia: list = field(default_factory=list)

    def __init__(
        self,
        size=[0.025, 0.025, 0.2],
        pos_CoM=[0.0, 0.0, 0.0],
        mass=2,
        diaginertia=[0.0001, 0.0001, 0.0001],
    ):
        self.size = size
        self.pos_CoM = pos_CoM
        self.mass = mass
        self.diaginertia = diaginertia


@dataclass
class TireParameters:
    size: list = field(default_factory=list)  # [radius, width]
    pos_CoM: list = field(default_factory=list)
    mass: float = 0
    diaginertia: list = field(default_factory=list)

    def __init__(
        self,
        size=[0.0825, 0.03],
        pos_CoM=[0.0, 0.0, 0.0],
        mass=1,
        diaginertia=[0.0001, 0.0001, 0.0001],
    ):
        self.size = size
        self.pos_CoM = pos_CoM
        self.mass = mass
        self.diaginertia = diaginertia


@dataclass
class ChassisParameters:
    size: list = field(default_factory=list)  # [x, y, z]
    pos_CoM: list = field(default_factory=list)
    mass: float = 0
    diaginertia: list = field(default_factory=list)
    leg_offset: list = field(default_factory=list)

    def __init__(
        self,
        size=[0.22, 0.16, 0.1],
        pos_CoM=[0.0, 0.0, 0.0],
        mass=15,
        diaginertia=[0.0001, 0.0001, 0.0001],
        leg_offset=[0.0, 0.0, 0.0],
    ):
        self.size = size
        self.pos_CoM = pos_CoM
        self.mass = mass
        self.diaginertia = diaginertia
        self.leg_offset = leg_offset


class Link:
    def __init__(
        self, name, parameters: LinkParameters, use_complex_inertia=False, **kwargs
    ) -> None:
        self.mjcf_model = mjcf.RootElement(model=name + "_link")
        self.parameters = parameters
        self.name = name

        self.body = self.mjcf_model.worldbody.add(
            "body",
            name="body_link_" + self.name,
            **kwargs.get("body", {}),
        )

        self.geom = self.body.add(
            "geom",
            name="geom_link_" + self.name,
            size=self.parameters.size,
            type="box",
            mass=self.parameters.mass,
            pos=[0, -self.parameters.size[1], -self.parameters.size[2]],
            rgba=[0, 0.3, 0.5, 1],
            **kwargs.get("geom", {}),
        )
        self.joint = self.body.add(
            "joint", type="hinge",
            name="joint_link_" + self.name, 
            axis=[0, 1, 0],
            **kwargs.get("joint", {}),
        )

        self.out_site = self.body.add(
            "site",
            pos=np.array(
                [
                    0,
                    -self.parameters.size[1],
                    -self.parameters.size[2],
                ]
            )
            * 2,
            name="link_out_site_" + self.name,
            size=[1e-6] * 3,
        )

        if use_complex_inertia:
            self.body.add(
                "inertial",
                pos=self.parameters.pos_CoM,
                mass=self.parameters.mass,
                diaginertia=self.parameters.diaginertia,
            )

    def add_actuator(self, name, type_actuator, **kwargs):
        self.mjcf_model.actuator.add(
            type_actuator, name=name, joint="joint_link_" + self.name, **kwargs
        )

    def add_sensor(self, name, type_sensor, **kwargs):
        self.mjcf_model.sensor.add(
            type_sensor, name=name, joint="joint_link_" + self.name, **kwargs
        )


class Tire:
    def __init__(
        self, name, parameters: TireParameters, use_complex_inertia=False, **kwargs
    ) -> None:
        self.mjcf_model = mjcf.RootElement(name + "_tire")
        self.parameters = parameters
        self.name = name

        self.body = self.mjcf_model.worldbody.add(
            "body",
            name="body_tire_" + self.name,
        )

        pos_tire_geom = np.array([0, -self.parameters.size[1], 0])
        self.geom = self.body.add(
            "geom",
            name="geom_tire_" + self.name,
            size=self.parameters.size,
            type="cylinder",
            mass=self.parameters.mass,
            pos=pos_tire_geom,
            euler=[90, 0, 0],
            rgba=[0, 0.3, 0.5, 1],
            **kwargs.get("geom", {}),
        )

        self.joint = self.body.add(
            "joint", name="joint_tire_" + self.name, type="hinge", axis=[0, 1, 0],
            **kwargs.get("joint", {}),
        )

        if use_complex_inertia:
            self.body.add(
                "inertial",
                pos=self.parameters.pos_CoM,
                mass=self.parameters.mass,
                diaginertia=self.parameters.diaginertia,
            )

    def add_actuator(self, name, type_actuator, **kwargs):
        self.mjcf_model.actuator.add(
            type_actuator, name=name, joint="joint_tire_" + self.name, **kwargs
        )

    def add_sensor(self, name, type_sensor, **kwargs):
        self.mjcf_model.sensor.add(
            type_sensor, name=name, joint="joint_tire_" + self.name, **kwargs
        )


class Chassis:
    def __init__(
        self,
        name,
        parameters: ChassisParameters,
        position,
        orientation,
        use_complex_inertia,
        type_dof: DoFChassis = DoFChassis.PLANE,
    ) -> None:
        self.mjcf_model = mjcf.RootElement("robot")
        self.parameters = parameters
        self.name = name
        self.position = position
        self.orientation = orientation

        self.body = self.mjcf_model.worldbody.add(
            "body",
            name="body_chassis_" + self.name,
            pos=self.position,
            euler=self.orientation,
        )

        self.body.add(
            "camera",
            name="camera_track_chassis_" + self.name,
            pos=[0, -3, 1],
            zaxis=[0, -1, 0.5],
            mode="track",
        )
        self.geom = self.body.add(
            "geom",
            name="geom_chassis_" + self.name,
            size=self.parameters.size,
            type="box",
            mass=self.parameters.mass,
            rgba=[0, 0.3, 0.5, 1]
        )


        self.pos_right_joint = -np.array([0, self.parameters.size[1], 0]) + np.array(
            self.parameters.leg_offset
        )
        self.right_site = self.body.add(
            "site",
            name="right_site_chassis_" + self.name,
            pos=self.pos_right_joint,
            size=[1e-6] * 3,
        )

        self.pos_left_joint = np.array([0, self.parameters.size[1], 0]) + np.array(
            self.parameters.leg_offset
        )
        self.left_site = self.body.add(
            "site",
            pos=self.pos_left_joint,
            name="left_site_chassis_" + self.name,
            size=[1e-6] * 3,
            euler=[0, 0, 180],
        )
        if use_complex_inertia:
            self.body.add(
                "inertial",
                pos=self.parameters.pos_CoM,
                mass=self.parameters.mass,
                diaginertia=self.parameters.diaginertia,
            )

    def add_lights(self, **kwargs):
        target = "body_chassis_" + self.name
        main_position = np.array(self.parameters.size) + np.array([0.03, 0.03, 0.05])

        for num, xy_multiply in enumerate(product([1, -1], repeat=2)):
            multiply = np.array([*xy_multiply, 1])
            pos = multiply * main_position
            self.mjcf_model.worldbody.add(
                "light",
                name="spotlight_" + str(num),
                mode="targetbodycom",
                pos=pos,
                target=target,
                **kwargs
            )

    def add_additional_frames(self, name, rel_pos):
        offset_y = np.array([0, self.parameters.chassis[1], 0])
        position = []
        position.append(np.array(rel_pos) + offset_y)
        position.append(np.array(rel_pos) - offset_y)
        eulers = [[0, 0, 180], [0, 0, 0]]
        names = ["left_site_chassis_" + name, "right_site_chassis_" + name]

        for name, pos, euler in zip(names, position, eulers):
            self.body.add("site", name=name, pos=pos, size=[1e-6] * 3, euler=euler)

    def add_imu(self, rel_pos=[0, 0, 0]):
        self.body.add(
            "site",
            name="imu",
            pos=rel_pos,
            size=[1e-6] * 3,
        )
        self.mjcf_model.sensor.add("gyro", name=self.name + "_gyro", site="imu")
        self.mjcf_model.sensor.add(
            "accelerometer", name=self.name + "_accelerometer", site="imu"
        )
        self.mjcf_model.sensor.add(
            "magnetometer", name=self.name + "_magnetometer", site="imu"
        )
        self.mjcf_model.sensor.add(
            "framequat", name=self.name + "_orientation", objtype="site", objname="imu"
        )

    def define_dof(self, type_dof: DoFChassis = DoFChassis.PLANE):
        if type_dof == DoFChassis.PLANE:
            self.body.add(
                "joint", name="x_chassis_" + self.name, type="slide", axis=[1, 0, 0]
            )
            self.body.add(
                "joint", name="z_chassis_" + self.name, type="slide", axis=[0, 0, 1]
            )
            self.body.add(
                "joint", name="pitch_chassis_" + self.name, type="hinge", axis=[0, 1, 0]
            )
        if type_dof == DoFChassis.FREE:
            self.body.add("freejoint", name="free_floating_chassis_" + self.name)
        if type_dof == DoFChassis.FIX:
            pass
