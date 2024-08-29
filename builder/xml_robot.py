import os
from dataclasses import dataclass
import lxml.etree as etree
import numpy as np
import modern_robotics as mr

from dm_control import mjcf

from builder.robot_part import ChassisParameters, LinkParameters, TireParameters, DoFChassis
from builder.robot_part import Chassis, Link, Tire

from spatial_functions.kinematics import homogeneous_matrix


@dataclass
class RobotParameters:
    chassis = ChassisParameters()
    upper_link = LinkParameters()
    down_link = LinkParameters()
    tire = TireParameters()
    apirate: float = 2000

class OpenKinematicRobot:
    def __init__(
        self,
        robot_parameters: RobotParameters,
        q_init_right: np.ndarray,
        q_init_left: np.ndarray,
        chassis_dof= DoFChassis.PLANE,
        use_complex_inertia=False,
    ) -> None:


        self.parameters: RobotParameters = robot_parameters
        self.chassis_type_dof = chassis_dof
        self.initial_q_right = q_init_right
        self.initial_q_left = q_init_left
        self.chassis: Chassis = Chassis(
            "",
            self.parameters.chassis,
            [0, 0, 0],
            [0, 0, 0],
            use_complex_inertia=use_complex_inertia,
        )
        self.mjcf_model = self.chassis.mjcf_model
        self.chassis.define_dof(type_dof=self.chassis_type_dof)

        self.mjcf_model.option.apirate = self.parameters.apirate
        geom_tire_params = {"friction": [0.95, 0.005, 0.0001], "priority": 1, "condim": 6}
        joint_params = {"damping": 0.0239}
        t_motor_ak80_6 = {"armature":0.01090125, "damping":0.0239, "frictionloss":0.1334} # https://github.com/google-deepmind/mujoco_menagerie/blob/main/google_barkour_v0/barkour_v0.xml
        self.right_upper_link: Link = Link(
            "right_upper",
            self.parameters.upper_link,
            use_complex_inertia=use_complex_inertia,
            **{"joint":t_motor_ak80_6}
        )
        self.right_down_link: Link = Link(
            "right_down",
            self.parameters.down_link,
            use_complex_inertia=use_complex_inertia,
            **{"joint":t_motor_ak80_6}
        )

        self.right_tire: Tire = Tire(
            "right",
            self.parameters.tire,
            use_complex_inertia=use_complex_inertia,
            **{"geom":geom_tire_params, "joint":joint_params},
            
        )

        self.left_upper_link: Link = Link(
            "left_upper",
            self.parameters.upper_link,
            use_complex_inertia=use_complex_inertia,
            **{"joint":joint_params}
        )
        self.left_down_link: Link = Link(
            "left_down",
            self.parameters.down_link,
            use_complex_inertia=use_complex_inertia,
            **{"joint":joint_params}
        )
        self.left_tire: Tire = Tire(
            "left",
            self.parameters.tire,
            use_complex_inertia=use_complex_inertia,
            **{"geom":geom_tire_params, "joint":joint_params}
        )

        self.Jr, self.Jl = self.define_jacobine()
        
        Hb_r = self.fk_right_leg(q_init_right)
        trans_b_r_tire = Hb_r[2][:3, 3]

        Hb_l = self.fk_left_leg(q_init_left)
        trans_b_l_tire = Hb_l[2][:3, 3]

        self.chassis_z_offset = (
            -np.min([trans_b_r_tire[2], trans_b_l_tire[2]])
            + self.parameters.tire.size[0]
        )
        

        self.right_down_link.out_site.attach(self.right_tire.mjcf_model)
        self.right_upper_link.out_site.attach(self.right_down_link.mjcf_model)
        self.chassis.right_site.attach(self.right_upper_link.mjcf_model)

        self.left_down_link.out_site.attach(self.left_tire.mjcf_model)
        self.left_upper_link.out_site.attach(self.left_down_link.mjcf_model)

        self.chassis.left_site.attach(self.left_upper_link.mjcf_model)
        self.chassis.add_lights()

    def fk_right_leg(self, q):
        w_rt_u_lk = np.array([0, 1, 0])
        w_rt_d_lk = np.array([0, 1, 0])
        w_rt_tire = np.array([0, 1, 0])

        r_rt_u_lk = np.array(self.chassis.pos_right_joint)
        r_rt_d_lk = np.array(self.right_upper_link.out_site.pos) + r_rt_u_lk
        r_rt_tire = r_rt_d_lk + np.array(self.right_down_link.out_site.pos)

        v_rt_u_lk = -mr.VecToso3(w_rt_u_lk) @ r_rt_u_lk
        v_rt_d_lk = -mr.VecToso3(w_rt_d_lk) @ r_rt_d_lk
        v_rt_tire = -mr.VecToso3(w_rt_tire) @ r_rt_tire

        Hb_rt_u_lk_init = np.eye(4)
        Hb_rt_u_lk_init[:3, 3] = np.array(self.chassis.pos_right_joint) + np.array(
            self.right_upper_link.body.geom[0].pos
        )

        Hb_rt_d_lk_init = np.eye(4)
        Hb_rt_d_lk_init[:3, 3] = (
            np.array(self.chassis.pos_right_joint)
            + np.array(self.right_upper_link.out_site.pos)
            + np.array(self.right_down_link.body.geom[0].pos)
        )

        Hb_rt_tire_init = np.eye(4)
        Hb_rt_tire_init[:3, 3] = (
            np.array(self.chassis.pos_right_joint)
            + np.array(self.right_upper_link.out_site.pos)
            + np.array(self.right_down_link.out_site.pos)
            + np.array(self.right_tire.body.geom[0].pos)
        )

        e_T_rul_bb = lambda q: homogeneous_matrix(w_rt_u_lk, v_rt_u_lk, q)
        e_T_rdl_b_rul = lambda q: homogeneous_matrix(w_rt_d_lk, v_rt_d_lk, q)
        e_T_rt_b_rdl = lambda q: homogeneous_matrix(w_rt_tire, v_rt_tire, q)

        Hb_rt_u_lk = e_T_rul_bb(q[0]) @ Hb_rt_u_lk_init
        Hb_rt_d_lk = e_T_rul_bb(q[0]) @ e_T_rdl_b_rul(q[1]) @ Hb_rt_d_lk_init
        Hb_rt_tire = (
            e_T_rul_bb(q[0])
            @ e_T_rdl_b_rul(q[1])
            @ e_T_rt_b_rdl(q[2])
            @ Hb_rt_tire_init
        )

        return Hb_rt_u_lk, Hb_rt_d_lk, Hb_rt_tire

    def define_jacobine(self):
        lu = self.parameters.upper_link.size[2]*2
        ld = self.parameters.down_link.size[2]*2
        
        J_r = lambda q: np.array([[-ld * np.cos(q[0] + q[1]) - lu * np.cos(q[0]),
                                       -ld * np.cos(q[0] + q[1])],
                                      [ ld * np.sin(q[0] + q[1]) + lu * np.sin(q[0]),
                                       ld * np.sin(q[0] + q[1])]])
        
        J_l = lambda q: np.array([[ld * np.cos(q[0] + q[1]) + lu * np.cos(q[0]),
                                        ld * np.cos(q[0] + q[1])],
                                        [ld * np.sin(q[0] + q[1]) + lu * np.sin(q[0]),
                                        ld * np.sin(q[0] + q[1])]])

        return J_r, J_l
    def left_simmetry(self, p):
        return np.array([-p[0], -p[1], p[2]])

    def fk_left_leg(self, q):
        w_lt_u_lk = np.array([0, -1, 0])
        w_lt_d_lk = np.array([0, -1, 0])
        w_lt_tire = np.array([0, -1, 0])

        r_lt_u_lk = np.array(self.chassis.pos_left_joint)
        r_lt_d_lk = self.left_simmetry(self.left_upper_link.out_site.pos) + r_lt_u_lk
        r_lt_tire = r_lt_d_lk + self.left_simmetry(self.left_down_link.out_site.pos)

        v_lt_u_lk = -mr.VecToso3(w_lt_u_lk) @ r_lt_u_lk
        v_lt_d_lk = -mr.VecToso3(w_lt_d_lk) @ r_lt_d_lk
        v_lt_tire = -mr.VecToso3(w_lt_tire) @ r_lt_tire

        Hb_lt_u_lk_init = np.eye(4)
        Hb_lt_u_lk_init[:3, 3] = np.array(
            self.chassis.pos_left_joint
        ) + self.left_simmetry(self.left_upper_link.body.geom[0].pos)

        Hb_lt_d_lk_init = np.eye(4)
        Hb_lt_d_lk_init[:3, 3] = (
            np.array(self.chassis.pos_left_joint)
            + self.left_simmetry(self.left_upper_link.out_site.pos)
            + self.left_simmetry(self.left_down_link.body.geom[0].pos)
        )

        Hb_lt_tire_init = np.eye(4)
        Hb_lt_tire_init[:3, 3] = (
            np.array(self.chassis.pos_left_joint)
            + self.left_simmetry(self.left_upper_link.out_site.pos)
            + self.left_simmetry(self.left_down_link.out_site.pos)
            + self.left_simmetry(self.left_tire.body.geom[0].pos)
        )

        e_T_lul_bb = lambda q: homogeneous_matrix(w_lt_u_lk, v_lt_u_lk, q)
        e_T_ldl_b_lul = lambda q: homogeneous_matrix(w_lt_d_lk, v_lt_d_lk, q)
        e_T_lt_b_ldl = lambda q: homogeneous_matrix(w_lt_tire, v_lt_tire, q)

        Hb_lt_u_lk = e_T_lul_bb(q[0]) @ Hb_lt_u_lk_init
        Hb_lt_d_lk = e_T_lul_bb(q[0]) @ e_T_ldl_b_lul(q[1]) @ Hb_lt_d_lk_init
        Hb_lt_tire = (
            e_T_lul_bb(q[0])
            @ e_T_ldl_b_lul(q[1])
            @ e_T_lt_b_ldl(q[2])
            @ Hb_lt_tire_init
        )

        return Hb_lt_u_lk, Hb_lt_d_lk, Hb_lt_tire
    
    def add_paylaod(self, mass_payload):
        self.chassis.body.add(
            "geom",
            name="payload",
            size=[0.1, 0.1, 0.1],
            pos=[0, 0, 0.2],
            type="box",
            mass=mass_payload,
            rgba=[0.5, 0.5, 0.5, 1],
        )

    def save_xml(self, dir=".", name="robot.xml"):
        
        post_process_array = lambda x: str(x.tolist()).replace('[','').replace(']',' ').replace(',','')

        mjcf.export_with_assets(self.mjcf_model, dir, name)
        root = etree.parse(os.path.join(dir, name)).getroot()
        find_class = etree.XPath("//*[@class]")
        for c in find_class(root):
            c.attrib["class"] = "robot/"
        for num, bad in enumerate(root.xpath("//default[@class=\'robot/\']")):
            if num == 0:
                continue
            bad.getparent().remove(bad)
        
        if self.chassis_type_dof == DoFChassis.FREE:
            chassis_frame = np.array([0, 0, self.chassis_z_offset, 1, 0,0,0])
            str_chassis_frame = post_process_array(chassis_frame)
        elif self.chassis_type_dof == DoFChassis.PLANE:
            chassis_frame = np.array([0, self.chassis_z_offset, 0])
            str_chassis_frame = post_process_array(chassis_frame)
        else:
            str_chassis_frame = ""

        
        str_initial_q_right = post_process_array(self.initial_q_right)
        str_initial_q_left = post_process_array(self.initial_q_left)
        qpos = str_chassis_frame + str_initial_q_right + str_initial_q_left
        qpos = qpos.rstrip()
        keyframe = etree.SubElement(root, "keyframe")
        key = etree.Element("key", time="0", name="init", qpos=qpos)
        keyframe.append(key)
        with open(os.path.join(dir, name), "wb") as f:
            f.write(etree.tostring(root, pretty_print=True))