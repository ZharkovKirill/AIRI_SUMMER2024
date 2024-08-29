import os
import lxml.etree as etree

import numpy as np

class RobotContext:
    def __init__(self, xml_file) -> None:
        self.path_to_xml = xml_file
        self.root_robot = etree.parse(xml_file).getroot()
        
        actuators = self.root_robot.xpath("//actuator")[0]
        self.mask_left_leg_conf = np.zeros(len(actuators))
        self.mask_right_leg_conf = np.zeros(len(actuators))
        self.mask_left_wheel = np.zeros(len(actuators))
        self.mask_right_wheel = np.zeros(len(actuators))
        
        for i, actuator in enumerate(actuators):
            tire_flag = actuator.attrib["joint"].find("tire")
            left_flag = actuator.attrib["joint"].find("left")
            
            if tire_flag != -1 and left_flag != -1:
                self.mask_left_wheel[i] = 1
            if tire_flag != -1 and left_flag == -1:
                self.mask_right_wheel[i] = 1
            if tire_flag == -1 and left_flag != -1:
                self.mask_left_leg_conf[i] = 1
            if tire_flag == -1 and left_flag == -1:
                self.mask_right_leg_conf[i] = 1
        
        self.left_pos = []
        self.right_pos = []
        self.left_vel = []
        self.right_vel = []
        
        pos_sensors = self.root_robot.xpath("//sensor/jointpos")
        vel_sensors = self.root_robot.xpath("//sensor/jointvel")
        for sens in pos_sensors:
            left_flag = sens.attrib["joint"].find("left")
            if left_flag != -1:
                self.left_pos.append(sens.attrib["name"])
            if left_flag == -1:
                self.right_pos.append(sens.attrib["name"])
        
        for sens in vel_sensors:
            left_flag = sens.attrib["joint"].find("left")
            if left_flag != -1:
                self.left_vel.append(sens.attrib["name"])
            if left_flag == -1:
                self.right_vel.append(sens.attrib["name"])

        gyro_xml = self.root_robot.xpath("//sensor/gyro")
        accelerometer_xml = self.root_robot.xpath("//sensor/accelerometer")
        magnetometer_xml = self.root_robot.xpath("//sensor/magnetometer")
        orientation_xml = self.root_robot.xpath("//sensor/framequat")
        
        
        self.gyro  = gyro_xml[0].attrib["name"]
        self.accelerometer = accelerometer_xml[0].attrib["name"]
        self.magnetometer = magnetometer_xml[0].attrib["name"]
        self.orientation = orientation_xml[0].attrib["name"]



        
    def get_control_vector(self, left_configuration, left_wheel, right_configuration, right_wheel):
        control_vector = np.zeros(len(self.mask_left_leg_conf))
        control_vector[self.mask_left_leg_conf == 1] = left_configuration
        control_vector[self.mask_right_leg_conf == 1] = right_configuration
        control_vector[self.mask_left_wheel == 1] = left_wheel
        control_vector[self.mask_right_wheel == 1] = right_wheel
        return control_vector

if __name__== "__main__":
    rob_context = RobotContext("D:\\Work_be2r_lab\\NIOKR_2023\\mujoco\\mujoco-simulation\\models\\robot.xml")