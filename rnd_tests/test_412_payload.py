import numpy as np
import mujoco as mj
from copy import deepcopy
from scipy.spatial.transform import Rotation as R

import sys

import xml
sys.path.append('./')

from builder.robot_part import DoFChassis
import builder.xml_robot as rbtb
import builder.xml_scene as scene
import simulation.old_envs as sim
from simulation.context import RobotContext

import keyboard
# import tty
# import termios

# def readchar():
    
#     fd = sys.stdin.fileno()
#     old_settings = termios.tcgetattr(fd)
#     try:
#         tty.setraw(sys.stdin.fileno())
#         ch = sys.stdin.read(1)
#     finally:
#         termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
#     return ch

# def readkey(getchar_fn=None):
#     getchar = getchar_fn or readchar
#     c1 = getchar()
#     if ord(c1) != 0x1b:
#         return c1
#     c2 = getchar()
#     if ord(c2) != 0x5b:
#         return c1
#     c3 = getchar()
#     return chr(0x10 + ord(c3) - 65)


# from pynput import keyboard

# def on_press(key):
#     try:
#         print('alphanumeric key {0} pressed'.format(key.char))
#     except AttributeError:
#         print('special key {0} pressed'.format(key))

# def on_release(key):
#     print('{0} released'.format(key))
#     if key == keyboard.Key.esc:
#         # Stop listener
#         return False


def wrap_stinky_to_right_quat(q):
    out_q = np.zeros(4)
    out_q[0:3] = q[1:]
    out_q[3] = q[0]
    
    return out_q

params = rbtb.RobotParameters()

xml_robot_builder = rbtb.OpenKinematicRobot(params, np.array([np.pi/4, -np.pi/2, 0]), np.array([-np.pi/4, np.pi/2, 0]), DoFChassis.FREE) 
xml_robot_builder.add_paylaod(5)
# xml_scene = scene.FlatScene()
# xml_scene = scene.SlideScene()
# xml_scene = scene.Scene()

# xml_scene.save_xml("models")

kv = 2 * np.pi * 100 / 60

tires = [xml_robot_builder.left_tire, xml_robot_builder.right_tire]

links = [xml_robot_builder.left_upper_link, xml_robot_builder.right_upper_link, xml_robot_builder.left_down_link, xml_robot_builder.right_down_link]

#t_motor_ak80_6 = {"biastype":"affine", "gainprm":[35, 0, 0], "biasprm":[0, -35, -0.65], "ctrlrange":[-2, 2]} 

# my_general_model =  {"gainprm":[1, 0, 0,0,0,0], "biasprm":[0, 0, -kv,0,0,0], "gear":[9,0,0,0,0,0], "forcerange":[-58,58], "ctrlrange":[-1,1], "ctrllimited":True, "forcelimited":True}

my_motor_model =  {"gear":[9,0,0,0,0,0], "forcerange":[-50,50], "ctrlrange":[-50,50], "ctrllimited":True, "forcelimited":True}
# my_motor_model =  {"forcerange":[-50,50], "ctrlrange":[-50,50], "ctrllimited":True, "forcelimited":True}



for tire in tires:

    tire.add_sensor("pos", "jointpos")#, **{"noise":0.05})

    tire.add_sensor("vel","jointvel")#,**{"noise":0.00})

    tire.add_actuator("act", "motor", **{"ctrlrange":[-50,50], "ctrllimited":True})


for link in links:

    link.add_sensor("pos", "jointpos")#, **{"noise":0.05})
    link.add_sensor("vel", "jointvel")#, **{"noise":0.05})

    link.add_actuator("act", "motor", **my_motor_model )
 

xml_robot_builder.chassis.add_imu()
xml_robot_builder.save_xml("models")

env = sim.Environment("models/robot.xml", "models/scene.xml")
# env = sim.Environment("models/robot.xml", "models/scene.xml")
robot_context = RobotContext("models/robot.xml")

# %%
error_1_prev = 0
error_2_prev = 0
error_3_prev = 0
error_4_prev = 0



x = lambda t: 1 * np.sin(0.1*2 * np.pi * t)
y = lambda t: 1 * np.cos(0.1*2 * np.pi * t) - 1
dx = lambda t: 0.1*10 * 2 * np.pi * np.cos(0.1*2 * np.pi * t)
dy = lambda t: 0.1*-10 * 2 * np.pi * np.sin(0.1*2 * np.pi * t)

d = params.chassis.size[2]
r = params.tire.size[0]

kp_th = 0
kp_v = 0
err_theta_prev = 0

actual_pos = []
des_pos = []

K = np.array([[ -3.60555128,  10.30372101, -58.21402709,  -7.26896898,   2.03970951,  -16.7978224 ], [-3.60555128, -10.30372101, -58.21402709,  -7.26896898,  -2.03970951,  -16.7978224 ]])

import controler.linear_model as lm

m_chassis = params.chassis.mass + params.down_link.mass*2 + params.upper_link.mass*2
m_tire = params.tire.mass

Ixx_chassis = 1/12 * m_chassis * (params.chassis.size[1] ** 2 + params.chassis.size[2] ** 2)
Iyy_chassis = 1/12 * m_chassis * (params.chassis.size[0] ** 2 + params.chassis.size[2] ** 2)
Izz_chassis = 1/12 * m_chassis * (params.chassis.size[0] ** 2 + params.chassis.size[1] ** 2)

Iwa_tire = 1/2 * m_tire * params.tire.size[0] ** 2
Iwd_tire = 1/4 * m_tire * params.tire.size[0] ** 2 +  1/12 * m_tire * params.tire.size[1]**2 * 4

r_tire = params.tire.size[0]
b_chassis = params.chassis.size[1] + params.down_link.size[1]*2 + params.upper_link.size[1]*2 + params.tire.size[1]

lin_model = lm.LinModelSigway(m_chassis, r_tire, Iwa_tire, Iwd_tire, b_chassis, 9.81, Iyy_chassis, Izz_chassis, m_tire, 0.0239, 0.0239)

states = []
control_signal = []
time_arr = []
length_history = [0]

prev_l_lqr = 0
actual_com_robot = []
K = np.zeros(6)

prev_tire_pos_l = 0
prev_tire_pos_r = 0

prev_x = 0
prev_y = 0
prev_alpha = 0
prev_p = 0 

des_pos_r = np.array([0, -0.648185424949238+0.0825])
des_pos_l = np.array([0, -0.648185424949238+0.0825])

beta_angle_correction = 0.155

coeff_ramp_movs = np.deg2rad(20) / 30
# Collect events until released
# listener = keyboard.Listener(on_press=on_press, on_release=on_release)
# listener.start()
def controller(model, data):
    
    
    global prev_l_lqr
    global prev_tire_pos_l
    global prev_tire_pos_r
    
    global prev_x
    global prev_y
    global prev_alpha
    global prev_p
    global des_pos_r
    global des_pos_l
    global beta_angle_correction
    
    global K

    des_dalpha = 0
    v_des = 1.4
    p_des = prev_p
    alpha_des = prev_alpha

    MAX_TORQUE = 7

    
    des_pos.append([x(data.time), y(data.time)])
    actual_pos.append(deepcopy(data.qpos[:2]))


    data_sensors = env.get_data_sensors(robot_context)
    
    rpy_angles_body = R.from_quat(wrap_stinky_to_right_quat(data_sensors.chassis_imu.orientation)).as_euler('xyz')
    
    leg_q_l = np.array([data_sensors.left_leg['left_upper_link/pos'][0], 
                    data_sensors.left_leg['left_upper_link/left_down_link/pos'][0],
                    data_sensors.left_leg['left_upper_link/left_down_link/left_tire/pos'][0]])
    
    leg_q_r = np.array([data_sensors.right_leg['right_upper_link/pos'][0], 
                        data_sensors.right_leg['right_upper_link/right_down_link/pos'][0],
                        data_sensors.right_leg['right_upper_link/right_down_link/right_tire/pos'][0]])
    
    fk_r = xml_robot_builder.fk_right_leg(leg_q_r)
    fk_l = xml_robot_builder.fk_left_leg(leg_q_l)
    
    l_pendulum = np.linalg.norm(fk_r[2][:3,3]) - 0.202
    length_history.append(l_pendulum)
    

    
    # %% Odometric Localization
    
    Del_tire_R = leg_q_r[2] - prev_tire_pos_r
    Del_tire_L = leg_q_l[2] - prev_tire_pos_l

    Del_s = r/2 * (Del_tire_L + Del_tire_R)
    
    Del_alp = r/ 2 / b_chassis * (Del_tire_R - Del_tire_L)
    
    alpha = rpy_angles_body[2]
    
    x_curr = actual_pos[-1][0] #prev_x + Del_s/Del_alp * (np.sin(alpha) - np.sin(prev_alpha)) if not np.isclose(Del_alp, 0) else 0
    y_curr = actual_pos[-1][1] #prev_y - Del_s/Del_alp * (np.cos(alpha) - np.cos(prev_alpha)) if not np.isclose(Del_alp, 0) else 0
    
    prev_alpha = alpha
    prev_x = x_curr
    prev_y = y_curr
    
    actual_com_robot.append(np.linalg.norm(data.subtree_com[0] - (data.xpos[6]+data.xpos[12])/2))
    
    alpha = rpy_angles_body[2]
    beta = rpy_angles_body[1]
    x_Rc = - l_pendulum * np.sin(beta)
    x_Ir = x_curr - x_Rc
    y_Ir = y_curr - x_Rc * np.sin(alpha)

    p = x_Ir * np.cos(alpha) + y_Ir * np.sin(alpha)
    
    dtheta_l = -data_sensors.left_leg['left_upper_link/left_down_link/left_tire/vel'][0]  #left wheel velocity
    dtheta_r = data_sensors.right_leg['right_upper_link/right_down_link/right_tire/vel'][0] #right wheel velocity

    v = params.tire.size[0]/2 * (dtheta_l + dtheta_r)
    
    dalpha = r/d/2*(dtheta_r - dtheta_l)#data_sensors.chassis_imu.gyro[2]

    dbeta = data_sensors.chassis_imu.gyro[1]

    X = np.array([p, alpha, beta, v, dalpha, dbeta])

    # aplha_des *0 and v_des*0 for testing robot in positon mode
    # remove zeros for trajectory control 
    
    # 10 deg
    # v_des = 0.8
    # p_des = data.time*v_des
    # 20 deg
    # v_des = 0.6
    # p_des = 0
    # 30 deg
    # v_des = 0.8
    # p_des = 0
    
    # one_leg 25 deg
    # v_des = 1.4
    # p_des = 0#data.time*v_des
    #____________________________________________________________________________________________________
    if np.abs(prev_l_lqr - l_pendulum) >= 0.2:
        # K = lin_model.calculate_lqr(np.diag([0.78,6.37,39.06,0.25, 0.19, 11.11]), np.diag([0.03, 0.03]), l_pendulum)
        K = lin_model.calculate_lqr(np.diag([10,6.37,39.06,0.25, 0.19, 11.11]), np.diag([0.03, 0.03]), l_pendulum)
        prev_l_lqr = l_pendulum
    X_des = np.array([p_des, alpha_des, beta_angle_correction, v_des, des_dalpha, 0])
    u_0 = np.array([0,0])
    u = u_0 - K @ (X - X_des)
    

    right_wheel = u[0] if np.abs(u[0]) < MAX_TORQUE else np.sign(u[0])*MAX_TORQUE
    left_wheel = -u[1] if np.abs(-u[1]) < MAX_TORQUE else np.sign(-u[1])*MAX_TORQUE
    
    J_l = xml_robot_builder.Jl(leg_q_l[:-1])
    J_r = xml_robot_builder.Jr(leg_q_r[:-1])
    
    Kp = np.diag([100, 100])
    Kd = np.diag([40, 40])
    
    gamma = rpy_angles_body[0]

    right_wheel += 7*(coeff_ramp_movs * data.time)    
    left_wheel -= 7*(coeff_ramp_movs * data.time)    
    # err_gamma = 0 - gamma
    # k_gamma =  -10
    
    # del_pos_l = 0
    # del_pos_r = 0
    
    # if np.round(gamma,2) > 0:
    #     del_pos_l = err_gamma * k_gamma
    #     left_wheel -= 5 * np.abs(del_pos_l) if np.abs(del_pos_l) <= 0.5 else 0.5 * 5
    # elif np.round(gamma,2) < 0:
    #     del_pos_r = -err_gamma * k_gamma
    #     right_wheel += 3 * np.abs(del_pos_r) if np.abs(del_pos_r) <= 0.5 else 0.5 * 3
    
    # print(f"del L:{del_pos_r} {del_pos_l}")
    print(f"tau {right_wheel}, {left_wheel}")
    print(f"X {np.round(X,2)}")
    
    # des_pos_r = np.array([0, -0.648185424949238+0.0825 + del_pos_r])
    # des_pos_l = np.array([0, -0.648185424949238+0.0825 + del_pos_l])
    
    act_pos_r = fk_r[2][:3,3]
    act_pos_r = np.array([act_pos_r[0], act_pos_r[2]])
    
    act_pos_l = fk_l[2][:3,3]
    act_pos_l = np.array([act_pos_l[0], act_pos_l[2]])
    
    dq_left = np.array([data_sensors.left_leg['left_upper_link/vel'], data_sensors.left_leg['left_upper_link/left_down_link/vel'], data_sensors.left_leg['left_upper_link/left_down_link/left_tire/vel']])
    
    dq_right = np.array([data_sensors.right_leg['right_upper_link/vel'], data_sensors.right_leg['right_upper_link/right_down_link/vel'], data_sensors.right_leg['right_upper_link/right_down_link/right_tire/vel']])
    
    left_tau = J_l.T @ (Kp @ (des_pos_l - act_pos_l) + Kd @ (np.array([0,0]) - np.squeeze(J_l @ dq_left[:-1])))
    right_tau = J_r.T @ (Kp @ (des_pos_r - act_pos_r) + Kd @ (np.array([0,0]) - np.squeeze(J_r @ dq_right[:-1])))
    
    right_conf = np.array([right_tau[0], right_tau[1]])
    left_conf =  np.array([left_tau[0], left_tau[1]])

    prev_p = p
    prev_alpha = alpha
    
    states.append(X)
    control_signal.append([left_wheel, right_wheel])
    time_arr.append(deepcopy(data.time))
    try:
        env.data.ctrl = robot_context.get_control_vector(left_conf, left_wheel, right_conf, right_wheel)
    except ValueError:
        env.data.ctrl[:-2] = robot_context.get_control_vector(left_conf, left_wheel, right_conf, right_wheel)
        env.data.ctrl[-2] = coeff_ramp_movs * data.time
        env.data.ctrl[-1] = coeff_ramp_movs 


env.reset()
mj.set_mjcb_control(controller)
env.run_simulation()
print(env.data)
env.get_state(robot_context)
env.get_data_sensors(robot_context)


import matplotlib.pyplot as plt

actual_pos = np.array(actual_pos)
des_pos = np.array(des_pos)
states = np.array(states).T
time_arr = np.array(time_arr)
control_signal = np.array(control_signal).T
# length_history = np.array(length_history)
# 
# plt.plot(actual_pos[:, 0], actual_pos[:, 1], label="actual")
# plt.plot(des_pos[:, 0], des_pos[:, 1], "--", label="desired")
# plt.legend()
# plt.show()
# # 
# ax = []
# ax.append(plt.subplot(3, 2, 1))
# ax[-1].plot(time_arr, states.T[:,0], "-",linewidth = 1.5, label="p")
# ax[-1].grid()
# ax[-1].legend()
# ax.append(plt.subplot(3, 2, 2))
# ax[-1].plot(time_arr, states.T[:,3], "-",linewidth = 1.5, label="v")
# ax[-1].grid()
# ax[-1].legend()
# ax.append(plt.subplot(3, 2, 3))
# ax[-1].plot(time_arr, states.T[:,1:3], "-",linewidth = 1.5, label=("alpha", "beta"))
# ax[-1].grid()
# ax[-1].legend()
# ax.append(plt.subplot(3, 2, 4))
# ax[-1].plot(time_arr, states.T[:,4:], "-",linewidth = 1.5, label=("dalpha", "dbeta"))
# ax[-1].grid()
# ax[-1].legend()
# ax.append(plt.subplot(3, 1, 3))
# # ax[-1].plot(time_arr[time_arr< 2], control_signal.T[time_arr< 2,:], label=("left_wheel", "right_wheel"))
# ax[-1].plot(time_arr, control_signal.T, label=("left_wheel", "right_wheel"))
# ax[-1].grid()
# ax[-1].legend()
# # 
# # 
# # plt.plot(time_arr, length_history, "-",linewidth = 1.5, label="calc")
# # plt.plot(time_arr, actual_com_robot, "--",linewidth = 1, label="actual")
# plt.grid()
# plt.show()


# %%
