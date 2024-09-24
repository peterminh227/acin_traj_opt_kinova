import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.mujoco_creator.env_assembly import CustomEnvironment
import json
import arc_core
import numpy as np
import mujoco
import time
from mujoco.viewer import launch_passive
from utils.environment_utils import get_joint_ids, get_actuators_ids


def get_joint_and_actuator_ids(prefix, joint_names, actuator_names, mj_model):
    joint_ids = get_joint_ids([f'{prefix}/{joint}' for joint in joint_names], mj_model)
    actuator_ids = get_actuators_ids([f'{prefix}/{actuator}' for actuator in actuator_names], mj_model)
    return joint_ids, actuator_ids


# Define environment and generate iiwas with UR grippers
env = CustomEnvironment()
env.generate_iiwas_with_ur_grippers()

# Compile model
mj_model, mj_data = env.compile_model()

# Define joint and actuator names for robot 0
robot0_joint_names = ['iiwa/joint_{}'.format(i) for i in range(1, 8)]
robot0_actuator_names = ['iiwa/torq_j{}'.format(i) for i in range(1, 8)]
joint_ids_robot0, actuator_ids_robot0 = get_joint_and_actuator_ids('cage', robot0_joint_names, robot0_actuator_names,
                                                                   mj_model)

# Define joint and actuator names for robot 1
robot1_joint_names = ['iiwa_1/joint_{}'.format(i) for i in range(1, 8)]
robot1_actuator_names = ['iiwa_1/torq_j{}'.format(i) for i in range(1, 8)]
joint_ids_robot1, actuator_ids_robot1 = get_joint_and_actuator_ids('cage', robot1_joint_names, robot1_actuator_names,
                                                                   mj_model)

# Define joint and actuator names for left L-axis
left_laxis_joint_names = ['x_left', 'y_left']
left_laxis_actuator_names = ['actuatorX_left', 'actuatorY_left']
joint_ids_laxis_left, actuator_ids_laxis_left = get_joint_and_actuator_ids('cage', left_laxis_joint_names,
                                                                           left_laxis_actuator_names, mj_model)

# Define joint and actuator names for right L-axis
right_laxis_joint_names = ['x_right', 'y_right']
right_laxis_actuator_names = ['actuatorX_right', 'actuatorY_right']
joint_ids_laxis_right, actuator_ids_laxis_right = get_joint_and_actuator_ids('cage', right_laxis_joint_names,
                                                                             right_laxis_actuator_names, mj_model)

# Start logging
arc_core.start_logging(arc_core.LogLevel.Debug)

# redirect stdout and stderr to print arc warnings and debug info
with arc_core.ostream_redirect(stdout=True, stderr=True):
    # Constants
    Ts = mj_model.opt.timestep
    sim_flag = False
    hanging = True
    T_traj = 2

    # Controller parameters
    ctr_param = arc_core.Iiwa.IiwaContrParam(sim_flag)
    ctr_param_linaxis = arc_core.LinearAxis.LinAxisContrParam()

    # Slow and fast normalization factors
    f_c_slow_norm = ctr_param.f_c_slow * (2 * Ts)
    f_c_fast_norm = ctr_param.f_c_fast * (2 * Ts)

    # Joint and Cartesian parameters
    js_param = arc_core.Iiwa.JointCTParameter(ctr_param)
    ts_param = arc_core.Iiwa.CartesianCTParameter(ctr_param)

    # Singular perturbation parameter
    sp_param = arc_core.Iiwa.SingularPerturbationParameter(ctr_param.K_sp, ctr_param.D_sp, f_c_fast_norm)

    # Robot models
    robot_model = arc_core.Iiwa.RobotModel(f_c_slow_norm, Ts)
    robot_model2 = arc_core.Iiwa.RobotModel(f_c_slow_norm, Ts)
    l_axis_left = arc_core.LinearAxis.RobotModel()
    l_axis_right = arc_core.LinearAxis.RobotModel()

    # Friction and gravity compensation parameters
    B_fc = robot_model.get_B()
    fc_param = arc_core.Iiwa.FrictionCompensationParameter(ctr_param.L_fc, B_fc, f_c_fast_norm)
    gc_param = arc_core.Iiwa.GravityCompParameter(ctr_param.D_gc)

    # IIWA controllers
    arc_contr = arc_core.Iiwa.LBRIiwa(robot_model, arc_core.Time(Ts), js_param, ts_param, sp_param, fc_param, gc_param,
                                      hanging)
    arc_contr2 = arc_core.Iiwa.LBRIiwa(robot_model2, arc_core.Time(Ts), js_param, ts_param, sp_param, fc_param,
                                       gc_param, hanging)

    # Linear axis controllers
    js_param_laxis = arc_core.LinearAxis.JointCTParameter(ctr_param_linaxis)
    ts_param_laxis = arc_core.LinearAxis.CartesianCTParameter(ctr_param_linaxis)
    arc_contr_laxis_left = arc_core.LinearAxis.LinearAxis(l_axis_left, js_param_laxis, ts_param_laxis,
                                                          arc_core.Time(Ts))
    arc_contr_laxis_right = arc_core.LinearAxis.LinearAxis(l_axis_right, js_param_laxis, ts_param_laxis,
                                                           arc_core.Time(Ts))

    # Disable singular perturbation and friction compensation
    arc_contr.set_singular_perturbation_state(False)
    arc_contr2.set_singular_perturbation_state(False)
    arc_contr.set_friction_compensation_state(False)
    arc_contr2.set_friction_compensation_state(False)

    # IIWA initializations
    q0 = mj_data.qpos[joint_ids_robot0]
    q01 = mj_data.qpos[joint_ids_robot1]
    joint_init_pos = np.ones((7, 1))
    arc_contr.start(arc_core.Time(mj_data.time), q0, joint_init_pos, arc_core.Time(T_traj))
    arc_contr2.start(arc_core.Time(mj_data.time), q01, joint_init_pos, arc_core.Time(T_traj))

    # Linear axis initializations
    q0_laxis_left = mj_data.qpos[joint_ids_laxis_left]
    q0_laxis_right = mj_data.qpos[joint_ids_laxis_right]
    joint_init_pos_l_axis = np.array([0.5, 0.5])
    arc_contr_laxis_left.start(arc_core.Time(mj_data.time), q0_laxis_left, joint_init_pos_l_axis, arc_core.Time(T_traj))
    arc_contr_laxis_right.start(arc_core.Time(mj_data.time), q0_laxis_right, joint_init_pos_l_axis,
                                arc_core.Time(T_traj))

    # Trajectory servers
    com_server_robot0 = arc_core.Iiwa.TrajectoryServer(arc_contr)
    com_server_robot1 = arc_core.Iiwa.TrajectoryServer(arc_contr2, server_port=20001, client_port=21001)
    com_server_l_axis_left = arc_core.LinearAxis.TrajectoryServer(arc_contr_laxis_left, server_port=30002,
                                                                  client_port=31002)
    com_server_l_axis_right = arc_core.LinearAxis.TrajectoryServer(arc_contr_laxis_right, server_port=40002,
                                                                   client_port=41002)

    framerate = 100
    frame_time = 1.0 / framerate
    print("frame_time: ", frame_time)
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        # Close the viewer automatically after 30 wall-seconds.
        start = time.time()
        last_frame_time = time.time()
        while viewer.is_running():
            # Record the start time of each iteration for timekeeping
            step_start = time.time()

            # Poll trajectory servers for incoming data
            com_server_robot0.poll()
            com_server_robot1.poll()
            com_server_l_axis_left.poll()
            com_server_l_axis_right.poll()

            # Update control signals for Robot 0
            q0_robot0 = mj_data.qpos[joint_ids_robot0]
            tau_sens_act_robot0 = np.zeros((7, 1))
            tau_motor_act_robot0 = np.zeros((7, 1))
            tau_set_robot0 = arc_contr.update(arc_core.Time(mj_data.time), q0_robot0, tau_sens_act_robot0,
                                              tau_motor_act_robot0, False)
            mj_data.ctrl[actuator_ids_robot0] = tau_set_robot0

            # Update control signals for Robot 1
            q0_robot1 = mj_data.qpos[joint_ids_robot1]
            tau_sens_act_robot1 = np.zeros((7, 1))
            tau_motor_act_robot1 = np.zeros((7, 1))
            tau_set_robot1 = arc_contr2.update(arc_core.Time(mj_data.time), q0_robot1, tau_sens_act_robot1,
                                               tau_motor_act_robot1, False)
            mj_data.ctrl[actuator_ids_robot1] = tau_set_robot1

            # Update control signals for left linear axis
            q0_laxis_left = mj_data.qpos[joint_ids_laxis_left]
            tau_sens_act_laxis_left = np.zeros((2, 1))
            tau_set_laxis_left = arc_contr_laxis_left.update(arc_core.Time(mj_data.time), q0_laxis_left)
            mj_data.ctrl[actuator_ids_laxis_left] = tau_set_laxis_left

            # Update control signals for right linear axis
            q0_laxis_right = mj_data.qpos[joint_ids_laxis_right]
            tau_sens_act_laxis_right = np.zeros((2, 1))
            tau_set_laxis_right = arc_contr_laxis_right.update(arc_core.Time(mj_data.time), q0_laxis_right)
            mj_data.ctrl[actuator_ids_laxis_right] = tau_set_laxis_right

            # Step the MuJoCo physics simulation
            mujoco.mj_step(mj_model, mj_data)

            # Send updated robot data to trajectory servers
            com_server_robot0.send_robot_data(arc_core.Time(mj_data.time))
            com_server_robot1.send_robot_data(arc_core.Time(mj_data.time))
            com_server_l_axis_left.send_robot_data(arc_core.Time(mj_data.time))
            com_server_l_axis_right.send_robot_data(arc_core.Time(mj_data.time))

            # Get robot states and errors for analysis
            iiwa_state_robot0 = arc_core.Iiwa.RobotState(arc_contr, mj_data.time)
            iiwa_state_robot1 = arc_core.Iiwa.RobotState(arc_contr2, mj_data.time)
            e_js_robot0 = arc_contr.get_joint_error()
            e_js_robot1 = arc_contr2.get_joint_error()
            e_ts_robot0 = arc_contr.get_cartesian_error()
            e_ts_robot1 = arc_contr2.get_cartesian_error()

            # Calculate time remaining until next frame
            time_until_next_frame = frame_time - (time.time() - last_frame_time)
            if time_until_next_frame < 0:
                # If time until next frame is negative, reset last frame time and synchronize with viewer
                last_frame_time = time.time()
                viewer.sync()  # Pick up changes to the physics state, apply perturbations, update options from GUI
                print("q_set (Robot 0):", arc_contr.get_q_set())
                print("q_set (Robot 1):", arc_contr2.get_q_set())

            # Sleep to maintain the desired timestep
            time_until_next_step = mj_model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
