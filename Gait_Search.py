from mujoco_py import load_model_from_path, MjSim
import numpy as np
import os
import argparse
from function_util import alive_time_dependent_learning_rate
from SNN_util import Stochastic_LIF, Pacemaker_Stochastic_LIF, motor_pool_internal_connection_generator
from Astrocyte_util import Li_Rinzel

def main(args):
    # Set random seed
    np.random.seed(args.seed)
    
    # Model
    model = load_model_from_path("./MuJoCo_A1_Robot_Model/xml/a1_gait_search.xml")
    sim = MjSim(model)
    
    num_limb = 4
    
    ###############################################################################
    # Parameters
    
    # Temporal parameters
    dt = model.opt.timestep
    simulation_frequency = 1 / dt
    max_session_length_s = 10
    max_session_length_step = int(max_session_length_s * simulation_frequency)
    non_alive_waiting_num_step = 0.5 * simulation_frequency
    alive_length_memory_size = 10
    adaptive_learning_duration_before_terminate = 1
    max_learning_starting_time_s = 2
    
    # Sensory input parameters
    thigh_limit_range_front = (0.6, 1.4) + (0.05, -0.05)
    thigh_limit_range_rear = (0.7, 1.5) + (0.05, -0.05)
    calf_limit_range_front = (-1.6, -1.0) + (0.05, -0.05)
    calf_limit_range_rear = calf_limit_range_front
    
    # Motor output parameters
    motor_impulse_decay_tau = 0.100
    motor_angular_impulse_per_spike_thigh = 0.07 # Kg m^2 / s (unit of angular momentum)
    motor_angular_impulse_per_spike_calf = 0.11 # Kg m^2 / s (unit of angular momentum)
    motor_torque_increment_thigh = motor_angular_impulse_per_spike_thigh / motor_impulse_decay_tau
    motor_torque_increment_calf = motor_angular_impulse_per_spike_calf / motor_impulse_decay_tau
    front_rear_torque_gain = [1,1]
    flexor_extensor_torque_gain = [1,1]
    hip_target_pos_front = -0.1 # negative is outwards
    hip_target_pos_rear = -0.1
    hip_P = 30
    hip_I = 1e1
    hip_D = 0
    initial_limb_pos = (0.7, 0.3)
    
    # Learning parameters
    negative_relative_learning_rate = 0.3
    max_weight = 0.05
    min_weight = -0.05
    spike_trace_decay_tau = 0.01
    STDP_decay_tau = 2
    learning_rate = 5e-10
    
    # Reward parameters
    xvel_reward_coef = 1
    yawomega_abs_reward_coef = -0.1 # -1
    pitchomega_abs_reward_coef = -0.1 # -1
    rollomega_abs_reward_coef = -0.1 # -1
    alive_zcos_threshold = 0.5
    signaling_reward_memory_size = int(0.1 * simulation_frequency)
    
    # SNN
    MOTOR_internal_strength_at_0_dist = 4
    MOTOR_internal_rbf_decay_coef = 0.3
    MOTOR_V1_V2b_weight = 2
    V1_V2b_MOTOR_weight = -50
    
    pool_size = 20
    MOTOR_resting_potential = 0
    MOTOR_membrane_potential_decay_tau = 0.009
    MOTOR_threshold = 10
    MOTOR_refractory_period = 0.005
    MOTOR_AP_trigger_range = 0.2
    
    MOTOR_Ca_concentration_increment_per_AP = 1
    MOTOR_Ca_concentration_decay_tau = 0.250
    MOTOR_Ca_dependent_K_channel_threshold = 10
    MOTOR_Ca_dependent_K_channel_sensitivity = 10
    MOTOR_K_potential_change_coef = 8000
    
    MOTOR_background_stimulation_potential_rate_thigh = 1380 # /s
    MOTOR_background_stimulation_potential_rate_calf = MOTOR_background_stimulation_potential_rate_thigh # /s
    MOTOR_background_stimulation_potential_rate_fluc = 0.5
    MOTOR_background_stimulation_potential_rate_velocity_gain = 40 # /(m/s)
    
    thigh_pos_limit_inhibition_rate = -400 # mV/s
    
    num_V1 = 1
    V1_V2b_resting_potential = 0
    V1_V2b_membrane_potential_decay_tau = 0.009
    V1_V2b_threshold = 10
    V1_V2b_refractory_period = 0.003
    V1_V2b_AP_trigger_range = 0.2
    
    
    # Astrocyte 
    # Li-Rinzel Calcium Model
    IP3_star = 0.16
    r_IP3 = 0.5
    tau_IP3 = 7
    a2 = 0.2
    r_C = 6
    r_L = 0.11
    C0 = 2
    c1 = 0.185
    v_ER = 0.8
    k_ER = 0.1
    d1 = 0.13
    d2 = 1.049
    d3 = 0.9434
    d5 = 0.08234
    
    Ca2_threshold = 0.3
    AG_increment_per_AP = 1e-3 # uM/s
    AG_decay_tau = 1 # s
    ADO_increment_per_release = 0.01
    ADO_decay_tau = 1
    ADO_release_refractory_period = 0.3
    ADO_efficiency = args.ADO_effi # 1.8e-5
    
    ###############################################################################
    # SNN-based CPG construction
    
    # Neurons
    MOTOR_thigh_flexor = {}
    MOTOR_thigh_extensor = {}
    V1_V2b_thigh_flexor_to_extensor = {}
    V1_V2b_thigh_extensor_to_flexor = {}
    MOTOR_calf_flexor = {}
    MOTOR_calf_extensor = {}
    V1_V2b_calf_flexor_to_extensor = {}
    V1_V2b_calf_extensor_to_flexor = {}
    V1_V2b_thigh_extensor_to_calf_flexor = {}
    V1_V2b_thigh_flexor_to_calf_extensor = {}
    for l in range(num_limb):
        # thigh
        MOTOR_thigh_flexor[l] = Pacemaker_Stochastic_LIF(
            pool_size, 
            MOTOR_resting_potential, 
            MOTOR_threshold, 
            MOTOR_membrane_potential_decay_tau, 
            MOTOR_refractory_period, 
            MOTOR_AP_trigger_range, 
            MOTOR_Ca_concentration_increment_per_AP, 
            MOTOR_Ca_concentration_decay_tau, 
            MOTOR_Ca_dependent_K_channel_threshold, 
            MOTOR_Ca_dependent_K_channel_sensitivity, 
            MOTOR_K_potential_change_coef)
        MOTOR_thigh_extensor[l] = Pacemaker_Stochastic_LIF(
            pool_size, 
            MOTOR_resting_potential, 
            MOTOR_threshold, 
            MOTOR_membrane_potential_decay_tau, 
            MOTOR_refractory_period, 
            MOTOR_AP_trigger_range, 
            MOTOR_Ca_concentration_increment_per_AP, 
            MOTOR_Ca_concentration_decay_tau, 
            MOTOR_Ca_dependent_K_channel_threshold, 
            MOTOR_Ca_dependent_K_channel_sensitivity, 
            MOTOR_K_potential_change_coef)
        V1_V2b_thigh_flexor_to_extensor[l] = Stochastic_LIF(
            num_V1, 
            V1_V2b_resting_potential, 
            V1_V2b_threshold, 
            V1_V2b_membrane_potential_decay_tau, 
            V1_V2b_refractory_period, 
            V1_V2b_AP_trigger_range)
        V1_V2b_thigh_extensor_to_flexor[l] = Stochastic_LIF(
            num_V1, 
            V1_V2b_resting_potential, 
            V1_V2b_threshold, 
            V1_V2b_membrane_potential_decay_tau, 
            V1_V2b_refractory_period, 
            V1_V2b_AP_trigger_range)
        
        # calf
        MOTOR_calf_flexor[l] = Pacemaker_Stochastic_LIF(
            pool_size, 
            MOTOR_resting_potential, 
            MOTOR_threshold, 
            MOTOR_membrane_potential_decay_tau, 
            MOTOR_refractory_period, 
            MOTOR_AP_trigger_range, 
            MOTOR_Ca_concentration_increment_per_AP, 
            MOTOR_Ca_concentration_decay_tau, 
            MOTOR_Ca_dependent_K_channel_threshold, 
            MOTOR_Ca_dependent_K_channel_sensitivity, 
            MOTOR_K_potential_change_coef)
        MOTOR_calf_extensor[l] = Pacemaker_Stochastic_LIF(
            pool_size, 
            MOTOR_resting_potential, 
            MOTOR_threshold, 
            MOTOR_membrane_potential_decay_tau, 
            MOTOR_refractory_period, 
            MOTOR_AP_trigger_range, 
            MOTOR_Ca_concentration_increment_per_AP, 
            MOTOR_Ca_concentration_decay_tau, 
            MOTOR_Ca_dependent_K_channel_threshold, 
            MOTOR_Ca_dependent_K_channel_sensitivity, 
            MOTOR_K_potential_change_coef)
        V1_V2b_calf_flexor_to_extensor[l] = Stochastic_LIF(
            num_V1, 
            V1_V2b_resting_potential, 
            V1_V2b_threshold, 
            V1_V2b_membrane_potential_decay_tau, 
            V1_V2b_refractory_period, 
            V1_V2b_AP_trigger_range)
        V1_V2b_calf_extensor_to_flexor[l] = Stochastic_LIF(
            num_V1, 
            V1_V2b_resting_potential, 
            V1_V2b_threshold, 
            V1_V2b_membrane_potential_decay_tau, 
            V1_V2b_refractory_period, 
            V1_V2b_AP_trigger_range)
        
        V1_V2b_thigh_extensor_to_calf_flexor[l] = Stochastic_LIF(
            num_V1, 
            V1_V2b_resting_potential, 
            V1_V2b_threshold, 
            V1_V2b_membrane_potential_decay_tau, 
            V1_V2b_refractory_period, 
            V1_V2b_AP_trigger_range)
        V1_V2b_thigh_flexor_to_calf_extensor[l] = Stochastic_LIF(
            num_V1, 
            V1_V2b_resting_potential, 
            V1_V2b_threshold, 
            V1_V2b_membrane_potential_decay_tau, 
            V1_V2b_refractory_period, 
            V1_V2b_AP_trigger_range)
        
    
    # Connections
    MOTOR_thigh_flexor_internal_connection = {}
    MOTOR_thigh_extensor_internal_connection = {}
    MOTOR_calf_flexor_internal_connection = {}
    MOTOR_calf_extensor_internal_connection = {}
    MOTOR_to_V1 = MOTOR_V1_V2b_weight * np.ones(pool_size)
    V1_V2b_to_R = V1_V2b_MOTOR_weight * np.ones(pool_size)
    for l in range(num_limb):
        # thigh
        MOTOR_thigh_flexor_internal_connection[l] = motor_pool_internal_connection_generator(
            pool_size, 
            MOTOR_internal_strength_at_0_dist, 
            MOTOR_internal_rbf_decay_coef)
        MOTOR_thigh_extensor_internal_connection[l] = motor_pool_internal_connection_generator(
            pool_size, 
            MOTOR_internal_strength_at_0_dist, 
            MOTOR_internal_rbf_decay_coef)
        
        # calf
        MOTOR_calf_flexor_internal_connection[l] = motor_pool_internal_connection_generator(
            pool_size, 
            MOTOR_internal_strength_at_0_dist, 
            MOTOR_internal_rbf_decay_coef)
        MOTOR_calf_extensor_internal_connection[l] = motor_pool_internal_connection_generator(
            pool_size, 
            MOTOR_internal_strength_at_0_dist, 
            MOTOR_internal_rbf_decay_coef)
    
    MOTOR_thigh_flexor_trace = {}
    MOTOR_thigh_extensor_trace = {}
    MOTOR_calf_flexor_trace = {}
    MOTOR_calf_extensor_trace = {}
    for l in range(num_limb):
        MOTOR_thigh_flexor_trace[l] = 0
        MOTOR_thigh_extensor_trace[l] = 0
        MOTOR_calf_flexor_trace[l] = 0
        MOTOR_calf_extensor_trace[l] = 0
        
    muscle_table = [MOTOR_thigh_flexor, MOTOR_thigh_extensor, MOTOR_calf_flexor, MOTOR_calf_extensor]
    V1_V2b_table = [V1_V2b_thigh_extensor_to_flexor, V1_V2b_thigh_flexor_to_extensor, V1_V2b_calf_extensor_to_flexor, V1_V2b_calf_flexor_to_extensor, V1_V2b_thigh_extensor_to_calf_flexor, V1_V2b_thigh_flexor_to_calf_extensor]
    trace_table = [MOTOR_thigh_flexor_trace, MOTOR_thigh_extensor_trace, MOTOR_calf_flexor_trace, MOTOR_calf_extensor_trace]
    
    
    # Astrocyte
    astrocyte_thigh_flexor = {}
    astrocyte_thigh_extensor = {}
    for l in range(num_limb):
        astrocyte_thigh_flexor[l] = Li_Rinzel(
            IP3_star,
            r_IP3,
            tau_IP3,
            a2,
            r_C,
            r_L,
            C0,
            c1,
            v_ER,
            k_ER,
            d1,
            d2,
            d3,
            d5)
        astrocyte_thigh_extensor[l] = Li_Rinzel(
            IP3_star,
            r_IP3,
            tau_IP3,
            a2,
            r_C,
            r_L,
            C0,
            c1,
            v_ER,
            k_ER,
            d1,
            d2,
            d3,
            d5)    
    Astrocyte_table = [astrocyte_thigh_flexor, astrocyte_thigh_extensor]
      
    # 2-AG concentration
    AG_thigh_flexor = {}
    AG_thigh_extensor = {}
    for l in range(num_limb):
        AG_thigh_flexor[l] = 0  
        AG_thigh_extensor[l] = 0  
    AG_table = [AG_thigh_flexor, AG_thigh_extensor]
    
    # ADO concentration
    ADO_thigh_flexor = {}
    ADO_thigh_extensor = {}
    for l in range(num_limb):
        ADO_thigh_flexor[l] = 0  
        ADO_thigh_extensor[l] = 0  
    ADO_table = [ADO_thigh_flexor, ADO_thigh_extensor]
    
    # ADO refractory
    ADO_refractory_thigh_flexor = {}
    ADO_refractory_thigh_extensor = {}
    for l in range(num_limb):
        ADO_refractory_thigh_flexor[l] = 0  
        ADO_refractory_thigh_extensor[l] = 0  
    ADO_refractory_table = [ADO_refractory_thigh_flexor, ADO_refractory_thigh_extensor]
    
    # Inter-limb connection
    gait_STDP = np.zeros((num_limb, num_limb, 2, 2,)) 
    gait_connection = np.zeros((num_limb, num_limb, 2, 2))
    gait_connection_learning_mask = np.ones((num_limb, num_limb, 2, 2))
    for l in range(num_limb):
        gait_connection_learning_mask[l][l] = np.zeros((2, 2))
    
    
    ###############################################################################
    # Main loop
    session_counter = 0
    max_num_session = args.num_session
    
    alive_length_memory = np.zeros(alive_length_memory_size) # session length memory
    
    # Recorders and AP counters
    session_length_recorder = np.zeros(max_num_session)
    final_distance_recorder = np.zeros(max_num_session)
    average_reward_recorder = np.zeros(max_num_session)
    MOTOR_thigh_AP_counter = np.zeros(max_num_session)
    MOTOR_calf_AP_counter = np.zeros(max_num_session)
    V1_V2b_AP_counter = np.zeros(max_num_session)
    
    while 1:
        
        sample_counter = 0 # sample counter of current session
        learning_starting_time_s = np.clip(np.mean(alive_length_memory) - adaptive_learning_duration_before_terminate, 0, max_learning_starting_time_s)
    
        # Session-wise reward
        rewards = np.zeros(max_session_length_step) # reward memory
        signaling_reward_memory = np.zeros(signaling_reward_memory_size)
        signaling_reward_recorder = np.zeros(max_session_length_step)
        
        # Alive counter
        terminate_counter = 0
        
        # Hip motor control initialization
        hip_error_accumulate = np.zeros(4)
        hip_error_previous_step = np.zeros(4)
        
        # Motor variables
        motor_torque_trace = np.zeros(12)
        
        # Total power
        power_accumulator = 0
    
        
        # Session initialization ################################
        
        # Reseting all neuron state variables
        sim.reset()
        for l in range(num_limb):
            # neurons
            for muscle in [0,1,2,3]:
                muscle_table[muscle][l].reset()  
            V1_V2b_thigh_flexor_to_extensor[l].reset()
            V1_V2b_thigh_extensor_to_flexor[l].reset()
            V1_V2b_thigh_extensor_to_calf_flexor[l].reset()
            V1_V2b_thigh_flexor_to_calf_extensor[l].reset()
            V1_V2b_calf_flexor_to_extensor[l].reset()
            V1_V2b_calf_extensor_to_flexor[l].reset()
            MOTOR_thigh_flexor_trace[l] = 0
            MOTOR_thigh_extensor_trace[l] = 0
            MOTOR_calf_flexor_trace[l] = 0
            MOTOR_calf_extensor_trace[l] = 0
            
        gait_STDP = np.zeros((num_limb, num_limb, 2, 2,))
    
        # Setting initial limb joint positions
        for l in range(num_limb):
            thigh_index = l*3 + 1 + 7
            calf_index = l*3 + 2 + 7
            if l < 2:
                sim.data.qpos[thigh_index] = initial_limb_pos[0] * thigh_limit_range_front[0] + initial_limb_pos[1] * thigh_limit_range_front[1]
                sim.data.qpos[calf_index] = initial_limb_pos[0] * calf_limit_range_front[0] + initial_limb_pos[1] * calf_limit_range_front[1]
            else:
                sim.data.qpos[thigh_index] = initial_limb_pos[0] * thigh_limit_range_rear[0] + initial_limb_pos[1] * thigh_limit_range_rear[1]
                sim.data.qpos[calf_index] = initial_limb_pos[0] * calf_limit_range_rear[0] + initial_limb_pos[1] * calf_limit_range_rear[1]
        
        # Update learning progress
        learning_progress_coef = alive_time_dependent_learning_rate((1,0), np.mean(alive_length_memory) / max_session_length_s, 0.9, 0.02) 
        
        
        # Session simulation start ################################
        
        sim.step()
        for t in range(int(max_session_length_s * simulation_frequency)):
            
            # Sensor input
            sensordata = sim.data.sensordata
            vel = np.linalg.norm(sensordata[37:40])
            
            # SNN
            # Neurons and connections update
            for l in range(num_limb):
                # Neurons
                MOTOR_thigh_flexor[l].step(dt)
                MOTOR_thigh_extensor[l].step(dt)
                V1_V2b_thigh_flexor_to_extensor[l].step(dt)
                V1_V2b_thigh_extensor_to_flexor[l].step(dt)
                V1_V2b_thigh_extensor_to_calf_flexor[l].step(dt)
                V1_V2b_thigh_flexor_to_calf_extensor[l].step(dt)
                MOTOR_calf_flexor[l].step(dt)
                MOTOR_calf_extensor[l].step(dt)
                V1_V2b_calf_flexor_to_extensor[l].step(dt)
                V1_V2b_calf_extensor_to_flexor[l].step(dt)
                
                # Connections
                MOTOR_thigh_flexor[l].inject_potential(np.matmul(MOTOR_thigh_flexor[l].AP, MOTOR_thigh_flexor_internal_connection[l]) 
                                                + V1_V2b_thigh_extensor_to_flexor[l].AP * V1_V2b_to_R)
                MOTOR_thigh_extensor[l].inject_potential(np.matmul(MOTOR_thigh_extensor[l].AP, MOTOR_thigh_extensor_internal_connection[l]) 
                                                + V1_V2b_thigh_flexor_to_extensor[l].AP * V1_V2b_to_R)
                V1_V2b_thigh_flexor_to_extensor[l].inject_potential(np.matmul(MOTOR_thigh_flexor[l].AP, MOTOR_to_V1))
                V1_V2b_thigh_extensor_to_flexor[l].inject_potential(np.matmul(MOTOR_thigh_extensor[l].AP, MOTOR_to_V1))
                V1_V2b_thigh_extensor_to_calf_flexor[l].inject_potential(np.matmul(MOTOR_thigh_extensor[l].AP, MOTOR_to_V1))
                V1_V2b_thigh_flexor_to_calf_extensor[l].inject_potential(np.matmul(MOTOR_thigh_flexor[l].AP, MOTOR_to_V1))
                MOTOR_calf_flexor[l].inject_potential(np.matmul(MOTOR_calf_flexor[l].AP, MOTOR_calf_flexor_internal_connection[l]) 
                                                + V1_V2b_calf_extensor_to_flexor[l].AP * V1_V2b_to_R
                                                + V1_V2b_thigh_extensor_to_calf_flexor[l].AP * V1_V2b_to_R)
                MOTOR_calf_extensor[l].inject_potential(np.matmul(MOTOR_calf_extensor[l].AP, MOTOR_calf_extensor_internal_connection[l]) 
                                                + V1_V2b_calf_flexor_to_extensor[l].AP * V1_V2b_to_R
                                                + V1_V2b_thigh_flexor_to_calf_extensor[l].AP * V1_V2b_to_R)
                V1_V2b_calf_flexor_to_extensor[l].inject_potential(np.matmul(MOTOR_calf_flexor[l].AP, MOTOR_to_V1))
                V1_V2b_calf_extensor_to_flexor[l].inject_potential(np.matmul(MOTOR_calf_extensor[l].AP, MOTOR_to_V1))
                
                # Inter-limb connection for limb l
                for m in range(num_limb):
                    if l == m: continue
                    for r_source in [0,1]:
                        for r_target in [0,1]:
                            source_muscle_AP = muscle_table[r_source][l].AP
                            target_muscle = muscle_table[r_target][m]
                            target_muscle.inject_potential(np.sum(source_muscle_AP) * gait_connection[l][m][r_source][r_target] * np.ones(pool_size))
    
                # Limit position inhibition
                thigh_index = l*3 + 1
                calf_index = l*3 + 2
                if sensordata[thigh_index] < (thigh_limit_range_front if l < 2 else thigh_limit_range_rear)[0]:
                    MOTOR_thigh_flexor[l].inject_potential(thigh_pos_limit_inhibition_rate * dt * np.ones(pool_size))
                if sensordata[thigh_index] > (thigh_limit_range_front if l < 2 else thigh_limit_range_rear)[1]:
                    MOTOR_thigh_extensor[l].inject_potential(thigh_pos_limit_inhibition_rate * dt * np.ones(pool_size))
      
                # Background stimulation
                for muscle in [0,1,2,3]:
                    if muscle < 2:
                        muscle_table[muscle][l].inject_potential((MOTOR_background_stimulation_potential_rate_thigh
                                                                  + MOTOR_background_stimulation_potential_rate_velocity_gain * vel) * dt 
                                                        * (1 + MOTOR_background_stimulation_potential_rate_fluc * 2 * (np.random.rand(pool_size) - 0.5)))
                    else:
                        muscle_table[muscle][l].inject_potential((MOTOR_background_stimulation_potential_rate_calf
                                                                  + MOTOR_background_stimulation_potential_rate_velocity_gain * vel) * dt 
                                                        * (1 + MOTOR_background_stimulation_potential_rate_fluc * 2 * (np.random.rand(pool_size) - 0.5)))
                        
                # Astrocyte release
                for muscle in [0,1]:
                    AG_table[muscle][l] = (AG_table[muscle][l] + AG_increment_per_AP * np.sum(muscle_table[muscle][l].AP)) * np.exp(-dt/AG_decay_tau)
                    Astrocyte_table[muscle][l].step(dt, AG_table[muscle][l])
                    if Astrocyte_table[muscle][l].Ca2 > Ca2_threshold and ADO_refractory_table[muscle][l] <= 0:
                        ADO_table[muscle][l] = ADO_table[muscle][l] + ADO_increment_per_release
                        ADO_refractory_table[muscle][l] = ADO_release_refractory_period
                    ADO_table[muscle][l] *= np.exp(-dt/ADO_decay_tau)
                    ADO_refractory_table[muscle][l] -= dt
            
            # Muscle trace
            for l in range(num_limb):
                for muscle in [0,1,2,3]:
                    trace_table[muscle][l] = (trace_table[muscle][l] + np.sum(muscle_table[muscle][l].AP)) * np.exp(-dt/spike_trace_decay_tau)
            
            # Record spike number
            for l in range(num_limb):
                MOTOR_thigh_AP_counter[session_counter] += (np.sum(MOTOR_thigh_flexor[l].AP) + np.sum(MOTOR_thigh_extensor[l].AP))
                MOTOR_calf_AP_counter[session_counter] += (np.sum(MOTOR_calf_flexor[l].AP) + np.sum(MOTOR_calf_extensor[l].AP))
                for v1 in V1_V2b_table:
                    V1_V2b_AP_counter[session_counter] += np.sum(v1[l].AP)
                    
            
            # Motor output
            for l in range(num_limb):
                thigh_index = l*3 + 1
                calf_index = l*3 + 2
                front_rear_index = int(l/2)
                
                motor_torque_trace[thigh_index] = (motor_torque_trace[thigh_index] 
                                                   + front_rear_torque_gain[front_rear_index] 
                                                   * motor_torque_increment_thigh
                                                   * (np.sum(MOTOR_thigh_extensor[l].AP * flexor_extensor_torque_gain[1]) 
                                                      - np.sum(MOTOR_thigh_flexor[l].AP * flexor_extensor_torque_gain[0]))) * np.exp(-dt / motor_impulse_decay_tau)
                
                motor_torque_trace[calf_index] = (motor_torque_trace[calf_index] 
                                                  + front_rear_torque_gain[front_rear_index] 
                                                  * motor_torque_increment_calf
                                                  * (np.sum(MOTOR_calf_extensor[l].AP * flexor_extensor_torque_gain[1]) 
                                                     - np.sum(MOTOR_calf_flexor[l].AP * flexor_extensor_torque_gain[0]))) * np.exp(-dt / motor_impulse_decay_tau)
            
            for m in range(12):
                if m%3 == 0: # hip
                    hip_index = int(m/3)
                    if m/3 < 2: # front
                        hip_error = sensordata[m] - (hip_target_pos_front if hip_index%2 == 0 else -hip_target_pos_front)
                    else: # rear
                        hip_error = sensordata[m] - (hip_target_pos_rear if hip_index%2 == 0 else -hip_target_pos_rear)
                    hip_error_accumulate[hip_index] += hip_error * dt
                    hip_error_rate = (hip_error - hip_error_previous_step[hip_index]) / dt
                    hip_error_previous_step[hip_index] = hip_error
                    hip_torque_output = -hip_P * hip_error - hip_I * hip_error_accumulate[hip_index] - hip_D * hip_error_rate
                    sim.data.ctrl[m] = hip_torque_output
                else: # thigh and calf
                    sim.data.ctrl[m] = motor_torque_trace[m]
                
            power_accumulator += np.inner(sim.data.ctrl, sim.data.sensordata[12:24]) * dt
            
            
            # MuJoCo step
            sim.step()
    
    
            # Reward
            xvel = sensordata[37]
            rollomega_abs = np.abs(sensordata[27])
            pitchomega_abs = np.abs(sensordata[28])
            yawomega_abs = np.abs(sensordata[29])
            
            reward = (xvel_reward_coef * xvel
                    + yawomega_abs_reward_coef * yawomega_abs
                    + pitchomega_abs_reward_coef * pitchomega_abs
                    + rollomega_abs_reward_coef * rollomega_abs)
            
            rewards[sample_counter] = reward
            
            # signaling_reward_memory[0] = signaling_reward
            signaling_reward_memory[0] = reward
            signaling_reward_memory = np.roll(signaling_reward_memory, 1)
            signaling_reward = reward - 0.5 * np.mean(signaling_reward_memory)
            signaling_reward_recorder[t] = signaling_reward
            
            # STDP signal
            for l in range(num_limb):
                # inter-limb STDP
                for m in range(num_limb):
                    if l == m: continue
                    for r_source in [0,1]:
                        for r_target in [0,1]:
                            source_trace = trace_table[r_source][l]
                            source_AP = muscle_table[r_source][l].AP
                            target_trace = trace_table[r_target][m]
                            target_AP = muscle_table[r_target][m].AP
                            
                            gait_STDP[l][m][r_source][r_target] = ((gait_STDP[l][m][r_source][r_target]
                            + source_trace * np.sum(target_AP)
                            - negative_relative_learning_rate 
                            * np.sum(source_AP) * target_trace)
                            * np.exp(-dt/STDP_decay_tau))
                    
            # if in learning period
            if t >= learning_starting_time_s * simulation_frequency:
                current_inter_limb_learning_rate = learning_progress_coef * learning_rate
                
                # inter-limb connection
                gait_connection_reward_STDP_update = (current_inter_limb_learning_rate 
                * gait_STDP
                * signaling_reward)
                
                ADO_formatted = np.array([list(i.values()) for i in ADO_table]).flatten(order='F')
                ADO_gait_connection_bias = -(learning_progress_coef 
                                            * ADO_efficiency 
                                            * np.array([[[[ADO_formatted[m*2], ADO_formatted[m*2+1]],[ADO_formatted[m*2], ADO_formatted[m*2+1]]] for m in range(num_limb)] for l in range(num_limb)]) 
                                            * gait_connection_learning_mask)
                
                gait_connection += (gait_connection_reward_STDP_update 
                * (max_weight - gait_connection) 
                * (gait_connection - min_weight) 
                / (max_weight - min_weight)**2)
                
                gait_connection += (ADO_gait_connection_bias 
                * (max_weight - gait_connection) 
                * (gait_connection - min_weight) 
                / (max_weight - min_weight)**2)
                            
            
            sample_counter += 1
            
            # alive status update
            if sensordata[48] < alive_zcos_threshold:
                terminate_counter += 1
            
            # terminate session if not alive
            if terminate_counter == non_alive_waiting_num_step: 
                break
            
        # Session termination ################################
        
        # update session length memory
        alive_length_memory[0] = sample_counter / simulation_frequency
        alive_length_memory = np.roll(alive_length_memory, 1)
        
        session_length_recorder[session_counter] = sample_counter * dt
        final_distance_recorder[session_counter] = sensordata[30]
        average_reward_recorder[session_counter] = np.mean(rewards[0:sample_counter])
            
        # if learnin started
        if sample_counter > int(learning_starting_time_s * simulation_frequency):
            print("{:d}\t{:.1f} \t{:.1e}({:.1f})\t{:d}\t{:.1f}\t{:.1f}"
                  .format(session_counter, 
                          np.mean(rewards[0:sample_counter]), 
                          np.mean(signaling_reward_recorder[int(learning_starting_time_s * simulation_frequency):sample_counter]), 
                          np.std(signaling_reward_recorder[int(learning_starting_time_s * simulation_frequency):sample_counter]),
                          sample_counter,
                          sensordata[30],
                          power_accumulator/(sample_counter * dt)))
        else:
            print("{:d}\t{:.1f}\t{:d}\t{:.1f}"
                  .format(session_counter, 
                          np.mean(rewards[0:sample_counter]),
                          sample_counter,
                          sensordata[30]))
            
        # printing inter-limb connection and astrocyte state variables
        if session_counter % 20 == 0:
            with np.printoptions(precision=5, suppress=True):
                print('Inter-limb weights:')
                print(gait_connection)
                print('Calcium concentration')
                print(np.array([[astrocyte.Ca2 for astrocyte in muscle.values()] for muscle in Astrocyte_table]))
                print('ADO concentration:')
                print(np.array([list(muscle.values()) for muscle in ADO_table]))
                
        
        session_counter += 1
        if session_counter == max_num_session: 
            break
    
    # %% save recorders and AP counters
    log_root = args.log_dir
    log_root = (log_root + '/') if (log_root[-1] != '/') else log_root
    if not os.path.exists(log_root):
        os.makedirs(log_root)
        
    result_folder_name = "seed_{:d}--ADO_effi_{:.1e}--num_session_{:d}".format(args.seed,
                                                                             args.ADO_effi,
                                                                             args.num_session)
    if not os.path.exists(log_root+result_folder_name):
        os.makedirs(log_root+result_folder_name)
        
    result_counter = 0
    while os.path.exists(log_root + result_folder_name + '/gait_search_{:d}.npz'.format(result_counter)):
        result_counter += 1
        
    with open(log_root + result_folder_name + '/gait_search_{:d}.npz'.format(result_counter), 'wb') as f:
        np.savez(f, gait_connection = gait_connection, 
                 session_length_recorder = session_length_recorder,
                 final_distance_recorder = final_distance_recorder,
                 average_reward_recorder = average_reward_recorder,
                 MOTOR_thigh_AP_counter = MOTOR_thigh_AP_counter,
                 MOTOR_calf_AP_counter = MOTOR_calf_AP_counter,
                 V1_V2b_AP_counter = V1_V2b_AP_counter)
    
# %% arguments parser and entrance
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--ADO-effi", type=float, default=2.3e-5)
    parser.add_argument("--num-session", type=int, default=400)
    return parser.parse_args()

if __name__ == "__main__":
    main(parse_args())
    
    
    
    
    
    
    
    