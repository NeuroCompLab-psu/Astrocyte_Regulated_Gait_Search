# Astrocyte Regulated Gait Search

## Overview
Neuromorphic computing systems, where information is transmitted through action potentials in a bio-plausible fashion, is gaining increasing interest due to its promise of low-power event-driven computing. Application of neuromorphic computing in robotic locomotion research have largely focused on Central Pattern Generators (CPGs) for bionics robotic control algorithms - inspired from neural circuits governing the collaboration of the limb muscles in animal movement. Implementation of artificial CPGs on neuromorphic hardware platforms can potentially enable adaptive and energy-efficient edge robotics applications in resource constrained environments. However, underlying rewiring mechanisms in CPG for gait emergence process is not well understood. This work addresses the missing gap in literature pertaining to CPG plasticity and underscores the critical homeostatic functionality of astrocytes - a cellular component in the brain that is believed to play a major role in multiple brain functions. This paper introduces an astrocyte regulated Spiking Neural Network (SNN)-based CPG for learning locomotion gait through Reward-Modulated STDP for quadruped robots, where the astrocytes help build inhibitory connections among the artificial motor neurons in different limbs. The SNN-based CPG is simulated on a multi-object physics simulation platform resulting in the emergence of a trotting gait while running the robot on flat ground. 23.3X computational power savings is observed in comparison to a state-of-the-art reinforcement learning based robot control algorithm. Such a neuroscience-algorithm co-design approach can potentially enable a quantum leap in the functionality of neuromorphic systems incorporating glial cell functionality.

## Package Requirements
- MuJoCo 2.1.0
- mujoco-py 2.1.0.12
- python 3.8

## Running a simulation
Main script: Gait_Search.py

SNN_util.py, Astrocyte_util.py and function_util.py must be in the same dir as the main script.

bash template:
```
python ./Gait_Search.py \
--log-dir ./results \
--seed 1 \
--ADO-effi 1.8e-5 \
--num-session 400
```

### Parameter interpretation and default (optimal) values
```
--log-dir: log file path (required)
--seed: random number generator seed (default: 2)
--ADO-effi: Adenosine efficiency (default: 1.8e-5)
--num-session: number of training batches (default: 400)
```

## Interpretation of Gait_Search.py on-screen print:
Example:
```
3	0.9 	5.0e-01(0.3)	8327	9.6	1547.8
```
Interpretation:

session number, average reward, mean of signaling reward (standard deviation of signaling reward) session length in steps, final x-directional position, average motor power

## Reference
Please cite this code with the following bibliography:

Han, Z., & Sengupta, A. (2023). Astrocyte Regulated Neuromorphic Central Pattern Generator Control of Legged Robotic Locomotion (Version 2). arXiv. https://doi.org/10.48550/ARXIV.2312.15805

```
@article{title={Astrocyte Regulated Neuromorphic CPG Control of Legged Robotic Locomotion},
  author={Han, Zhuangyu and Sengupta, Abhronil},
  journal={arXiv preprint arXiv:2312.15805},
  year={2023}
}
```
