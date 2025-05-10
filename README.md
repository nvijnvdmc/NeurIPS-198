# Diagnosing Faults in Deep Reinforcement Learning Policy-Guided Systems: Settings and Benchmarks

This project provides a collection of **benchmark bundles** to be used as inputs for future diagnosis algorithms. It includes fault-injected trajectory data for seven classic Gymnasium environments, along with policy models, fault configurations, and tools to generate custom benchmarks.

## 🧠 Overview

The primary goal of this project is to support **future research on the diagnosis of faulty policy executions** in DRL systems. It does so by providing **precomputed trajectories** under various fault scenarios, which can serve as inputs to diagnostic algorithms.

This resource is designed for researchers working on:
- Plan and policy execution monitoring
- Fault diagnosis in sequential decision-making
- Robustness analysis of learned policies
- Learning of faulty behaviors in policy-guided systems

## 📦 Included Benchmarks

This bundle contains data for the following **Gymnasium environments**:

- `Acrobot-V1`
- `CartPole-V1`
- `MountainCar-V0`
- `Taxi-V3`
- `FrozenLake-V1`
- `Breakout-V4`
- `PongNoFrameskip-V0`

Each environment's bundle includes:
- An **Excel file** of trajectories generated under policy-guided execution with faults
- The **DRL policy** used to generate those trajectories
- Fault scenario input files (e.g., action stuttering, stuck actions, incorrect observations)
- **Code** to regenerate or extend the experiments


### 📁 Special Note on Breakout and Pong

Due to the size of the raw data for **Breakout** and **Pong**:
- The benchmark results are split into **10 Excel files**, each corresponding to a unique random seed (`1` through `10`).
- For each Excel file, a **corresponding `.7z` file** is provided, containing text files with the complete observations and trajectories. **Note that those files are big!** To extract them a preactitioner will need **up to 70Gb of free disk space**. We encourage the practitioners to use 7zip's 'test' function on each 7z file before unzipping it.
- This was necessary because the full-length string representations of trajectories and observations **exceed the character limit of Excel cells**.
- To use these Excel files, there is a need to implement reading the observations and trajectories for each instance. We believe this is easy to do, so we leave the implementation to the user.

### 📋 Excel structure
Each environment's Excel file includes information on the generated trajectories.
Each row shows information on a single experiment. Each column presents specific information value.
Below we specify the names of the columns and what information they include:
```
| **Name**                 | **Type**        | **Description**                                                                                    |
| ------------------------ | --------------- | -------------------------------------------------------------------------------------------------- |
| `domain_name`            | Domain constant | The name of the Gymnasium environment. Determines the environment.                                 |
| `model_name`             | Domain constant | The name of the DRL-generated policy that guides the agent in this environment.                    |
| `seeds`                  | Domain constant | The different seeds for this environment. Determine the starting states.                           |
| `modelled_fault_modes`   | Domain constant | The different fault modes that can be applied to this environment.                                 |
| `fault_probabilities`    | Domain constant | Determine the probabilities with which a fault will occur in every state.                          |
| `instances`              | Domain constant | The different instance numbers to be run for this environment.                                     |
| `seed`                   | Instance input  | The seed for this instance.                                                                        |
| `execution_fault_mode`   | Instance input  | The fault mode of this instance.                                                                   |
| `fault_probability`      | Instance input  | The fault probability for this instance.                                                           |
| `instance`               | Instance input  | The instance number of the current instance.                                                       |
| `registered_actions`     | Instance output | The actions that the policy chose for the agent in every state.                                    |
| `faulty_actions_indices` | Instance output | The ordinal action indices for actions that failed.                                                |
| `observations`           | Instance output | The observed states resulting from the execution of this instance.                                 |
| `trajectory_execution`   | Instance output | The trajectory of this instance (an alternating series of states and actions).                     |
| `num_registered_actions` | Instance output | The number of the registered actions for this instance (the execution length in terms of actions). |
| `num_faulty_actions`     | Instance output | The number of actions that failed (also indicates the size of the faulty actions indices series).  |
| `num_observations`       | Instance output | The number of observed states (the execution length in terms of states).                           |
```

## 🛠️ Installation
The project is built using Python 3.8.7. Make sure you have the correct version before beginning.

First, clone the repository:

```bash
git clone https://github.com/nvijnvdmc/NeurIPS-198.git
cd NeurIPS-198
```
Then configure a virtual environment and install the required dependencies:
```bash
pip install -r requirements.txt
```


## 📁 Directory Structure
Once installing the requirements is done, your project should have the folllowing structure. It contains input files for trajectory generation, policies, code, and the resulting trajectories stored in Excel files: 
```
NeurIPS-198/
    .venv/
    common/
        consts.py
        executor.py
        fault_mode_generators.py
        rl_models.py
        state_refiners.py
        wrappers.py
    p02_traj_factory/
        inputs/
            i1000_Acrobot.json
            i2000_CartPole.json
            i3000_MountainCar.json
            i4000_Taxi.json
            i5000_FrozenLake.json
            i6000_Breakout.json
            i7000_PongNoFrameskip.json
        ouputs/
            i1000_Acrobot.xlsx
            i2000_CartPole.xlsx
            i3000_MountainCar.xlsx
            i4000_Taxi.xlsx
            i5000_FrozenLake.xlsx
            i6000_Breakout-1.xlsx
            i6000_Breakout-1_obs_trajs.7z
            i6000_Breakout-2.xlsx
            i6000_Breakout-2_obs_trajs.7z
            ...
            i6000_Breakout-10.xlsx
            i6000_Breakout-10_obs_trajs.7z
            i7000_PongNoFrameskip-1.xlsx
            i7000_PongNoFrameskip-1_obs_trajs.7z
            i7000_PongNoFrameskip-2.xlsx
            i7000_PongNoFrameskip-2_obs_trajs.7z
            ...
            i7000_PongNoFrameskip-10.xlsx
            i7000_PongNoFrameskip-10_obs_trajs.7z
        trained_models/
            Acrobot_v1__PPO.zip
            Breakout_v4__A2C.zip
            CartPole_v1__PPO.zip
            FrozenLake_v1__PPO.zip
            MountainCar_v0_DQN.zip
            PongNoFrameskip_v0_A2C.zip
            Taxi_v3__PPO.zip
            trained models origins.txt
        p02_main.py
        p02_traj_factory.py
    requirements.txt
    README.md
```

## 🚀 Usage
To generate new benchmark bundle for one of the environments specified above, do the following steps:
1. Select the environment you wish to generate trajectories for by copying its name (rows 12-18) in the [`./p02_traj_factory/p02_main.py`](./p02_traj_factory/p02_main.py) file. Below is a code snippet of the file:
```python
10    # available arguments:
11    #
12    #           "i1000_Acrobot.json"
13    #           "i2000_CartPole.json"
14    #           "i3000_MountainCar.json"
15    #           "i4000_Taxi.json"
16    #           "i5000_FrozenLake.json"
17    #           "i6000_Breakout.json"
18    #           "i7000_PongNoFrameskip.json"
19    #
20
21    filename = "i1000_Acrobot.json"
```
2. Go to the input file for the selected environment in the directory [`./p02_traj_factory/inputs`](./p02_traj_factory/inputs), and set the input parameters accordingly. Refer to the Excel structure for elaboration on the different input options.
3. Run [`./p02_traj_factory/p02_main.py`](./p02_traj_factory/p02_main.py)


## 📖 Citation

If you use this project or its benchmark datasets in your research, please cite:

```
@misc{tbd,
  title={Diagnosing Faults in Deep Reinforcement Learning Policy-Guided Systems: Settings and Benchmarks},
  author={tbd},
  year={2025},
  note={tbd},
  url={https://github.com/tbd/tbd}
}
```

## 👤 Authors

**TBD**  
A group of great people, will be revealed upon acception.

## 📄 License

This project is licensed under the CC BY-NC 4.0 License. See the [LICENSE](./LICENSE) file for details.