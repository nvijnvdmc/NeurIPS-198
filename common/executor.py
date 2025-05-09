import random

import numpy as np
import gym
import ale_py
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

from common.rl_models import models
from common.state_refiners import refiners
from common.wrappers import wrappers
from common.consts import RENDER_MODE, MAX_EXEC_LEN, DEBUG_PRINT


def execute(domain_name,
            model_name,
            policy_type,
            seed,
            execution_fault_mode,
            fault_probability,
            fault_mode_generator):
    if domain_name == 'Breakout_v4':
        trajectory, faulty_actions_indices = execute_breakout_v4(domain_name, model_name, policy_type, seed, execution_fault_mode, fault_probability, fault_mode_generator)
    elif domain_name == 'PongNoFrameskip_v0':
        trajectory, faulty_actions_indices = execute_pongnoframeskip_v0(domain_name, model_name, policy_type, seed, execution_fault_mode, fault_probability, fault_mode_generator)
    else:
        trajectory, faulty_actions_indices = execute_gym(domain_name, model_name, policy_type, seed, execution_fault_mode, fault_probability, fault_mode_generator)

    return trajectory, faulty_actions_indices


def execute_gym(domain_name,
                model_name,
                policy_type,
                seed,
                execution_fault_mode,
                fault_probability,
                fault_mode_generator):
    # initialize environment
    env = wrappers[domain_name](gym.make(domain_name.replace('_', '-'), render_mode=RENDER_MODE))
    initial_obs, _ = env.reset(seed=seed)
    # print(f'initial observation: {initial_obs.tolist()}')

    # load trained model
    model_path = f"trained_models/{domain_name}__{model_name}.zip"
    model = models[model_name].load(model_path, env=env)

    # initialize execution fault mode
    execution_fault_mode_function = fault_mode_generator.generate_fault_mode_function(execution_fault_mode)

    # initializing empty trajectory
    trajectory = []

    faulty_actions_indices = []
    action_number = 1
    done = False
    exec_len = 1
    obs, _ = env.reset()
    while not done and exec_len < MAX_EXEC_LEN:
        trajectory.append(obs)
        if DEBUG_PRINT:
            print(f'a#:{action_number} [PREVOBS]: {obs.tolist() if not isinstance(obs, int) else obs}')
        action, _ = model.predict(refiners[domain_name](obs), deterministic=policy_type)
        action = int(action)
        trajectory.append(action)
        if random.random() < fault_probability:
            faulty_action = execution_fault_mode_function(action)
        else:
            faulty_action = action
        if faulty_action != action:
            faulty_actions_indices.append(action_number)
        if DEBUG_PRINT:
            if action != faulty_action:
                print(f'a#:{action_number} [FAILURE] - planned: {action}, actual: {faulty_action}')
            else:
                print(f'a#:{action_number} [SUCCESS] - planned: {action}, actual: {faulty_action}')
        obs, reward, done, trunc, info = env.step(faulty_action)
        if DEBUG_PRINT:
            print(f'a#:{action_number} [NEXTOBS]: {obs.tolist() if not isinstance(obs, int) else obs}\n')
        action_number += 1
        exec_len += 1

    trajectory.append(obs)
    env.close()

    return trajectory, faulty_actions_indices


def execute_breakout_v4(domain_name,
                        model_name,
                        policy_type,
                        seed,
                        execution_fault_mode,
                        fault_probability,
                        fault_mode_generator):
    # initialize environment
    env = make_atari_env(domain_name.replace('_', '-'), n_envs=1, seed=seed)
    initial_obs = env.reset()
    # print(f'initial observation: {initial_obs.tolist()}')

    # load trained model
    model_path = f"trained_models/{domain_name}__{model_name}.zip"
    model = models[model_name].load(model_path, env=env)

    # initialize execution fault mode
    execution_fault_mode_function = fault_mode_generator.generate_fault_mode_function(execution_fault_mode)

    # initializing empty trajectory
    trajectory = []

    faulty_actions_indices = []
    action_number = 1
    done = False
    exec_len = 1
    obs = env.reset()
    while not done and exec_len < MAX_EXEC_LEN:
        trajectory.append(obs)
        if DEBUG_PRINT:
            print(f'a#:{action_number} [PREVOBS]: {obs.tolist() if not isinstance(obs, int) else obs}')
        action, _ = model.predict(obs, deterministic=policy_type)
        action = int(action[0])
        trajectory.append(action)
        if random.random() < fault_probability:
            faulty_action = execution_fault_mode_function(action)
        else:
            faulty_action = action
        if faulty_action != action:
            faulty_actions_indices.append(action_number)
        if DEBUG_PRINT:
            if action != faulty_action:
                print(f'a#:{action_number} [FAILURE] - planned: {action}, actual: {faulty_action}')
            else:
                print(f'a#:{action_number} [SUCCESS] - planned: {action}, actual: {faulty_action}')
        faulty_action = np.array([faulty_action])
        obs, reward, done, info = env.step(faulty_action)
        if DEBUG_PRINT:
            print(f'a#:{action_number} [NEXTOBS]: {obs.tolist() if not isinstance(obs, int) else obs}\n')
        action_number += 1
        exec_len += 1

    trajectory.append(obs)
    env.close()

    return trajectory, faulty_actions_indices


def execute_pongnoframeskip_v0(domain_name,
                               model_name,
                               policy_type,
                               seed,
                               execution_fault_mode,
                               fault_probability,
                               fault_mode_generator):
    # initialize environment
    env = make_atari_env(domain_name.replace('_', '-'), n_envs=1, seed=seed)
    env = VecFrameStack(env, n_stack=4)
    initial_obs = env.reset()
    # print(f'initial observation: {initial_obs.tolist()}')

    # load trained model
    model_path = f"trained_models/{domain_name}__{model_name}.zip"
    model = models[model_name].load(model_path, env=env)

    # initialize execution fault mode
    execution_fault_mode_function = fault_mode_generator.generate_fault_mode_function(execution_fault_mode)

    # initializing empty trajectory
    trajectory = []

    faulty_actions_indices = []
    action_number = 1
    done = False
    exec_len = 1
    obs = env.reset()
    while not done and exec_len < MAX_EXEC_LEN:
        trajectory.append(obs)
        if DEBUG_PRINT:
            print(f'a#:{action_number} [PREVOBS]: {obs.tolist() if not isinstance(obs, int) else obs}')
        action, _ = model.predict(obs, deterministic=policy_type)
        action = int(action[0])
        trajectory.append(action)
        if random.random() < fault_probability:
            faulty_action = execution_fault_mode_function(action)
        else:
            faulty_action = action
        if faulty_action != action:
            faulty_actions_indices.append(action_number)
        if DEBUG_PRINT:
            if action != faulty_action:
                print(f'a#:{action_number} [FAILURE] - planned: {action}, actual: {faulty_action}')
            else:
                print(f'a#:{action_number} [SUCCESS] - planned: {action}, actual: {faulty_action}')
        faulty_action = np.array([faulty_action])
        obs, reward, done, info = env.step(faulty_action)
        if DEBUG_PRINT:
            print(f'a#:{action_number} [NEXTOBS]: {obs.tolist() if not isinstance(obs, int) else obs}\n')
        action_number += 1
        exec_len += 1

    trajectory.append(obs)
    env.close()

    return trajectory, faulty_actions_indices
