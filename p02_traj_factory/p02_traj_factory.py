import json
from datetime import datetime
import numpy as np
import xlsxwriter
import base64
import pickle

from common.fault_mode_generators import FaultModeGeneratorDiscrete
from common.executor import execute


def read_json_data(params_file):
    with open(params_file, 'r') as file:
        json_data = json.load(file)
    return json_data


# separating trajectory to actions and states
def separate_trajectory(trajectory_execution):
    registered_actions = []
    observations = []
    for i in range(len(trajectory_execution)):
        if i % 2 == 1:
            registered_actions.append(trajectory_execution[i])
        else:
            observations.append(trajectory_execution[i])
    if len(registered_actions) == len(observations):
        registered_actions = registered_actions[:-1]
    return registered_actions, observations


def generate_trajectory(domain_name,
                        model_name,
                        policy_type,
                        seed,
                        execution_fault_mode,
                        fault_probability):
    # ### initialize fault model generator
    fault_mode_generator = FaultModeGeneratorDiscrete()

    # ### execute to get trajectory
    print(f'executing with fault mode: {execution_fault_mode}\n========================================================================================')
    trajectory_execution = []
    faulty_actions_indices = []
    num_tries = 1
    while len(faulty_actions_indices) == 0:
        if num_tries > 100:
            raise ValueError('Tried too hard but didnt get a faulty traj.')
        print(f"try {num_tries}")
        trajectory_execution, faulty_actions_indices = execute(domain_name,
                                                               model_name,
                                                               policy_type,
                                                               seed,
                                                               execution_fault_mode,
                                                               fault_probability,
                                                               fault_mode_generator)
        num_tries += 1

    # ### separating trajectory to actions and states
    registered_actions, observations = separate_trajectory(trajectory_execution)

    # ### make sure the last faulty action was not an action that was removed
    if faulty_actions_indices[-1] > len(registered_actions):
        faulty_actions_indices = faulty_actions_indices[:-1]

    return registered_actions, faulty_actions_indices, observations, trajectory_execution


def prepare_record(domain_name, model_name, policy_types, seeds, modelled_fault_modes, fault_probabilities, instances,
                   policy_type, seed, execution_fault_mode, fault_probability, instance,
                   registered_actions, faulty_actions_indices, observations, trajectory_execution):
    record = {
        "domain_name": domain_name,
        "model_name": model_name,
        "policy_types": policy_types,
        "seeds": seeds,
        "fault_probabilities": fault_probabilities,
        "instances": instances,
        "modelled_fault_modes": modelled_fault_modes,
        "policy_type": policy_type,
        "seed": seed,
        "execution_fault_mode": execution_fault_mode,
        "fault_probability": fault_probability,
        "instance": instance,
        "registered_actions": registered_actions,
        "faulty_actions_indices": faulty_actions_indices,
        "observations": observations,
        "trajectory_execution": trajectory_execution
    }
    return record


def full_ndarray_list_to_string(array_list):
    # Set printing options to avoid clipping
    np.set_printoptions(threshold=np.inf, linewidth=np.inf, suppress=False)

    strings = []
    for i, arr in enumerate(array_list):
        if isinstance(arr, int):
            strings.append(f"{arr}")
        else:
            arr_str = np.array2string(arr)
            strings.append(f"{arr_str}")

    joined = ',\n'.join(strings)
    ret = f'[{joined}]'
    return ret

def write_records_to_excel(records, filename):
    columns = [
        {'header': '01_f_domain_name'},
        {'header': '02_f_model_name'},
        {'header': '03_f_policy_types'},
        {'header': '04_f_seeds'},
        {'header': '05_f_modelled_fault_modes'},
        {'header': '06_f_fault_probabilities'},
        {'header': '07_f_instances'},
        {'header': '08_policy_type'},
        {'header': '09_i_seed'},
        {'header': '10_i_execution_fault_mode'},
        {'header': '11_i_fault_probability'},
        {'header': '12_i_instance'},
        {'header': '13_O_registered_actions'},
        {'header': '14_O_faulty_actions_indices'},
        {'header': '15_O_observations'},
        {'header': '16_O_trajectory_execution'},
        {'header': '17_O_num_registered_actions'},
        {'header': '18_O_num_faulty_actions'},
        {'header': '19_O_num_observations_ie_exec_length'},
    ]
    rows = []
    for i in range(len(records)):
        record_i = records[i]
        print(f"rec {i}: {record_i['policy_type']}_{record_i['seed']}_{record_i['execution_fault_mode']}_{float(record_i['fault_probability'])}_{record_i['instance']}")
        np.set_printoptions(threshold=np.inf, linewidth=np.inf, suppress=False)
        if record_i['domain_name'] in ['Breakout_v4', 'PongNoFrameskip_v0']:
            obs_string = full_ndarray_list_to_string(record_i['observations'])
            with open(f"outputs/{filename}_{record_i['policy_type']}_{record_i['seed']}_{record_i['execution_fault_mode']}_{float(record_i['fault_probability'])}_{record_i['instance']}_obs.txt", "w") as f:
                f.write(obs_string)
            traj_string = full_ndarray_list_to_string(record_i['trajectory_execution'])
            with open(f"outputs/{filename}_{record_i['policy_type']}_{record_i['seed']}_{record_i['execution_fault_mode']}_{float(record_i['fault_probability'])}_{record_i['instance']}_traj.txt", "w") as f:
                f.write(traj_string)
        row = [
            record_i['domain_name'],                                    # 01_f_domain_name
            record_i['model_name'],                                     # 02_f_model_name
            str(record_i['policy_types']),                              # 03_f_policy_types
            str(record_i['seeds']),                                     # 04_f_seeds
            str(record_i['modelled_fault_modes']),                      # 05_f_modelled_fault_modes
            str(record_i['fault_probabilities']),                       # 06_f_fault_probabilities
            str(record_i['instances']),                                 # 07_f_instances
            record_i['policy_type'],                                    # 08_f_policy_type
            record_i['seed'],                                           # 09_i_seed
            record_i['execution_fault_mode'],                           # 10_i_execution_fault_mode
            float(record_i['fault_probability']),                       # 11_i_fault_probability
            record_i['instance'],                                       # 12_i_instance
            str(record_i['registered_actions']),                        # 13_O_registered_actions
            str(record_i['faulty_actions_indices']),                    # 14_O_faulty_actions_indices
            str([str(obs) for obs in record_i['observations']]) if record_i['domain_name'] not in ['Breakout_v4', 'PongNoFrameskip_v0'] else "IN ZIP FILE",        # 15_O_observations
            str([str(m) for m in record_i['trajectory_execution']]) if record_i['domain_name'] not in ['Breakout_v4', 'PongNoFrameskip_v0'] else "IN ZIP FILE",    # 16_O_trajectory_execution
            len(record_i['registered_actions']),                        # 17_O_num_registered_actions
            len(record_i['faulty_actions_indices']),                    # 18_O_num_faulty_actions
            len(record_i['observations'])                               # 19_O_num_observations_ie_exec_length
        ]
        rows.append(row)
    workbook = xlsxwriter.Workbook(f"outputs/{filename}.xlsx")
    worksheet = workbook.add_worksheet('results')
    worksheet.add_table(0, 0, len(rows), len(columns) - 1, {'data': rows, 'columns': columns})
    workbook.close()


def generate_trajectories(filename):
    # ### parameters dictionary
    param_dict = read_json_data(f"inputs/{filename}")

    # ### prepare the records database to be written to the Excel file
    records = []

    # ### the domain name of this experiment (each experiment file has only one associated domain)
    domain_name = param_dict['domain_name']

    # ### the machine learning model name of this experiment (each experiment file has one associated ml model)
    model_name = param_dict['model_name']

    # ### the policy types
    policy_types = param_dict['policy_types']

    # ### the environment start point seeds
    seeds = param_dict['seeds']

    # ### the candidate fault modes that one of them is responsible for the faulty execution
    modelled_fault_modes = param_dict['modelled_fault_modes']

    # ### the possible fault probabilities
    fault_probabilities = param_dict['fault_probabilities']

    # ### the experiment instance numbers
    instances = param_dict['instances']

    total_instances_number = len(policy_types) * len(seeds) * len(modelled_fault_modes) * len(fault_probabilities) * len(instances)
    current_instance_number = 1
    start_time = datetime.now()

    # ### run the trajectory generation loop
    for policy_type_i, policy_type in enumerate(policy_types):
        for seed_i, seed in enumerate(seeds):
            for execution_fault_mode_i, execution_fault_mode in enumerate(modelled_fault_modes):
                for fault_probability_i, fault_probability in enumerate(fault_probabilities):
                    for instance_i, instance in enumerate(instances):
                        # ### create the faulty trajectory
                        registered_actions, faulty_actions_indices, observations, trajectory_execution = generate_trajectory(domain_name,
                                                                                                                             model_name,
                                                                                                                             policy_type,
                                                                                                                             seed,
                                                                                                                             execution_fault_mode,
                                                                                                                             fault_probability)
                        # ### logging
                        now = datetime.now()
                        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                        elapsed_time = now - start_time
                        hours, remainder = divmod(elapsed_time.total_seconds(), 3600)
                        minutes, seconds = divmod(remainder, 60)
                        print(f"{dt_string}: {current_instance_number}/{total_instances_number}")
                        print(f"elapsed time: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")
                        print(f"policy_type: {policy_type}, seed: {seed}, execution_fault_mode: {execution_fault_mode}, fault_probability: {fault_probability}, instance: {instance}")
                        print(f"registered actions: {str(registered_actions)}")
                        print(f"number of  actions: {len(registered_actions)}")

                        # ### preparing record for writing to Excel file
                        record = prepare_record(domain_name, model_name, policy_types, seeds, modelled_fault_modes, fault_probabilities, instances,
                                                policy_type, seed, execution_fault_mode, fault_probability, instance,
                                                registered_actions, faulty_actions_indices, observations, trajectory_execution)
                        records.append(record)

                        print(f'\n')
                        current_instance_number += 1

    # ### write records to an Excel file
    write_records_to_excel(records, filename.split(".")[0])

    print(9)
