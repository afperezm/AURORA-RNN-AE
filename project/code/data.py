import gym
import numpy as np
import pandas
import torch

from project.code.lunar_lander_map_elites_complex_model import simulate
from torch.utils.data import Dataset
from tqdm import tqdm


class VecDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data).float()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx, :]


class VecSeqDataset(Dataset):
    def __init__(self, data, lengths):
        self.data = torch.tensor(data).float()
        self.lengths = torch.tensor(lengths).int()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx, :], self.lengths[idx]


def generate_test_data(aggregate=True, flatten=True):

    env_seed = 1339

    env = gym.make("LunarLander-v2")


    df = pandas.read_pickle("../../results_stochastic/MAP-ELITES_LUNDAR-LANDER_221118-031234/archive_1000.pkl")

    # archive = GridArchive(
    #     [50, 50],
    #     [(-1.0, 1.0), (-1.0, 1.0)],
    # )

    solutions = df.to_numpy()[:, 5:]
    # objectives = df.to_numpy()[:, 4:5]
    # behavior_descriptors = df.to_numpy()[:, 2:4]

    # archive.initialize(solutions.shape[1])
    #
    # for idx in tqdm(range(solutions.shape[0])):
    #     archive.add(solutions[idx], objectives[idx], behavior_descriptors[idx])

    states_list = []

    for solution in tqdm(solutions):
        result = simulate(env, solution, env_seed)
        states_list.append(result[3])

    env.close()

    max_length = max([states.shape[0] for states in states_list])

    states_array = np.vstack(states_list)

    min_values = np.min(states_array, axis=0)
    max_values = np.max(states_array, axis=0)

    states_array_scaled = []

    for states in states_list:
        if aggregate:
            states_array_scaled.append(states.mean(axis=0))
        else:
            states = (states - min_values) / (max_values - min_values)
            states_array_scaled.append(np.pad(states, pad_width=[(0, max_length - len(states)), (0, 0)]))

    states_array_scaled = np.array(states_array_scaled)

    if aggregate is False and flatten is True:
        states_array_scaled = states_array_scaled.reshape(states_array_scaled.shape[0], -1)

    return states_array_scaled

    # return solutions, objectives, states_list, states_array_scaled
