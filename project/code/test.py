import argparse
import cv2
from collections import OrderedDict

import copy
import gym
import json
import os
import numpy as np
import pandas
import time
import torch

# from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
from project.code.networks import PolicyNet
from ribs.archives import GridArchive
from ribs.visualize import grid_archive_heatmap
from tqdm import tqdm

DEVICE = None
FPS = 25
NUM_ACTIONS = 4
NUM_HIDDEN = 64
OBSERVATION_SIZE = 8
PARAMS = None
STATS_WINDOW = 100


def get_episode_reward(env, policy, seed):

    # Reset environment
    if seed is not None:
        state, info = env.reset(seed=seed)
    else:
        state, info = env.reset()

    # Set model to evaluation mode
    policy.eval()

    # Initialize counters
    episode_reward = 0.0
    screens_list = []

    # Evaluation loop
    while True:

        # Sample current policy
        state_tensor = torch.FloatTensor(state)
        action = torch.distributions.Categorical(policy(state_tensor)).sample().item()

        # Take action on current state
        state, reward, terminated, truncated, info = env.step(action)

        # Render environment
        screen = env.render()

        # Update counters
        episode_reward += reward
        screens_list.append(screen)

        # Break loop if game is done
        if terminated or truncated:
            break

    screens_array = np.array(screens_list)

    return episode_reward, screens_array


def main():

    archive_name = PARAMS.archive_name
    archive_dir = PARAMS.archive_dir
    # summary_dir = PARAMS.summary_dir

    test_exp_name = f"TEST-{archive_name}_{time.strftime('%y%m%d')}-{time.strftime('%H%M%S')}"

    if not os.path.exists(os.path.join(archive_dir, test_exp_name)):
        os.makedirs(os.path.join(archive_dir, test_exp_name))

    # if not os.path.exists(os.path.join(summary_dir, test_exp_name)):
    #     os.makedirs(os.path.join(summary_dir, test_exp_name))

    # Dump program arguments
    with open(os.path.join(archive_dir, test_exp_name, "params.json"), "w") as f:
        json.dump(vars(PARAMS), f)

    # # Initialize tensorboard summary writer
    # summary_writer = SummaryWriter(log_dir=os.path.join(summary_dir, f"{test_exp_name}"))

    # Create game environment
    env = gym.make("LunarLander-v2", render_mode="rgb_array")

    # Retrieve game constants
    action_dim = env.action_space.n
    state_dim = env.observation_space.shape[0]

    assert action_dim == NUM_ACTIONS
    assert state_dim == OBSERVATION_SIZE

    archive = GridArchive(
        [50, 50],  # 10 bins in each dimension.
        [(-1.0, 1.0), (-1.0, 1.0)],  # (0, 10) for foot-right and (0, 10) for foot-left.
    )

    df = pandas.read_pickle(os.path.join(archive_dir, f'{archive_name}.pkl'))

    sols = df.to_numpy()[:, 5:5 + 4996]
    objs = df.to_numpy()[:, 4:5]
    bcs = df.to_numpy()[:, 2:4]

    archive.initialize(sols.shape[1])

    for idx in tqdm(range(sols.shape[0])):
        archive.add(sols[idx], objs[idx], bcs[idx])

    grid_archive_heatmap(archive, vmin=-300, vmax=300)
    plt.show()

    elite = archive.elite_with_behavior([-0.5, 0.5])

    x = copy.deepcopy(elite.sol)

    # archive = np.load(f'{archive_dir}/{archive_name}')
    #
    # archive_rewards = []
    #
    # for archive_idx in range(archive.shape[0]):
    #     print(f'{archive_idx + 1}/{archive.shape[0]}')
    # # for archive_idx in range(archive.shape[0]):
    #     total_reward, behavior_descriptor = [], []
    #     for _ in range(10):
    #         result = evaluate_walker(archive[archive_idx])
    #         total_reward.append(result[0])
    #     archive_rewards.append(np.mean(total_reward))
    #
    # max_actor_idx = np.argmax(archive_rewards)
    #
    # x = archive[max_actor_idx]

    # Build state dictionary
    state_dict = OrderedDict()
    state_dict['fc1.0.weight'] = torch.tensor(x[0:NUM_HIDDEN * OBSERVATION_SIZE].reshape(NUM_HIDDEN, OBSERVATION_SIZE))
    state_dict['fc1.0.bias'] = torch.tensor(x[NUM_HIDDEN * OBSERVATION_SIZE:NUM_HIDDEN * OBSERVATION_SIZE + NUM_HIDDEN])
    state_dict['fc2.0.weight'] = torch.tensor(x[NUM_HIDDEN * OBSERVATION_SIZE + NUM_HIDDEN:NUM_HIDDEN * OBSERVATION_SIZE + NUM_HIDDEN + NUM_HIDDEN * NUM_HIDDEN].reshape(NUM_HIDDEN, NUM_HIDDEN))
    state_dict['fc2.0.bias'] = torch.tensor(x[NUM_HIDDEN * OBSERVATION_SIZE + NUM_HIDDEN + NUM_HIDDEN * NUM_HIDDEN:NUM_HIDDEN * OBSERVATION_SIZE + NUM_HIDDEN + NUM_HIDDEN * NUM_HIDDEN + NUM_HIDDEN])
    state_dict['fc3.0.weight'] = torch.tensor(x[NUM_HIDDEN * OBSERVATION_SIZE + NUM_HIDDEN + NUM_HIDDEN * NUM_HIDDEN + NUM_HIDDEN:NUM_HIDDEN * OBSERVATION_SIZE + NUM_HIDDEN + NUM_HIDDEN * NUM_HIDDEN + NUM_HIDDEN + NUM_HIDDEN * NUM_ACTIONS].reshape(NUM_ACTIONS, NUM_HIDDEN))
    state_dict['fc3.0.bias'] = torch.tensor(x[NUM_HIDDEN * OBSERVATION_SIZE + NUM_HIDDEN + NUM_HIDDEN * NUM_HIDDEN + NUM_HIDDEN + NUM_HIDDEN * NUM_ACTIONS:NUM_HIDDEN * OBSERVATION_SIZE + NUM_HIDDEN + NUM_HIDDEN * NUM_HIDDEN + NUM_HIDDEN + NUM_HIDDEN * NUM_ACTIONS + NUM_ACTIONS])

    # Create new policy network
    policy = PolicyNet(OBSERVATION_SIZE, NUM_ACTIONS, n_hidden=NUM_HIDDEN)

    # Load state dictionary
    policy.load_state_dict(state_dict)

    # # Initialize lists
    # stats_rewards = []

    T = 1

    for _ in tqdm(range(1, T + 1)):

        # Play episode
        episode_reward, video_array = get_episode_reward(env, policy, 0)
        # episode_reward, video_array = get_episode_reward(env, policy, None)

        # # Append episode reward to the stats list
        # stats_rewards.append(episode_reward)

        video_array = video_array.transpose((0, 3, 1, 2))

        frame_count, channels, height, width = video_array.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        video = cv2.VideoWriter(os.path.join(archive_dir, f'{archive_name}.mp4'), fourcc, float(FPS), (width, height))

        for frame in range(frame_count):
            video.write(video_array[frame].transpose((1, 2, 0)))

        video.release()

        # summary_writer.add_scalar("test_reward", episode_reward, episode_idx)
        # summary_writer.add_scalar("average_test_reward", np.mean(stats_rewards[-STATS_WINDOW:], axis=0), episode_idx)
        # summary_writer.flush()

    env.close()
    # summary_writer.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive_dir", help="Archives directory", required=True)
    parser.add_argument('--archive_name', help="Archive name", required=True)
    # parser.add_argument("--summary_dir", help="Summary directory", required=True)
    return parser.parse_args()


if __name__ == '__main__':
    PARAMS = parse_args()
    main()
