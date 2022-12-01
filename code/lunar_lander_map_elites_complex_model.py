import argparse
import gym
import json
import math
import numpy as np
import os
import time
import torch

from collections import OrderedDict
from ribs.archives import GridArchive
from ribs.emitters import GaussianEmitter
from ribs.optimizers import Optimizer
from tqdm import tqdm

from networks import PolicyNet

NUM_ACTIONS = 4
NUM_HIDDEN = 64
NUM_EMITTERS = 5
OBSERVATION_SIZE = 8
PARAMS = None


def random_layer_params(out_features, in_features):

    weight_param = torch.nn.parameter.Parameter(torch.zeros([out_features, in_features]), requires_grad=False)
    bias_param = torch.nn.parameter.Parameter(torch.zeros([out_features]), requires_grad=False)

    weight = torch.nn.init.kaiming_uniform_(weight_param, a=math.sqrt(5))

    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weight_param)
    bound = 1 / math.sqrt(fan_in)

    bias = torch.nn.init.uniform_(bias_param, -bound, bound)

    return weight.flatten(), bias.flatten()


def random_params(dim_x):

    fc1_weight, fc1_bias = random_layer_params(NUM_HIDDEN, OBSERVATION_SIZE)
    fc2_weight, fc2_bias = random_layer_params(NUM_HIDDEN, NUM_HIDDEN)
    fc3_weight, fc3_bias = random_layer_params(NUM_ACTIONS, NUM_HIDDEN)

    x = torch.hstack([fc1_weight, fc1_bias, fc2_weight, fc2_bias, fc3_weight, fc3_bias]).numpy()

    assert dim_x == x.shape[0]

    return x


def simulate(env, x, seed=None):
    """Simulates the lunar lander model.

    Args:
        env (gym.Env): A copy of the lunar lander environment.
        x (np.ndarray): The array of parameters for the linear policy.
        seed (int): The seed for the environment.
    Returns:
        total_reward (float): The reward accrued by the lander throughout its
            trajectory.
        impact_x_pos (float): The x position of the lander when it touches the
            ground for the first time.
        impact_y_vel (float): The y velocity of the lander when it touches the
            ground for the first time.
        sates_array (array): Array of observed states during simulation.
    """

    # Build state dictionary
    state_dict = OrderedDict()
    state_dict['fc1.0.weight'] = torch.tensor(x[0:NUM_HIDDEN * OBSERVATION_SIZE].reshape(NUM_HIDDEN, OBSERVATION_SIZE))
    state_dict['fc1.0.bias'] = torch.tensor(x[NUM_HIDDEN * OBSERVATION_SIZE:NUM_HIDDEN * OBSERVATION_SIZE + NUM_HIDDEN])
    state_dict['fc2.0.weight'] = torch.tensor(x[NUM_HIDDEN * OBSERVATION_SIZE + NUM_HIDDEN:NUM_HIDDEN * OBSERVATION_SIZE + NUM_HIDDEN + NUM_HIDDEN * NUM_HIDDEN].reshape(NUM_HIDDEN, NUM_HIDDEN))
    state_dict['fc2.0.bias'] = torch.tensor(x[NUM_HIDDEN * OBSERVATION_SIZE + NUM_HIDDEN + NUM_HIDDEN * NUM_HIDDEN:NUM_HIDDEN * OBSERVATION_SIZE + NUM_HIDDEN + NUM_HIDDEN * NUM_HIDDEN + NUM_HIDDEN])
    state_dict['fc3.0.weight'] = torch.tensor(x[NUM_HIDDEN * OBSERVATION_SIZE + NUM_HIDDEN + NUM_HIDDEN * NUM_HIDDEN + NUM_HIDDEN:NUM_HIDDEN * OBSERVATION_SIZE + NUM_HIDDEN + NUM_HIDDEN * NUM_HIDDEN + NUM_HIDDEN + NUM_HIDDEN * NUM_ACTIONS].reshape(NUM_ACTIONS, NUM_HIDDEN))
    state_dict['fc3.0.bias'] = torch.tensor(x[NUM_HIDDEN * OBSERVATION_SIZE + NUM_HIDDEN + NUM_HIDDEN * NUM_HIDDEN + NUM_HIDDEN + NUM_HIDDEN * NUM_ACTIONS:NUM_HIDDEN * OBSERVATION_SIZE + NUM_HIDDEN + NUM_HIDDEN * NUM_HIDDEN + NUM_HIDDEN + NUM_HIDDEN * NUM_ACTIONS + NUM_ACTIONS])

    # Create new policy
    model = PolicyNet(OBSERVATION_SIZE, NUM_ACTIONS, n_hidden=NUM_HIDDEN)

    # Load state dictionary
    model.load_state_dict(state_dict)

    # Reset environment
    if seed is not None:
        state, info = env.reset(seed=seed)
    else:
        state, info = env.reset()

    # Set model to evaluation mode
    model.eval()

    # Initialize counters
    episode_reward = 0.0
    impact_x_pos = None
    impact_y_vel = None
    all_y_velocities = []
    states_list = []

    # Evaluation loop
    while True:

        # Sample current policy
        state_tensor = torch.FloatTensor(state)
        # action = torch.distributions.Categorical(model(state_tensor)).sample().item()
        action = torch.argmax(model(state_tensor)).item()

        # Take action on current state
        state, reward, terminated, truncated, info = env.step(action)

        # Refer to the definition of state here:
        # https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py#L306
        x_pos = state[0]
        y_vel = state[3]
        leg0_touch = bool(state[6])
        leg1_touch = bool(state[7])

        # Update counters
        episode_reward += reward
        all_y_velocities.append(y_vel)
        states_list.append(state)

        # Check if the lunar lander is impacting for the first time.
        if impact_x_pos is None and (leg0_touch or leg1_touch):
            impact_x_pos = x_pos
            impact_y_vel = y_vel

        if terminated or truncated:
            break

    # If the lunar lander did not land, set the x-pos to the one from the final
    # time step, and set the y-vel to the max y-vel (we use min since the lander
    # goes down).
    if impact_x_pos is None:
        impact_x_pos = x_pos
        impact_y_vel = min(all_y_velocities)

    behavior_descriptor = (impact_x_pos, impact_y_vel)

    states_array = np.array(states_list)

    return episode_reward, behavior_descriptor[0], behavior_descriptor[1], states_array


def display_video(model, seed):
    """Displays a video of the model in the environment."""

    # Monitor records a video of the environment.
    video_env = gym.wrappers.Monitor(
        gym.make("LunarLander-v2"),
        "videos",  # Video directory.
        force=True,  # Overwrite existing videos.
        video_callable=lambda idx: True,  # Make all episodes be recorded.
    )
    simulate(video_env, model, seed)
    video_env.close()  # Save video.


def main():

    output_dir = PARAMS.output_dir
    num_generations = PARAMS.num_generations
    sigma = PARAMS.sigma
    log_freq = PARAMS.log_freq
    seed = PARAMS.seed

    exp_name = f"MAP-ELITES_LUNAR-LANDER_{time.strftime('%y%m%d')}-{time.strftime('%H%M%S')}"

    # Create experiment directory
    if not os.path.exists(os.path.join(output_dir, exp_name)):
        os.makedirs(os.path.join(output_dir, exp_name))

    # Dump program arguments
    with open(os.path.join(output_dir, exp_name, "params.json"), "w") as f:
        json.dump(vars(PARAMS), f)

    env = gym.make("LunarLander-v2")

    archive = GridArchive(
        dims=[50, 50],
        ranges=[(-1.0, 1.0), (-3.0, 0.0)],
    )

    # Fix initial seed
    torch.manual_seed(seed)

    initial_model = random_params(4996)

    emitters = [
        GaussianEmitter(
            archive,
            initial_model.flatten(),
            sigma,
            batch_size=64,
            seed=s
        ) for s in range(seed, seed + NUM_EMITTERS)
    ]

    optimizer = Optimizer(archive, emitters)

    start_time = time.time()
    total_itrs = num_generations

    for itr in tqdm(range(1, total_itrs + 1)):
        # Request models from the optimizer
        solutions = optimizer.ask()
        # Evaluate the models and record the objectives and BCs
        objectives, descriptors = [], []
        for solution in solutions:
            obj, impact_x_pos, impact_y_vel, _ = simulate(env, solution, seed=seed)
            objectives.append(obj)
            descriptors.append([impact_x_pos, impact_y_vel])
        # Send the results back to the optimizer
        optimizer.tell(objectives, descriptors)
        # Logging
        if itr % log_freq == 0:
            elapsed_time = time.time() - start_time
            print(f"> {itr} itrs completed after {elapsed_time:.2f} s")
            print(f"  - Archive Size: {len(archive)}")
            print(f"  - Max Score: {archive.stats.obj_max}")
            print(f"  - Mean Score: {archive.stats.obj_mean}")
            df = archive.as_pandas()
            df.to_pickle(os.path.join(output_dir, exp_name, f"archive_{itr:04}.pkl"))


def parse_args():
    parser = argparse.ArgumentParser("Trainer of an agent to play OpenAy Gym Lunar Lander using MAP-Elites")
    parser.add_argument("--output_dir", help="Directory to store generated archives", required=True)
    parser.add_argument("--num_generations", default=1000, help="Max number of generations", type=int)
    parser.add_argument("--sigma", help="Mutation rate", type=float, default=0.5)
    parser.add_argument('--log_freq', type=int, default=25, help='Frequency for logging and saving training results')
    parser.add_argument("--seed", help="Master seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    PARAMS = parse_args()
    main()
