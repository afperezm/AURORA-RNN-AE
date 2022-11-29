import argparse
import gym
import json
import math
import numpy as np
import os
import time
import torch

from collections import OrderedDict
from functools import partial
from ribs.archives import GridArchive
from ribs.emitters import GaussianEmitter
from ribs.optimizers import Optimizer
from skimage.exposure import exposure
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from project.code.ae import VecAE, VecDataset
from project.code.data import MinMaxPaddedScaler, MAX_LENGTH, VecSeqDataset
from project.code.networks import PolicyNet
# from project.code.rnn import VecRnnAE, VecRnnVAE
from project.code.rnn import VecRnnAE
from project.code.vae import VecVAE

DEVICE = None
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


def random_params():

    fc1_weight, fc1_bias = random_layer_params(NUM_HIDDEN, OBSERVATION_SIZE)
    fc2_weight, fc2_bias = random_layer_params(NUM_HIDDEN, NUM_HIDDEN)
    fc3_weight, fc3_bias = random_layer_params(NUM_ACTIONS, NUM_HIDDEN)

    x = torch.hstack([fc1_weight, fc1_bias, fc2_weight, fc2_bias, fc3_weight, fc3_bias]).numpy()

    return x.shape[0], x


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

    # Create new policy network
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
    states_list = []

    # Evaluation loop
    while True:

        # Sample current policy
        state_tensor = torch.FloatTensor(state)
        # action = torch.distributions.Categorical(model(state_tensor)).sample().item()
        action = torch.argmax(model(state_tensor)).item()

        # Take action on current state
        state, reward, terminated, truncated, info = env.step(action)

        # Update counters
        episode_reward += reward
        states_list.append(state)

        # Break loop if game is done
        if terminated or truncated:
            break

    states_array = np.array(states_list)

    return episode_reward, states_array


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


def train_ae(model, scaler, states, num_epochs, batch_size, learning_rate, ckpt_dir=None):

    # Scale states data
    states_scaled = scaler.fit_transform(states)

    # Load data set
    if len(states_scaled.shape) == 3:
        states_lengths = np.array([s.shape[0] for s in states])
        dataset = VecSeqDataset(states_scaled, states_lengths)
    else:
        dataset = VecDataset(states_scaled)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    # Train Model
    with torch.enable_grad():
        model.fit(data_loader=data_loader, num_epochs=num_epochs, learning_rate=learning_rate, checkpoint_dir=ckpt_dir)

    # model.train()
    #
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # loss_function = nn.MSELoss()
    #
    # device = next(model.parameters()).device
    #
    # for epoch in range(1, num_epochs + 1):
    #
    #     epoch_loss = 0.0
    #
    #     for batch_idx, data in enumerate(data_loader):
    #         data = data.to(device)
    #         # compute reconstructions
    #         # reduced_batch, decoded_batch = model.forward(data)
    #         decoded_batch = model.forward(data)
    #         # compute training reconstruction loss
    #         # batch_loss = loss_function(decoded_batch, reduced_batch)
    #         batch_loss = loss_function(decoded_batch, data)
    #         # reset the gradients back to zero
    #         optimizer.zero_grad()
    #         # compute accumulated gradients
    #         batch_loss.backward()
    #         # perform parameter update based on current gradients
    #         optimizer.step()
    #         # add the mini-batch training loss to epoch loss
    #         epoch_loss += batch_loss.item()
    #
    #     epoch_loss = epoch_loss / len(data_loader)
    #
    #     if (epoch % log_freq) == 0:
    #         print(f'Loss at Epoch {epoch}:\t{epoch_loss}')


def compute_behavior_descriptors(model, scaler, states_list, batch_size):

    device = next(model.parameters()).device

    states_array = scaler.transform(states_list)

    # Load data set
    if len(states_array.shape) == 3:
        states_lengths = np.array([s.shape[0] for s in states_list])
        dataset = VecSeqDataset(states_array, states_lengths)
    else:
        dataset = VecDataset(states_array)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    model.eval()

    bcs_list = []

    if len(states_array.shape) == 3:
        for data, data_lengths in data_loader:
            data = data.to(device)
            hidden1, hidden2 = model.init_hidden(len(data))
            z = model.encode(data, data_lengths, hidden1)
            # Pick hidden state of last element in sequence according to sequence length
            z = z.view(len(data), -1, 2)
            for idx, data_length in enumerate(data_lengths):
                bcs_list.append(z[idx, data_length - 1, :].tolist())
    else:
        for data in data_loader:
            data = data.to(device)
            z = model.encode(data).squeeze().detach().cpu().numpy()
            bcs_list.extend(z.tolist())

    return bcs_list


def main():

    train_mode = PARAMS.train_mode.upper()
    time_mode = PARAMS.time_mode.upper()
    encode_mode = PARAMS.encode_mode.upper()
    output_dir = PARAMS.output_dir
    num_generations = PARAMS.num_generations
    sigma = PARAMS.sigma
    num_epochs = PARAMS.num_epochs
    batch_size = PARAMS.batch_size
    learning_rate = PARAMS.learning_rate
    log_freq = PARAMS.log_freq
    seed = PARAMS.seed

    exp_name = f"AURORA-{train_mode}_{time_mode}-{encode_mode}_LUNAR-LANDER_{time.strftime('%y%m%d')}-{time.strftime('%H%M%S')}"

    # Create experiment directory
    if not os.path.exists(os.path.join(output_dir, exp_name)):
        os.makedirs(os.path.join(output_dir, exp_name))

    # Dump program arguments
    with open(os.path.join(output_dir, exp_name, "params.json"), "w") as f:
        json.dump(vars(PARAMS), f)

    env = gym.make("LunarLander-v2")

    archive = GridArchive(
        dims=[50, 50],
        ranges=[(0.0, 1.0), (0.0, 1.0)],
    )

    # Fix initial seed
    torch.manual_seed(seed)

    num_params, initial_model = random_params()

    emitters = [
        GaussianEmitter(
            archive,
            initial_model.flatten(),
            sigma,
            batch_size=64,
            seed=s
        ) for s in range(seed, seed + NUM_EMITTERS)
    ]

    # States data scaler
    if time_mode == "AVG":
        sts_scaler = MinMaxPaddedScaler(averaging=True)
    elif time_mode == "STACK":
        sts_scaler = MinMaxPaddedScaler(averaging=False, stacking=True)
    elif time_mode == "PAD":
        sts_scaler = MinMaxPaddedScaler(averaging=False, stacking=False)
    else:
        raise ValueError("Unknown `time_mode` selected.")
    bcs_scaler = MinMaxScaler()  # Behavior descriptors scaler

    if time_mode == "STACK":
        input_space = MAX_LENGTH * OBSERVATION_SIZE
    else:
        input_space = OBSERVATION_SIZE

    if encode_mode == "AE":
        model = VecAE(input_space, 2).to(DEVICE)
    elif encode_mode == "VAE":
        model = VecVAE(input_space, 2).to(DEVICE)
    elif encode_mode == "RNN-AE":
        model = VecRnnAE(input_space, input_space, 2, 1).to(DEVICE)
    # elif encode_mode == "RNN-VAE":
    #     model = VecRnnVAE(input_space, input_space, 2, 1).to(DEVICE)
    else:
        raise ValueError("Unknown `encode_mode` selected.")

    optimizer = Optimizer(archive, emitters)

    start_time = time.time()
    total_itrs = num_generations

    for itr in tqdm(range(1, total_itrs + 1)):
        # Request models from the optimizer
        solutions = optimizer.ask()
        # Evaluate the models and record the objectives and states
        objectives, states = [], []
        for solution in solutions:
            obj, sts = simulate(env, solution, seed=seed)
            objectives.append(obj)
            states.append(sts)
        if itr in [1]:
            train_ae(model, sts_scaler, states, num_epochs, batch_size, learning_rate)
        descriptors_raw = compute_behavior_descriptors(model, sts_scaler, states, batch_size)
        if itr in [1]:
            bcs_scaler.fit(descriptors_raw)
        descriptors = bcs_scaler.transform(descriptors_raw)
        # Send the results back to the optimizer
        optimizer.tell(objectives, descriptors, metadata=states)
        # Logging
        if itr % log_freq == 0:
            elapsed_time = time.time() - start_time
            print(f"> {itr} itrs completed after {elapsed_time:.2f} s")
            print(f"  - Archive Size: {len(archive)}")
            print(f"  - Max Score: {archive.stats.obj_max}")
            print(f"  - Mean Score: {archive.stats.obj_mean}")
            df = archive.as_pandas(include_metadata=True)
            df.to_pickle(os.path.join(output_dir, exp_name, f"archive_{itr:04}.pkl"))
        if train_mode == "INC" and itr in [10, 30, 70, 150, 310, 630]:
            df = archive.as_pandas(include_metadata=True)
            # Retrieve solutions objectives and states
            solutions = [sol for sol in df.loc[:, [f'solution_{idx}' for idx in range(num_params)]].to_numpy(copy=True)]
            objectives = [obj for obj in df.loc[:, 'objective'].to_numpy(copy=True)]
            states = [meta for meta in df.loc[:, 'metadata'].to_numpy(copy=True)]
            # Retrain model and states scaler using states of solutions in the archive
            if encode_mode == "AE":
                model = VecAE(input_space, 2).to(DEVICE)
            elif encode_mode == "VAE":
                model = VecVAE(input_space, 2).to(DEVICE)
            elif encode_mode == "RNN-AE":
                model = VecRnnAE(input_space, input_space, 2, 1).to(DEVICE)
            # elif encode_mode == "RNN-VAE":
            #     model = VecRnnVAE(input_space, input_space, 2, 1).to(DEVICE)
            else:
                raise ValueError("Unknown `encode_mode` selected.")
            train_ae(model, sts_scaler, states, num_epochs, batch_size, learning_rate)
            # Recompute behavior descriptors
            descriptors_raw = compute_behavior_descriptors(model, sts_scaler, states, batch_size)
            # Retrain behavior descriptors scaler
            descriptors = bcs_scaler.fit_transform(descriptors_raw)
            # Clear archive and reinsert solutions with the new descriptors
            archive.clear()
            for sol, obj, beh, meta in zip(solutions, objectives, descriptors, states):
                archive.add(sol, obj, beh, meta)


def parse_args():
    parser = argparse.ArgumentParser("Trainer of an agent to play OpenAy Gym Lunar Lander using MAP-Elites")
    parser.add_argument("--output_dir", help="Directory to store generated archives and trained models", required=True)
    parser.add_argument("--train_mode", help="Aurora training mode", choices=["pre", "inc"], required=True)
    parser.add_argument("--time_mode", help="Time aggregation mode", choices=["avg", "stack", "pad"], required=True)
    parser.add_argument("--encode_mode", help="Encoding mode", choices=["ae", "rnn-ae", "vae", "rnn-vae"], required=True)
    parser.add_argument("--num_generations", help="Maximum number of generations", type=int, default=1000)
    parser.add_argument("--sigma", help="Mutation rate", type=float, default=0.5)
    parser.add_argument('--num_epochs', type=int, help='Number of epochs for auto encoder training', default=10)
    parser.add_argument('--batch_size', help="Batch size for auto encoder training", default=32)
    parser.add_argument('--learning_rate', type=float, help='Learning rate for auto encoder training', default=0.001)
    parser.add_argument('--log_freq', help='Frequency for logging and saving training results', type=int, default=25)
    parser.add_argument('--seed', help='Master seed', type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    PARAMS = parse_args()
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main()
