import argparse
import os
import numpy as np
import torch
import torch.utils.data

from project.code.data import generate_test_data, VecDataset, MinMaxPaddedScaler
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


DEVICE = None
PARAMS = None


class VecAE(nn.Module):
    def __init__(self, input_space, latent_space, num_layers=1):
        super().__init__()

        self.input_space = input_space
        self.latent_space = latent_space
        self.hidden_size = latent_space * 2
        self.num_layers = num_layers

        # self.rnn = nn.LSTM(input_size=input_space, hidden_size=input_space,
        #                    num_layers=self.num_layers, batch_first=True)

        self.encoder = nn.Sequential(
            nn.Linear(self.input_space, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.latent_space)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_space, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.input_space),
            # nn.Sigmoid()
        )

    # def encode(self, x):
    #     # Retrieve device from model parameters
    #     device = next(self.parameters()).device
    #
    #     # Bootstrap hidden and internal states
    #     h0 = torch.zeros(self.num_layers, x.size(0), self.input_space).to(device)
    #     c0 = torch.zeros(self.num_layers, x.size(0), self.input_space).to(device)
    #
    #     # Propagate input through LSTM
    #     output, (hn, cn) = self.rnn(x, (h0, c0))
    #
    #     # Propagate activated hidden state through auto encoder
    #     x = F.relu(hn, inplace=True)
    #     x = self.encoder(x)
    #
    #     return hn, x

    def encode(self, x):
        x = self.encoder(x)
        return x

    def forward(self, x):
        # hn, x = self.encode(x)
        x = self.encoder(x)
        x = F.relu(x, inplace=True)
        x = self.decoder(x)
        # return hn, x
        return x

    def fit(self, data_loader, num_epochs, learning_rate, checkpoint_dir=None):

        device = next(self.parameters()).device

        self.train()

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(1, num_epochs + 1):

            epoch_loss = 0.0

            for data in data_loader:
                data = data.to(device)
                # compute reconstructions
                # reduced_batch, decoded_batch = model.forward(data)
                decoded_batch = self.forward(data)
                # compute reconstruction loss
                # batch_loss = loss_function(decoded_batch, reduced_batch)
                batch_loss = criterion(decoded_batch, data)
                # reset the gradients back to zero
                optimizer.zero_grad()
                # compute accumulated gradients
                batch_loss.backward()
                # perform parameter update based on current gradients
                optimizer.step()
                # add the mini-batch training loss to epoch loss
                epoch_loss += batch_loss.item() * data.size(0)

            epoch_loss = epoch_loss / len(data_loader.dataset)

            print(f'Epoch: {epoch} \tTraining Loss: {epoch_loss:.6f}')

            if checkpoint_dir:
                torch.save(self.state_dict(), os.path.join(checkpoint_dir, f'ae-{epoch:04}.pth'))


def main():
    batch_size = PARAMS.batch_size
    learning_rate = PARAMS.learning_rate
    epochs = PARAMS.epochs
    seed = PARAMS.seed
    ckpt_dir = PARAMS.checkpoint_dir

    scaler = MinMaxPaddedScaler(averaging=False, stacking=True)
    states_array = generate_test_data()
    states_array_scaled = scaler.fit_transform(states_array)

    torch.manual_seed(seed)

    # Load data set
    dataset = VecDataset(states_array_scaled)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    n_dim = states_array_scaled.shape[1]
    n_lat = 2

    model = VecAE(n_dim, n_lat).to(DEVICE)

    # ----------- Train AE ------------------ #

    with torch.enable_grad():
        model.fit(data_loader=data_loader, num_epochs=epochs, learning_rate=learning_rate, checkpoint_dir=ckpt_dir)

    # ------------ Test AE ------------------ #

    # model.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'rnn_ae-0010.pth')))

    model.eval()

    bcs_list = []

    for data in data_loader:
        data = data.to(DEVICE)
        z = model.encode(data)
        print(z)
        bcs_list.extend(z.tolist())

    np.save('bcs_ae.npy', np.array(bcs_list))


def parse_args():
    parser = argparse.ArgumentParser(description='VAE Example')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--checkpoint_dir', required=True)
    return parser.parse_args()


# ------------------------------------------------------------------------------#
if __name__ == "__main__":
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    PARAMS = parse_args()
    main()
