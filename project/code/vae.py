import argparse
import os

import numpy as np
import torch
import torch.utils.data

from project.code.ae import VecAE
from project.code.data import generate_test_data, VecDataset
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader


DEVICE = None
PARAMS = None


class VecVAE(nn.Module):
    def __init__(self, input_space, latent_space):
        super(VecVAE, self).__init__()

        self.input_space = input_space
        self.latent_space = latent_space
        self.hidden_size = latent_space * 2

        self.fc1 = nn.Linear(self.input_space, self.hidden_size)
        self.fc21 = nn.Linear(self.hidden_size, latent_space)
        self.fc22 = nn.Linear(self.hidden_size, latent_space)
        self.fc3 = nn.Linear(latent_space, self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size, self.input_space)

    def encode_params(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def encode(self, x):
        # Encode
        mu, log_sigma = self.encode_params(x)
        # Re-parametrize
        z = self.reparametrize(mu, log_sigma)
        return z

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.tanh(self.fc4(h3))

    def reparametrize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode
        mu, log_sigma = self.encode_params(x)
        # Re-parametrize
        z = self.reparametrize(mu, log_sigma)
        # Decode
        x = self.decode(z)
        return x, mu, log_sigma

    # - LOSS FUNCTIONS ------------- - #

    # | Normalize by dimensionality: NO;  KL-Annealing: YES
    def loss_function_a(self, recon_x, x, mu, log_var, input_space, fit, kl_weight=1.0):
        bce = F.mse_loss(recon_x, x, reduction='sum')
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / mu.shape[0]
        return bce + (kl_weight * kld)

    # | Normalize by dimensionality: YES;  Annealing: NO
    def loss_function_b(self, recon_x, x, mu, log_var, input_space, fit, kl_weight=1.0):
        bce = F.mse_loss(recon_x, x, reduction='sum') / (input_space)
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / mu.shape[0]
        return bce + (0.1 * kld)

    # | Normalize by dimensionality: NO;  Annealing: NO
    def loss_function_c(self, recon_x, x, mu, log_var, input_space, fit, kl_weight=1.0):
        bce = F.mse_loss(recon_x, x, reduction='sum')
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / mu.shape[0]
        return bce + (0.1 * kld)

    # - ------------- ------------- - #

    def fit(self, data_loader, num_epochs, learning_rate, checkpoint_dir=None):

        device = next(self.parameters()).device

        self.train()

        loss_function = self.loss_function_c
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(1, num_epochs + 1):

            train_loss = []

            perc_done = epoch / (num_epochs + 1)
            kl_weight = np.min([perc_done * 4.0, 1.0])

            for data in data_loader:
                data = data.to(device)
                # compute reconstructions
                recon_batch, mu, log_sigma = self.forward(data)
                # compute reconstruction loss
                batch_loss = loss_function(recon_batch, data, mu, log_sigma, self.input_space, 0, kl_weight)
                # reset the gradients back to zero
                optimizer.zero_grad()
                # compute accumulated gradients
                batch_loss.backward()
                # perform parameter update based on current gradients
                optimizer.step()
                r = np.linalg.norm(recon_batch.detach().cpu().numpy() - data.detach().cpu().numpy(), axis=0)
                train_loss += [r]

            # per_input_loss = np.vstack(train_loss)
            # mean = np.mean(per_input_loss)
            # std = np.std(per_input_loss)

            epoch_loss = np.mean(train_loss)

            print(f'Epoch: {epoch} \tTraining Loss: {epoch_loss:.6f}')

            if checkpoint_dir:
                torch.save(self.state_dict(), os.path.join(checkpoint_dir, f'vae-{epoch:04}.pth'))

    def save(self, path):
        torch.save(self.state_dict, path)


def main():
    batch_size = PARAMS.batch_size
    learning_rate = PARAMS.learning_rate
    epochs = PARAMS.epochs
    seed = PARAMS.seed
    ckpt_dir = PARAMS.checkpoint_dir

    states_array_scaled = generate_test_data(aggregate=False, flatten=True)

    torch.manual_seed(seed)

    # Load data set
    dataset = VecDataset(states_array_scaled)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    n_dim = states_array_scaled.shape[1]
    n_lat = 2

    model = VecVAE(n_dim, n_lat).to(DEVICE)

    # ----------- Train VAE ----------------- #

    with torch.enable_grad():
        model.fit(data_loader=data_loader, num_epochs=epochs, learning_rate=learning_rate, checkpoint_dir=ckpt_dir)

    # ------------ Test VAE ----------------- #

    # model.load_state_dict(torch.load(os.path.join(ckpt_dir, 'vae-0010.pth')))
    #
    # model.eval()
    #
    # bcs_list = []
    #
    # for batch_idx, data in enumerate(data_loader):
    #     data = data.to(DEVICE)
    #     z = model.encode(data)
    #     print(z)
    #     bcs_list.extend(z.tolist())
    #
    # np.save('bcs_vae.npy', np.array(bcs_list))


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
