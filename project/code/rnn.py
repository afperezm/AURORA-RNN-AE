import os
import argparse
import torch

import numpy as np

from torch.utils.data import DataLoader

from torch import nn
from torch.nn import functional as F

from project.code.data import generate_test_data, VecSeqDataset, MinMaxPaddedScaler

PARAMS = None


class VecRnnAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, encoder_dim, n_layers, p=0.5):
        super(VecRnnAE, self).__init__()
        self.dropout = nn.Dropout(p)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.lstm_encoder = nn.LSTM(input_dim, hidden_dim, n_layers, dropout=p, batch_first=True)
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, encoder_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(encoder_dim * 2, encoder_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(encoder_dim * 2, hidden_dim),
        )
        self.lstm_decoder = nn.LSTM(hidden_dim, input_dim, n_layers, dropout=p, batch_first=True)

    def encode(self, x, x_lengths, hidden1):

        x = torch.nn.utils.rnn.pack_padded_sequence(x, x_lengths, batch_first=True, enforce_sorted=False)
        lstm_enc, hidden1 = self.lstm_encoder(x, hidden1)
        lstm_enc, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_enc, batch_first=True)
        lstm_enc = lstm_enc.contiguous().view(-1, self.hidden_dim)
        enc = self.dropout(lstm_enc)
        enc = F.relu(enc)
        enc = self.encoder(enc)

        return enc

    def forward(self, x, x_lengths, hidden1, hidden2):

        batch_size = x.size(0)

        enc = self.encode(x, x_lengths, hidden1)

        enc = F.relu(enc)

        dec = self.decoder(enc)
        dec = F.relu(dec)
        dec = self.dropout(dec)
        lstm_dec = dec.view(batch_size, -1, self.hidden_dim)
        lstm_dec = torch.nn.utils.rnn.pack_padded_sequence(lstm_dec, x_lengths, batch_first=True, enforce_sorted=False)
        lstm_dec, hidden2 = self.lstm_decoder(lstm_dec, hidden2)
        lstm_dec, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_dec, batch_first=True)

        return lstm_dec, hidden1, hidden2

    def init_hidden(self, batch_size):

        device = next(self.parameters()).device

        hidden1 = (torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device),
                   torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device))
        hidden2 = (torch.zeros(self.n_layers, batch_size, self.input_dim).to(device),
                   torch.zeros(self.n_layers, batch_size, self.input_dim).to(device))

        return hidden1, hidden2

    def fit(self, data_loader, num_epochs, learning_rate, clip=5, checkpoint_dir=None):

        device = next(self.parameters()).device

        self.train()

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(1, num_epochs + 1):

            epoch_loss = 0.0

            for data, data_lengths in data_loader:
                data = data.to(device)
                # compute reconstructions
                hidden1, hidden2 = self.init_hidden(len(data))
                outputs, hidden1, hidden2 = self.forward(data, data_lengths, hidden1, hidden2)
                # undo data padding
                data_packed = torch.nn.utils.rnn.pack_padded_sequence(data, data_lengths, batch_first=True,
                                                                      enforce_sorted=False)
                data_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(data_packed, batch_first=True)
                # compute reconstruction loss
                batch_loss = criterion(outputs, data_unpacked)
                # reset the gradients back to zero
                optimizer.zero_grad()
                # compute accumulated gradients
                batch_loss.backward()
                # clip gradients
                nn.utils.clip_grad_norm_(self.parameters(), clip)
                # perform parameter update based on current gradients
                optimizer.step()
                # add the mini-batch training loss to epoch loss
                epoch_loss += batch_loss.item() * data.size(0)

            epoch_loss = epoch_loss / len(data_loader.dataset)

            print(f'Epoch: {epoch} \tTraining Loss: {epoch_loss:.6f}')

            if checkpoint_dir:
                torch.save(self.state_dict(), os.path.join(checkpoint_dir, f'rnn_ae-{epoch:04}.pth'))


def main():
    batch_size = PARAMS.batch_size
    learning_rate = PARAMS.learning_rate
    epochs = PARAMS.epochs
    seed = PARAMS.seed
    ckpt_dir = PARAMS.checkpoint_dir
    hidden_dim = PARAMS.hidden_dim
    num_layers = PARAMS.num_layers

    os.makedirs(PARAMS.checkpoint_dir, exist_ok=True)

    states_array = generate_test_data()

    scaler = MinMaxPaddedScaler(averaging=False, stacking=False)
    states_array_scaled = scaler.fit_transform(states_array)

    states_lengths = np.array([states.shape[0] for states in states_array])

    torch.manual_seed(seed)

    dataset = VecSeqDataset(states_array_scaled, states_lengths)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    n_dim = PARAMS.input_dim
    n_lat = PARAMS.encoder_dim

    model = VecRnnAE(n_dim, hidden_dim, n_lat, num_layers).to(DEVICE)

    # ----------- Train RNN AE -------------- #

    with torch.enable_grad():
        model.fit(data_loader=data_loader, num_epochs=epochs, learning_rate=learning_rate, checkpoint_dir=ckpt_dir)

    # ----------- Test RNN AE --------------- #

    # model.load_state_dict(torch.load(os.path.join(PARAMS.checkpoint_dir, 'rnn_ae-0010.pth')))

    model.eval()

    bcs_list = []

    for data, data_lengths in data_loader:
        data = data.to(DEVICE)
        hidden1, hidden2 = model.init_hidden(len(data))
        z = model.encode(data, data_lengths, hidden1)
        print(z)
        bcs_list.extend(z.tolist())

    np.save('bcs_rnn-ae.npy', np.array(bcs_list))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--input_dim', type=int, required=True)
    parser.add_argument('--hidden_dim', type=int, required=True)
    parser.add_argument('--encoder_dim', type=int, required=True)
    parser.add_argument('--num_layers', type=int, required=True)
    parser.add_argument('--clip', help='gradient clipping', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--anomaly_threshold', type=float, default=5.0)
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--checkpoint_dir', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    PARAMS = parse_args()
    main()
