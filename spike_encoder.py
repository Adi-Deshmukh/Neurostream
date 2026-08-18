import torch
import torch.nn as nn
from lif_neuron import LIFNeuron

class SpikeEncoder(nn.Module):
    def __init__(self, n_channels, timesteps, threshold=1.0, tau=2.0):
        super().__init__()
        self.n_channels = n_channels  # EEG channels (22 in Dataset 2a)
        self.timesteps = timesteps    # how many timesteps per trial
        self.neurons = nn.ModuleList([
            LIFNeuron(threshold=threshold, tau=tau)
            for _ in range(n_channels)
        ])

    def forward(self, eeg_signal):
        # eeg_signal shape: (n_channels, timesteps)
        spike_train = []

        for t in range(self.timesteps):
            timestep_spikes = []
            for ch, neuron in enumerate(self.neurons):
                current = eeg_signal[ch, t].unsqueeze(0)
                spike = neuron(current)
                timestep_spikes.append(spike)
            spike_train.append(torch.stack(timestep_spikes))

        # reset all neurons after each trial
        for neuron in self.neurons:
            neuron.reset()

        return torch.stack(spike_train)  # shape: (timesteps, n_channels, 1)
    
encoder = SpikeEncoder(n_channels=3, timesteps=5)
fake_eeg = torch.rand(3, 5)  # 3 channels, 5 timesteps
print("Input EEG:\n", fake_eeg)
output = encoder(fake_eeg)
print("Spike train shape:", output.shape)
print("Spikes:\n", output.squeeze())