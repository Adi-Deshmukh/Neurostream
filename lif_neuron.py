import torch
import torch.nn as nn

class LIFNeuron(nn.Module):
    def __init__(self, threshold=1.0, tau=2.0, dt=1.0):
        super().__init__()
        self.threshold = threshold  # firing threshold
        self.tau = tau              # membrane time constant
        self.dt = dt                # timestep
        self.membrane = None        # will be initialized on first input

    def forward(self, input_current):
        # initialize membrane potential on first call
        if self.membrane is None:
            self.membrane = torch.zeros_like(input_current)

        # leak: membrane decays toward zero
        decay = torch.exp(torch.tensor(-self.dt / self.tau))
        self.membrane = self.membrane * decay + input_current

        # fire: where membrane crosses threshold
        spike = (self.membrane >= self.threshold).float()

        # reset: where spike occurred, membrane goes to zero
        self.membrane = self.membrane * (1.0 - spike)

        return spike

    def reset(self):
        self.membrane = None

neuron = LIFNeuron()
for t in range(10):
    current = torch.tensor([0.5])
    spike = neuron(current)
    print(f"t={t} membrane={neuron.membrane.item():.3f} spike={spike.item()}")