"""Temporal Attention Mechanism for Spiking Neural Networks.

Instead of uniform mean-rate decoding (which discards temporal ordering and treats all
timesteps equally), TemporalSpikeAttention dynamically weights timesteps based on
event-related activity (e.g. motor preparation vs imagination peak):

    score_t = v^T tanh(W_k s_t + b_k)
    alpha_t = softmax(score_t)
    feature = sum_t (alpha_t * s_t)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalSpikeAttention(nn.Module):
    """Learns temporal importance weights over spike sequences.

    Parameters
    ----------
    feature_dim : int
        Dimensionality of the feature vector at each timestep (e.g. 512).
    hidden_dim : int
        Internal projection dimension for attention scoring (default 128).
    """

    def __init__(self, feature_dim: int = 512, hidden_dim: int = 128) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        self.key_proj = nn.Linear(feature_dim, hidden_dim)
        self.score_proj = nn.Linear(hidden_dim, 1, bias=False)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.zeros_(self.key_proj.bias)
        nn.init.xavier_uniform_(self.score_proj.weight)

    def forward(
        self, sequence: torch.Tensor, return_weights: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Compute temporally attended feature vector.

        Parameters
        ----------
        sequence : Tensor, shape ``(batch_size, n_timesteps, feature_dim)``
            Spike counts or continuous neuron states across time.
        return_weights : bool
            If True, returns ``(attended_features, attention_weights)``.

        Returns
        -------
        attended : Tensor, shape ``(batch_size, feature_dim)``
        weights : Tensor (optional), shape ``(batch_size, n_timesteps)``
        """
        # sequence: (N, T, D)
        keys = torch.tanh(self.key_proj(sequence))  # (N, T, hidden_dim)
        scores = self.score_proj(keys).squeeze(-1)   # (N, T)
        weights = F.softmax(scores, dim=-1)         # (N, T)

        # Weighted sum across timesteps: (N, 1, T) @ (N, T, D) -> (N, 1, D) -> (N, D)
        attended = torch.bmm(weights.unsqueeze(1), sequence).squeeze(1)

        if return_weights:
            return attended, weights
        return attended
