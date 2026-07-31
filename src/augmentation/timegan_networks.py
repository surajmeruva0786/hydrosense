"""TimeGAN network components (Yoon, Jarrett & van der Schaar, 2019, README §9).

Operates on framed sequences of shape (batch, seq_len, feature_dim): a
raw waveform segment is chopped into fixed-size non-overlapping frames
(`src.augmentation.train_timegan.frame_waveform`) so a GRU sees a
tractable sequence length instead of hundreds of thousands of raw samples.

Five networks, all GRU-based, matching the paper's roles:

- **Embedder**   e: raw feature space -> latent space H
- **Recovery**   r: latent space H -> raw feature space (inverse of e)
- **Generator**  g: random noise Z -> latent space H
- **Supervisor** s: H_t -> H_{t+1} (next-step latent dynamics; also
  post-processes the generator's output so it obeys the same stepwise
  dynamics learned from real data)
- **Discriminator** d: H -> real/fake logit per timestep
"""

from __future__ import annotations

import torch
from torch import nn


class _GRUEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int, output_activation: str | None = "sigmoid"):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, output_dim)
        self.output_activation = output_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        out = self.head(out)
        if self.output_activation == "sigmoid":
            out = torch.sigmoid(out)
        return out


class Embedder(_GRUEncoder):
    """Raw framed waveform -> latent embedding sequence (bounded to [0, 1])."""

    def __init__(self, feature_dim: int, hidden_dim: int, num_layers: int = 3):
        super().__init__(feature_dim, hidden_dim, num_layers, hidden_dim, output_activation="sigmoid")


class Recovery(_GRUEncoder):
    """Latent embedding sequence -> reconstructed raw framed waveform."""

    def __init__(self, hidden_dim: int, feature_dim: int, num_layers: int = 3):
        super().__init__(hidden_dim, hidden_dim, num_layers, feature_dim, output_activation=None)


class Generator(_GRUEncoder):
    """Random noise sequence -> synthetic latent embedding sequence."""

    def __init__(self, noise_dim: int, hidden_dim: int, num_layers: int = 3):
        super().__init__(noise_dim, hidden_dim, num_layers, hidden_dim, output_activation="sigmoid")


class Supervisor(_GRUEncoder):
    """Latent embedding sequence H_t -> next-step prediction H_{t+1}.

    Trained on real embeddings (teacher forcing) and reused to refine the
    generator's raw output so synthetic sequences follow learned stepwise
    dynamics, not just the right marginal distribution.
    """

    def __init__(self, hidden_dim: int, num_layers: int = 2):
        super().__init__(hidden_dim, hidden_dim, num_layers, hidden_dim, output_activation="sigmoid")


class Discriminator(_GRUEncoder):
    """Latent embedding sequence -> per-timestep real/fake logit."""

    def __init__(self, hidden_dim: int, num_layers: int = 3):
        super().__init__(hidden_dim, hidden_dim, num_layers, 1, output_activation=None)


class TimeGAN(nn.Module):
    """Bundles the five TimeGAN sub-networks for a single (class-specific) generator."""

    def __init__(self, feature_dim: int, hidden_dim: int = 24, noise_dim: int | None = None, num_layers: int = 3):
        super().__init__()
        noise_dim = noise_dim or feature_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.noise_dim = noise_dim

        self.embedder = Embedder(feature_dim, hidden_dim, num_layers)
        self.recovery = Recovery(hidden_dim, feature_dim, num_layers)
        self.generator = Generator(noise_dim, hidden_dim, num_layers)
        self.supervisor = Supervisor(hidden_dim, max(1, num_layers - 1))
        self.discriminator = Discriminator(hidden_dim, num_layers)

    def sample_noise(self, batch_size: int, seq_len: int, device: str) -> torch.Tensor:
        return torch.rand(batch_size, seq_len, self.noise_dim, device=device)

    @torch.no_grad()
    def generate(self, batch_size: int, seq_len: int, device: str = "cpu") -> torch.Tensor:
        """Sample `batch_size` synthetic framed sequences of shape (batch, seq_len, feature_dim)."""
        self.eval()
        z = self.sample_noise(batch_size, seq_len, device)
        e_hat = self.generator(z)
        h_hat = self.supervisor(e_hat)
        return self.recovery(h_hat)
