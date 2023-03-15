"""
Definition of latent spaces for the multimodal setup.

Parts of this code originate from the files spaces.py and latent_spaces.py
from the following projects:
- https://github.com/brendel-group/cl-ica
- https://github.com/ysharma1126/ssl_identifiability
"""

from typing import Callable, List
from abc import ABC, abstractmethod
import numpy as np
import torch


class Space(ABC):
    @abstractmethod
    def uniform(self, size, device):
        pass

    @abstractmethod
    def normal(self, mean, std, size, device):
        pass

    @property
    @abstractmethod
    def dim(self):
        pass


class NRealSpace(Space):
    def __init__(self, n):
        self.n = n

    @property
    def dim(self):
        return self.n

    def uniform(self, size, device="cpu"):
        raise NotImplementedError("Not defined on R^n")

    def normal(self, mean, std, size, device="cpu", change_prob=1., Sigma=None):
        """Sample from a Normal distribution in R^N.
        Args:
            mean: Value(s) to sample around.
            std: Concentration parameter of the distribution (=standard deviation).
            size: Number of samples to draw.
            device: torch device identifier
        """
        if mean is None:
            mean = torch.zeros(self.n)
        if len(mean.shape) == 1 and mean.shape[0] == self.n:
            mean = mean.unsqueeze(0)
        if not torch.is_tensor(std):
            std = torch.ones(self.n) * std
        if len(std.shape) == 1 and std.shape[0] == self.n:
            std = std.unsqueeze(0)
        assert len(mean.shape) == 2
        assert len(std.shape) == 2

        if torch.is_tensor(mean):
            mean = mean.to(device)
        if torch.is_tensor(std):
            std = std.to(device)
        change_indices = torch.distributions.binomial.Binomial(probs=change_prob).sample((size, self.n)).to(device)
        if Sigma is not None:
            changes = np.random.multivariate_normal(np.zeros(self.n), Sigma, size)
            changes = torch.FloatTensor(changes).to(device)
        else:
            changes = torch.randn((size, self.n), device=device) * std
        return mean + change_indices * changes


class LatentSpace:
    """Combines a topological space with a marginal and conditional density to sample from."""

    def __init__(
        self, space: Space, sample_marginal: Callable, sample_conditional: Callable
    ):
        self.space = space
        self._sample_marginal = sample_marginal
        self._sample_conditional = sample_conditional

    @property
    def sample_conditional(self):
        if self._sample_conditional is None:
            raise RuntimeError("sample_conditional was not set")
        return lambda *args, **kwargs: self._sample_conditional(
            self.space, *args, **kwargs
        )

    @sample_conditional.setter
    def sample_conditional(self, value: Callable):
        assert callable(value)
        self._sample_conditional = value

    @property
    def sample_marginal(self):
        if self._sample_marginal is None:
            raise RuntimeError("sample_marginal was not set")
        return lambda *args, **kwargs: self._sample_marginal(
            self.space, *args, **kwargs
        )

    @sample_marginal.setter
    def sample_marginal(self, value: Callable):
        assert callable(value)
        self._sample_marginal = value

    @property
    def dim(self):
        return self.space.dim


class ProductLatentSpace(LatentSpace):
    """A latent space which is the cartesian product of other latent spaces."""

    def __init__(self, spaces: List[LatentSpace], a=None, B=None):
        """Assumes that the list of spaces is [c, s] or [c, s, m1, m2]."""
        self.spaces = spaces
        self.a = a
        self.B = B

        # determine dimensions, assuming the ordering [c, s, m1, m2]
        assert len(spaces) in (2, 4)  # either [c, s] or [c, s, m1, m2]
        self.content_n = spaces[0].dim
        self.style_n = spaces[1].dim
        self.modality_n = 0
        if len(spaces) > 2:
            assert spaces[2].dim == spaces[3].dim  # can be relaxed
            self.modality_n = spaces[2].dim

    def sample_conditional(self, z, size, **kwargs):
        z_new = []
        n = 0
        for s in self.spaces:
            if len(z.shape) == 1:
                z_s = z[n : n + s.space.n]
            else:
                z_s = z[:, n : n + s.space.n]
            n += s.space.n
            z_new.append(s.sample_conditional(z=z_s, size=size, **kwargs))

        return torch.cat(z_new, -1)

    def sample_marginal(self, size, **kwargs):
        z = [s.sample_marginal(size=size, **kwargs) for s in self.spaces]
        if self.a is not None and self.B is not None:
            content_dependent_style = torch.einsum("ij,kj -> ki", self.B, z[0]) + self.a
            z[1] += content_dependent_style  # index 1 is style
        return torch.cat(z, -1)

    def sample_z1_and_z2(self, size, device):
        z = self.sample_marginal(size=size, device=device)  # z = (c, s, m1, m2)
        z_tilde = self.sample_conditional(z, size=size, device=device)  # s -> s_tilde
        z1 = self.z_to_zi(z, modality=1)        # z1 = (c, s, m1)
        z2 = self.z_to_zi(z_tilde, modality=2)  # z2 = (c, s_tilde, m2)
        return z1, z2

    def z_to_csm(self, z):
        nc, ns, nm = self.content_n, self.style_n, self.modality_n
        ix_c = torch.tensor(range(0, nc), dtype=int)
        ix_s = torch.tensor(range(nc, nc + ns), dtype=int)
        ix_m1 = torch.tensor(range(nc + ns, nc + ns + nm), dtype=int)
        ix_m2 = torch.tensor(range(nc + ns + nm, nc + ns + nm*2), dtype=int)
        c = z[:, ix_c]
        s = z[:, ix_s]
        m1 = z[:, ix_m1]
        m2 = z[:, ix_m2]
        return c, s, m1, m2

    def zi_to_csmi(self, zi):
        nc, ns, nm = self.content_n, self.style_n, self.modality_n
        ix_c = torch.tensor(range(0, nc), dtype=int)
        ix_s = torch.tensor(range(nc, nc + ns), dtype=int)
        ix_mi = torch.tensor(range(nc + ns, nc + ns + nm), dtype=int)
        c = zi[:, ix_c]
        s = zi[:, ix_s]
        mi = zi[:, ix_mi]
        return c, s, mi

    def z_to_zi(self, z, modality):
        assert modality in [1, 2]
        c, s, m1, m2 = self.z_to_csm(z)
        if modality == 1:
            zi = torch.cat((c, s, m1), dim=-1)
        elif modality == 2:
            zi = torch.cat((c, s, m2), dim=-1)
        return zi

    @property
    def dim(self):
        return sum([s.dim for s in self.spaces])
