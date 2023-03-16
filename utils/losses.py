"""
Definition of loss functions.

This code originates from the following projects:
- https://github.com/brendel-group/cl-ica
- https://github.com/ysharma1126/ssl_identifiability
"""


from abc import ABC, abstractmethod
import torch


class CLLoss(ABC):
    """Abstract class to define losses in the CL framework that use one
    positive pair and one negative pair"""

    @abstractmethod
    def loss(self, z1, z2_con_z1, z3, z1_rec, z2_con_z1_rec, z3_rec):
        """
        z1_t = h(z1)
        z2_t = h(z2)
        z3_t = h(z3)
        and z1 ~ p(z1), z3 ~ p(z3)
        and z2 ~ p(z2 | z1)

        returns the total loss and componentwise contributions
        """
        pass

    def __call__(self, z1, z2_con_z1, z3, z1_rec, z2_con_z1_rec, z3_rec):
        return self.loss(z1, z2_con_z1, z3, z1_rec, z2_con_z1_rec, z3_rec)


class LpSimCLRLoss(CLLoss):
    """Extended InfoNCE objective for non-normalized representations based on an Lp norm.

    Args:
        p: Exponent of the norm to use.
        tau: Rescaling parameter of exponent.
        alpha: Weighting factor between the two summands.
        simclr_compatibility_mode: Use logsumexp (as used in SimCLR loss) instead of logmeanexp
        pow: Use p-th power of Lp norm instead of Lp norm.
    """

    def __init__(
        self,
        p: int = 2,
        tau: float = 1.0,
        alpha: float = 0.5,
        simclr_compatibility_mode: bool = False,
        simclr_denominator: bool = True,
        pow: bool = True,
    ):
        self.p = p
        self.tau = tau
        self.alpha = alpha
        self.simclr_compatibility_mode = simclr_compatibility_mode
        self.simclr_denominator = simclr_denominator
        self.pow = pow

    def loss(self, z1, z2_con_z1, z3, z1_rec, z2_con_z1_rec, z3_rec):
        del z1, z2_con_z1, z3

        if self.p < 1.0:
            # add small epsilon to make calculation of norm numerically more stable
            neg = torch.norm(
                torch.abs(z1_rec.unsqueeze(0) - z3_rec.unsqueeze(1) + 1e-12),
                p=self.p,
                dim=-1,
            )
            pos = torch.norm(
                torch.abs(z1_rec - z2_con_z1_rec) + 1e-12, p=self.p, dim=-1
            )
        else:
            neg = torch.pow(z1_rec.unsqueeze(1) - z3_rec.unsqueeze(0), float(self.p)).sum(dim=-1)
            pos = torch.pow(z1_rec - z2_con_z1_rec, float(self.p)).sum(dim=-1)

        if not self.pow:
            neg = neg.pow(1.0 / self.p)
            pos = pos.pow(1.0 / self.p)

        # all = torch.cat((neg, pos.unsqueeze(1)), dim=1)

        if self.simclr_compatibility_mode:
            neg_and_pos = torch.cat((neg, pos.unsqueeze(1)), dim=1)

            loss_pos = pos / self.tau
            loss_neg = torch.logsumexp(-neg_and_pos / self.tau, dim=1)
        else:
            if self.simclr_denominator:
                neg_and_pos = torch.cat((neg, pos.unsqueeze(1)), dim=1)
            else:
                neg_and_pos = neg

            loss_pos = pos / self.tau
            loss_neg = _logmeanexp(-neg_and_pos / self.tau, dim=1)

        loss = 2 * (self.alpha * loss_pos + (1.0 - self.alpha) * loss_neg)

        loss_mean = torch.mean(loss)
        # loss_std = torch.std(loss)

        loss_pos_mean = torch.mean(loss_pos)
        loss_neg_mean = torch.mean(loss_neg)

        return loss_mean, loss, [loss_pos_mean, loss_neg_mean]


def _logmeanexp(x, dim):
    # do the -log thing to use logsumexp to calculate the mean and not the sum
    # as log sum_j exp(x_j - log N) = log sim_j exp(x_j)/N = log mean(exp(x_j)
    N = torch.tensor(x.shape[dim], dtype=x.dtype, device=x.device)
    return torch.logsumexp(x, dim=dim) - torch.log(N)


def infonce_loss(z1, z2, sim_metric, criterion, tau=1.0):
    """
    This code originates from the following project:
    - https://github.com/ysharma1126/ssl_identifiability
    """
    sim11 = sim_metric(z1.unsqueeze(-2), z1.unsqueeze(-3)) / tau
    sim22 = sim_metric(z2.unsqueeze(-2), z2.unsqueeze(-3)) / tau
    sim12 = sim_metric(z1.unsqueeze(-2), z2.unsqueeze(-3)) / tau
    d = sim12.shape[-1]
    sim11[..., range(d), range(d)] = float('-inf')
    sim22[..., range(d), range(d)] = float('-inf')
    raw_scores1 = torch.cat([sim12, sim11], dim=-1)
    raw_scores2 = torch.cat([sim22, sim12.transpose(-1, -2)], dim=-1)
    raw_scores = torch.cat([raw_scores1, raw_scores2], dim=-2)
    targets = torch.arange(2 * d, dtype=torch.long, device=raw_scores.device)
    loss_value = criterion(raw_scores, targets)
    return loss_value
