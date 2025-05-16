import numpy as np
import torch
from problems.SAT import SAT_dataset
from problems.XORSAT import generate_3regular_3xorsat
from problems.subset_sum import generate_subset_sum_instance
from problems.contamination_control import contamination


def dec2bin(x, bits):
    # credit to Tiana
    # https://stackoverflow.com/questions/55918468/convert-integer-to-pytorch-tensor-of-binary-bits
    # mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).to(torch.get_default_dtype())


class SAT:
    def __init__(self, n, r=4.3, p0=0.08):
        batch = 1
        self.n = n
        self.dataset = SAT_dataset(batch, n, r, p0)
        clause_idx, clause_sign = self.dataset.generate_instances()
        self.clause_idx = clause_idx.squeeze(0)  # (n_clause, 3)
        self.clause_sign = (clause_sign < 0).squeeze(0)
        self.n_clause = self.clause_idx.shape[0]

    def E(self, s):
        """
        Compute the energy of a given spin configuration.

        Parameters:
        - s: Tensor of shape (batch, n) representing the spin configuration.

        Returns:
        - energy: Tensor of shape (batch,) representing the energy for each configuration.
        """

        # Compute the energy
        batch = s.shape[0]
        s_clause = s[torch.arange(batch).reshape(batch, 1, 1), self.clause_idx]  # (batch, n_clause, 3)
        energy = self.n_clause - (s_clause ^ self.clause_sign).any(dim=2).sum(dim=1)  # (batch, )
        return energy


class XORSAT:
    def __init__(self, n):
        self.n = n
        self.clause_idx, self.rhs, self.sol = generate_3regular_3xorsat(n, return_solution=True)
        self.clause_idx = torch.tensor(self.clause_idx, dtype=torch.int64)
        self.rhs = torch.tensor(self.rhs, dtype=torch.bool)

    def E(self, s):
        """
        Compute the energy of a given spin configuration.

        Parameters:
        - s: Tensor of shape (batch, n) representing the spin configuration.

        Returns:
        - energy: Tensor of shape (batch,) representing the energy for each configuration.
        """

        # Compute the energy
        batch = s.shape[0]
        s_clause = s[torch.arange(batch).reshape(batch, 1, 1), self.clause_idx]  # (batch, n_clause, 3)
        clauses = s_clause[:, :, 0] ^ s_clause[:, :, 1] ^ s_clause[:, :, 2]  # (batch, n_clause)
        energy = (self.rhs ^ clauses).sum(dim=1)
        return energy


class SubsetSum:
    def __init__(self, n):
        self.n = n
        self.a, self.t, self.sol = generate_subset_sum_instance(n, return_solution=True)
        self.a = torch.tensor(self.a, dtype=torch.int64)

    def E(self, s):
        """
        Compute the energy of a given spin configuration.

        Parameters:
        - s: Tensor of shape (batch, n) representing the spin configuration.

        Returns:
        - energy: Tensor of shape (batch,) representing the energy for each configuration.
        """

        # Compute the energy
        # diff = s.long() @ self.a - self.t
        diff = (s.long() * self.a).sum(dim=1) - self.t
        energy = (diff.abs() + 1).log()
        return energy


class IsingSparsification:
    def __init__(self, n, lambd=1e-2):
        self.n = 24
        self.lambd = lambd
        self.s = 2 * dec2bin(torch.arange(2 ** 16), 16) - 1
        self.J = (torch.rand(24) * (5 - 0.05) + 0.05) * (2 * torch.randint(0, 2, (24, )) - 1)
        s_idx = torch.arange(16, dtype=torch.int64).reshape(4, 4)
        edges_0 = torch.stack([s_idx[:, :-1].reshape(-1), s_idx[:, 1:].reshape(-1)], dim=0)
        edges_1 = torch.stack([s_idx[:-1, :].reshape(-1), s_idx[1:, :].reshape(-1)], dim=0)
        self.edges = torch.cat([edges_0, edges_1], dim=1)  # (2, 24)
        E = (self.J * (self.s[:, self.edges[0]] * self.s[:, self.edges[1]])).sum(dim=1)  # (2**16, )
        self.p = torch.exp(E)
        self.p = self.p / self.p.sum()  # (2**16, )
        self.log_p = torch.log(self.p)
        self.nonzero_mask = self.p > 0

    def E(self, x_all):
        loss = []
        for x in x_all:
            E = (self.J * x * (self.s[:, self.edges[0]] * self.s[:, self.edges[1]])).sum(dim=1)  # (2**16, )
            q = torch.exp(E)
            q = q / q.sum()
            nonzero_mask = self.nonzero_mask & (q > 0)
            KL = (self.p[nonzero_mask] * (self.log_p[nonzero_mask] - torch.log(q[nonzero_mask]))).sum()
            loss.append(KL + self.lambd * x.sum())
        loss = torch.stack(loss)
        return loss


class ContaminationControl:
    def __init__(self, n):
        self.n = n

    def E(self, s):
        batch = s.shape[0]
        loss = []
        for i in range(batch):
            s_i = s[i].cpu().numpy()
            cost_i = contamination(s_i, 100)
            loss.append(cost_i)
        loss = torch.tensor(loss, device=s.device)
        return loss


if __name__ == '__main__':
    problem = XORSAT(10)
    s = torch.tensor(problem.sol, dtype=bool).unsqueeze(0)
    print(problem.E(s))
