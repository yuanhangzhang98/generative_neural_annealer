import numpy as np
import torch


class SA:
    def __init__(self, problem, sample_budget=200, param=None):
        self.problem = problem
        self.n = problem.n
        self.sample_budget = sample_budget
        self.batch = 2
        self.param = {
            'beta_min': 0.1,
            'beta_max': 10.0,
            'anneal_epochs': 0.8,
        }
        if param is not None:
            self.param.update(param)
        self.beta_min = self.param['beta_min']
        self.beta_max = self.param['beta_max']
        self.anneal_epochs = int(sample_budget * self.param['anneal_epochs'] / self.batch)
        self.beta_schedule = np.logspace(np.log10(self.beta_min), np.log10(self.beta_max), self.anneal_epochs)

    def run(self):
        s = torch.randint(0, 2, (self.batch, self.n), dtype=torch.bool)
        E = self.problem.E(s)
        E_min = E.min().item()
        s_best = s[E.argmin()].clone()
        n_sample = self.batch
        E_min_curve = [E.min().item()]
        for i in range(1, self.sample_budget):
            beta = self.beta_schedule[min(i, self.anneal_epochs - 1)]
            flip_idx = torch.randint(0, self.n, (self.batch,))
            s_flip = s.clone()
            s_flip[torch.arange(self.batch), flip_idx] = ~s_flip[torch.arange(self.batch), flip_idx]
            E1 = self.problem.E(s_flip)
            accept_mask = torch.rand(self.batch) < torch.exp(-beta * (E1 - E))
            s[accept_mask] = s_flip[accept_mask]
            E[accept_mask] = E1[accept_mask]
            E_min_i = E.min().item()
            if E_min_i < E_min:
                E_min = E_min_i
                s_best = s[E.argmin()].clone()
            n_sample += self.batch
            E_min_curve.append(E.min().item())
            if n_sample >= self.sample_budget:
                break
        E_min_curve = np.array(E_min_curve)
        E_min_curve = np.minimum.accumulate(E_min_curve)
        return s_best, E_min, E_min_curve
