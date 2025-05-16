import os
import numpy as np
import torch
from tqdm import trange
from GNA.model import TransformerModel
from GNA.model_utils import dec2bin, bin2dec, compute_loss, sample, compute_grad_SR


class Optimizer_unlimited_sample:
    def __init__(self, problem, model, batch, lr, beta_min=0.1, beta_max_factor=10, dbeta_factor=1000, beta_max=10,
                 exact=False, use_SR=False, disable_compile=True):
        self.problem = problem
        self.model = model
        self.lr = lr
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        self.n = problem.n
        self.batch = batch
        self.beta_min = beta_min
        self.beta_max_init = beta_max_factor * beta_min
        self.d_beta_max = np.log(10) / dbeta_factor
        self.beta_max = beta_max
        self.exact = exact
        self.use_SR = use_SR
        if exact:
            self.basis = dec2bin(torch.arange(2 ** self.n), self.n).bool()
            self.E_basis = problem.E(self.basis).float()
        self.train_step_func = torch.compile(self.train_step, disable=disable_compile)

    def train_step(self, beta):
        samples, sample_weight = sample(self.model, self.n, int(1e6), self.batch, beta)

        E = self.problem.E(samples.transpose(0, 1)).float()
        E_min = E.min()
        if (E < 1e-3).any():
            sol_flag = True
            sol = samples.T[E < 1e-3]
        else:
            sol_flag = False
            sol = None

        F = ((E + sample_weight.log() / beta) * sample_weight).sum()
        F_var = ((E + sample_weight.log() / beta) ** 2 * sample_weight).sum() - F ** 2

        if self.exact:
            p_basis = (-beta * self.E_basis).exp()
            p_basis /= p_basis.sum()
            p_ground_truth = p_basis[bin2dec(samples.T, self.n).long()]
            true_KL = (sample_weight * (sample_weight.log() - p_ground_truth.log())).detach().sum().item()
        else:
            true_KL = None

        self.optimizer.zero_grad()
        if self.use_SR:
            compute_grad_SR(self.model, self.problem, samples, sample_weight, beta, lambd=1e-2)
        else:
            loss = compute_loss(self.model, self.problem, samples, sample_weight, beta)
            loss.backward()
        self.optimizer.step()

        return sol_flag, sol, E_min.item(), F.item(), F_var.item(), true_KL

    def train(self, n_iter, beta=None):
        F_std_curve = []
        E_min_curve = []
        KL_curve = []
        log_beta_max = np.log(self.beta_max_init)
        log_beta_min = np.log(self.beta_min)
        log_beta_max_upper_bound = np.log(self.beta_max)

        for i in trange(n_iter):
            if log_beta_max < log_beta_max_upper_bound:
                log_beta_max += self.d_beta_max
            beta = np.exp(np.random.rand(1) * (log_beta_max - log_beta_min) + log_beta_min).item()
            sol_flag, sol, E_min, F, F_var, true_KL = self.train_step_func(beta)
            F_std = np.sqrt(F_var)
            E_min_curve.append(E_min)
            F_std_curve.append(F_std)
            KL_curve.append(true_KL)
            print(f'n = {self.problem.n}, Iteration {i}, E_min: {E_min:.2f}  F: {F:.4f}  F_std: {F_std:.4f}  '
                  f'beta: {beta:.4f}')
            # if sol_flag:
            #     print(f'Solution found at iteration {i}: {sol}')
            #     break
        return sol_flag, sol, i, np.array(E_min_curve), np.array(F_std_curve), np.array(KL_curve)