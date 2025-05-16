import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import trange
from GNA.dataset import BinaryDataset
from GNA.model import TransformerModel
from GNA.model_utils import compute_prob, sample_without_weight

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Optimizer_limited_sample:
    def __init__(self, problem, models, batch, init_n_sample, sample_budget, iter_per_sample, lr, weight_decay,
                 beta_min=0.1, beta_max=10.0, anneal_epochs=50, fix_beta=False):
        self.problem = problem
        self.models = models
        self.n_model = len(models)
        self.lr = lr
        self.optimizers = [torch.optim.AdamW(model_i.parameters(), lr=lr, weight_decay=weight_decay)
                           for model_i in models]

        self.n = problem.n
        self.batch = batch
        self.init_n_sample = init_n_sample
        self.sample_budget = sample_budget
        self.iter_per_sample = iter_per_sample
        self.reset_model_every = 20

        self.beta_min = beta_min
        self.beta_max = beta_max
        self.anneal_epochs = anneal_epochs
        self.beta_schedule = np.logspace(np.log10(beta_min), np.log10(beta_max), self.anneal_epochs)
        self.fix_beta = fix_beta

        self.dataset = BinaryDataset(self.n, problem)
        init_samples = torch.randint(0, 2, (self.n, init_n_sample), dtype=torch.bool)
        self.dataset.add_samples(init_samples)
        # data size is small; use all the validation set
        self.best_val_losses = [float('inf') for _ in range(self.n_model)]
        self.best_model = [None for _ in range(self.n_model)]

    def compute_loss(self, model, data, beta):
        samples, E = data
        log_prob = compute_prob(model, samples, beta)  # (batch, )
        log_prob = log_prob.log_softmax(dim=0)  # (batch, )
        log_prob_target = (-beta * E).log_softmax(dim=0)  # (batch, )
        loss = (log_prob_target.exp() * (log_prob_target - log_prob)).sum()
        return loss

    def add_data(self, model, beta):
        duplicate_count = 0
        while True:
            if duplicate_count < 10:
                samples = sample_without_weight(model, self.n, 1, beta)
            else:
                samples = torch.randint(0, 2, (self.n, 1), dtype=torch.bool)
            dupe_flag = self.dataset.add_samples(samples)
            if dupe_flag:
                duplicate_count += 1
            else:
                break

    def train(self):
        loss_curve = np.zeros((self.n_model, self.sample_budget, self.iter_per_sample))
        val_loss_curve = np.zeros((self.n_model, self.sample_budget))
        min_E_curve = np.zeros((self.sample_budget,))

        for i in trange(self.sample_budget):
            beta = self.beta_schedule[min(i, self.anneal_epochs - 1)]
            for j in range(self.iter_per_sample):
                if self.fix_beta:
                    beta_train = beta
                else:
                    beta_train = np.random.uniform(self.beta_min, beta)

                # TODO: parallel evaluation of all models using vmap

                for k in range(self.n_model):
                    self.models[k].train()
                    data = self.dataset.get_train_samples(self.batch)
                    loss = self.compute_loss(self.models[k], data, beta_train)
                    self.optimizers[k].zero_grad()
                    loss.backward()
                    self.optimizers[k].step()
                    loss_curve[k, i, j] = loss.item()

            val_data = self.dataset.get_val_samples()
            with torch.no_grad():
                for k in range(self.n_model):
                    self.models[k].eval()
                    val_loss_k = self.compute_loss(self.models[k], val_data, beta)
                    val_loss_curve[k, i] = val_loss_k.item()
                    if val_loss_k < self.best_val_losses[k]:
                        self.best_val_losses[k] = val_loss_k.item()
                        self.best_model[k] = self.models[k].state_dict()
            current_best_model = np.argmin(val_loss_curve[:, i])
            if i > self.init_n_sample:
                self.add_data(self.models[current_best_model], beta)
            if i % self.reset_model_every == 0:
                for k in range(self.n_model):
                    self.models[k].load_state_dict(self.best_model[k])
                    self.best_val_losses[k] = float('inf')
                    self.best_model[k] = None

            min_E_curve[i] = self.dataset.E_min.item()

            # print(f'n={self.problem.n}, Iteration {i}, Loss: {loss:.4f}, beta: {beta:.4f}')
            print(f'n={self.problem.n} Iter {i} Loss: {np.min(loss_curve[:, i]):.4f} '
                  f'Validation Loss: {np.min(val_loss_curve[:, i]):.4f} '
                  f'min_E: {min_E_curve[i]:.4f}  beta: {beta:.4f}')
        return loss_curve, val_loss_curve, min_E_curve
