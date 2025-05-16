import numpy as np
import torch
from GNA.model import TransformerModel
from GNA.optimizer_limited_sample import Optimizer_limited_sample


class GNA:
    def __init__(self, problem, sample_budget=200, param=None):
        self.problem = problem
        self.n = problem.n
        self.sample_budget = sample_budget
        self.batch = 100
        self.iter_per_sample = 5
        self.n_model = 1
        self.param = {
            "anneal_epochs": 0.3263421254240193,
            "beta_max": 69.70367987883363,
            "beta_min": 0.05716372936912354,
            "d_model": 20,
            "lr": 0.0008200045551234297,
            "n_layers": 3,
            "n_head": 1,
            "weight_decay": 0.00014704587343194282,
            "fix_beta": False
        }
        if param is not None:
            self.param.update(param)

    def run(self):
        batch = self.batch
        init_n_sample = 20
        sample_budget = self.sample_budget

        lr = self.param['lr']
        weight_decay = self.param['weight_decay']
        beta_min = self.param['beta_min']
        beta_max = self.param['beta_max']
        anneal_epochs = int(self.param['anneal_epochs'] * sample_budget)
        d_model = int(self.param['d_model'])
        n_head = int(self.param['n_head'])
        num_layers = int(self.param['n_layers'])
        fix_beta = self.param['fix_beta']

        # model = TransformerModel(vocab_size=2, d_model=d_model, n_head=n_head,
        #                          num_layers=num_layers, max_seq_len=self.n + 1)
        models = [TransformerModel(vocab_size=2, d_model=d_model, n_head=n_head,
                                   num_layers=num_layers, max_seq_len=self.n + 1)
                  for _ in range(self.n_model)]
        optimizer = Optimizer_limited_sample(self.problem, models, batch, init_n_sample, sample_budget,
                                             self.iter_per_sample, lr, weight_decay, beta_min, beta_max, anneal_epochs,
                                             fix_beta)

        loss_curve, val_loss_curve, E_min_curve = optimizer.train()
        self.loss_curve = loss_curve
        self.val_loss_curve = val_loss_curve
        s_best, E_min = optimizer.dataset.s_best()
        return s_best, E_min.item(), E_min_curve
