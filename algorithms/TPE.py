import pickle
import os
import numpy as np
import torch
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK


class TPE:
    def __init__(self, problem, sample_budget=200, param=None):
        self.problem = problem
        self.n = problem.n
        self.sample_budget = sample_budget
        self.param = param
        self.space = {f's{i}': hp.choice(f's{i}', [0, 1]) for i in range(self.n)}
        self.result_dir = 'results'
        os.makedirs(self.result_dir, exist_ok=True)

    def objective(self, param):
        s = torch.tensor([param[f's{i}'] for i in range(self.n)], dtype=torch.bool).reshape(1, self.n)
        E = self.problem.E(s)
        return {'loss': E.item(), 'status': STATUS_OK}

    def run(self):
        tpe_trials = Trials()
        best = fmin(fn=self.objective, space=self.space, algo=tpe.suggest,
                    max_evals=self.sample_budget, trials=tpe_trials)
        s_best = torch.tensor([best[f's{i}'] for i in range(self.n)], dtype=torch.bool)
        data = []
        data_i = [tpe_trials.trials[i]['result'] for i in range(len(tpe_trials.trials))]
        data.extend(data_i)
        E_curve = np.array([d['loss'] for d in data])
        E_min = np.min(E_curve)
        E_min_curve = np.minimum.accumulate(E_curve)
        return s_best, E_min, E_min_curve

