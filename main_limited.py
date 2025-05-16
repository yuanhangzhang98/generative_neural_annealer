import os
import json
from tqdm import trange
import numpy as np
import torch
from problems.problem import IsingSparsification, ContaminationControl, SAT, XORSAT, SubsetSum

# Define and import your custom problem here
from problems.custom_problem import CustomProblem

from algorithms.SA import SA
from algorithms.TPE import TPE
from GNA.GNA import GNA

torch.set_default_tensor_type(torch.cuda.FloatTensor
                                 if torch.cuda.is_available()
                                 else torch.FloatTensor)


if __name__ == '__main__':
    import argparse
    parser_ = argparse.ArgumentParser()
    parser_.add_argument('--problem', dest='problem', type=str, default='IsingSparsification')
    parser_.add_argument('--n', dest='n', type=int, default=25)
    parser_.add_argument('--algorithm', dest='algorithm', type=str, default='GNA')
    parser_.add_argument('--sample_budget', dest='sample_budget', type=int, default=200)
    parser_.add_argument('--n_repeat', dest='n_repeat', type=int, default=10)
    args = parser_.parse_args()

    result_dir = f'results'
    os.makedirs(result_dir, exist_ok=True)

    supported_problems = ['IsingSparsification', 'ContaminationControl', 'SAT', 'XORSAT', 'SubsetSum']
    supported_algorithms = ['GNA-SA', 'GNA-PT', 'SA', 'TPE']

    try:
        problem_instance = eval(args.problem)
    except NameError:
        raise ValueError(f'Problem {args.problem} not found. Supported problems are: {supported_problems}\n'
                         f'If you have a custom problem, please define and import it.')

    param = {}
    if args.algorithm == 'GNA-SA':
        solver_instance = GNA
        param['fix_beta'] = True
    elif args.algorithm == 'GNA-PT':
        solver_instance = GNA
        param['fix_beta'] = False
    else:
        try:
            solver_instance = eval(args.algorithm)
        except NameError:
            raise ValueError(f'Algorithm {args.algorithm} not found. Supported algorithms are: {supported_algorithms}')

    n_repeat = args.n_repeat
    n = args.n
    sample_budget = args.sample_budget

    problem_name = args.problem
    algorithm = args.algorithm
    with open(f'{result_dir}/{problem_name}_{n}_{algorithm}.txt', 'a') as f:
        f.write(f'# Problem: {problem_name}, n: {n}, algorithm: {algorithm}, sample_budget: {sample_budget}\n')
    metric = np.zeros(n_repeat)
    E_min_curves = []

    for i in trange(n_repeat):
        problem = problem_instance(n)
        solver = solver_instance(problem, sample_budget, param)
        s_best, E_min, E_min_curve = solver.run()
        E_min_curves.append(E_min_curve)
        metric[i] = E_min
        print(f'Trial {i}, E_min: {E_min:.4f}')
        with open(f'{result_dir}/{problem_name}_{n}_{algorithm}.txt', 'a') as f:
            f.write(f'{i}\t{E_min:.4f}\n')

    metric_mean = metric.mean()
    metric_std = metric.std()
    json.dump({'param': solver.param,
               'loss': metric.tolist(),
               'loss_mean': metric_mean,
               'loss_std': metric_std},
              open(f'{result_dir}/{problem_name}_{n}_{algorithm}.json', 'w'), indent=4)
    E_min_curves = np.array(E_min_curves)
    np.save(f'{result_dir}/E_min_curves_{problem_name}_{n}_{algorithm}.npy', E_min_curves)
