import os
import json
from tqdm import trange
import numpy as np
import torch
from problems.problem import SAT, XORSAT, SubsetSum

# Define and import your custom problem here
from problems.custom_problem import CustomProblem

from GNA.model import TransformerModel
from GNA.optimizer_unlimited_sample import Optimizer_unlimited_sample

torch.set_default_tensor_type(torch.cuda.FloatTensor
                                 if torch.cuda.is_available()
                                 else torch.FloatTensor)


if __name__ == '__main__':
    import argparse
    parser_ = argparse.ArgumentParser()
    parser_.add_argument('--problem', dest='problem', type=str, default='SAT')
    parser_.add_argument('--n', dest='n', type=int, default=25)
    args = parser_.parse_args()

    result_dir = f'results'
    os.makedirs(result_dir, exist_ok=True)

    supported_problems = ['SAT', 'XORSAT', 'SubsetSum']

    try:
        problem_instance = eval(args.problem)
    except NameError:
        raise ValueError(f'Problem {args.problem} not found. Supported problems are: {supported_problems}\n'
                         f'If you have a custom problem, please define and import it.')

    n = args.n

    problem_name = args.problem
    algorithm = 'GNA'

    problem = problem_instance(n)
    model = TransformerModel(vocab_size=2, d_model=32, n_head=1, num_layers=4, max_seq_len=n + 1)
    batch = 1000
    lr = 5e-4
    beta_min = 0.1
    beta_max_factor = 10
    dbeta_factor = 10000
    beta_max = 100
    n_iter = 10000
    optimizer = Optimizer_unlimited_sample(problem, model, batch, lr, beta_min, beta_max_factor, dbeta_factor, beta_max)
    sol_flag, sol, sol_step, E_min_curve, F_std_curve, KL_curve = optimizer.train(n_iter)
    if sol_flag:
        print(f'Solution found at step {sol_step}: {sol}\n')
    else:
        print(f'Solution not found within {n_iter} iterations.\n')

    with open(f'{result_dir}/{problem_name}_{algorithm}_unlimited.txt', 'a') as f:
        f.write(f'{n}\t{sol_step}\n')
