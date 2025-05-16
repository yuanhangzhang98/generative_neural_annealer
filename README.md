# A Generative Neural Annealer for Black-Box Combinatorial Optimization

This repository is the official implementation of *A Generative Neural Annealer for Black-Box Combinatorial Optimization*. 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

Note that the installation of the PyTorch library depends on your system and hardware. See [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) for a detailed instruction. 

## Usage

To optimize a black-box function with limited query budget: 

```
python main_limited.py --problem SAT --n 25 --algorithm GNA-PT --sample_budget 200 --n_repeat 1
```

Arguments: 
- problem: Name of the problem. Supported problems: IsingSparsification, ContaminationControl, SAT, XORSAT, SubsetSum. You can also define your custom problem using the template in `problems/custom_problem.py`.
- n: Number of input Boolean variables. 
- algorithm: Name of the algorithm. Supported algorithms: GNA-SA, GNA-PT, SA, TPE. You can also try [BOCS](https://github.com/baptistar/BOCS) and [COMBO](https://github.com/QUVA-Lab/COMBO), two algorithms based on Bayesian optimization that we benchmarked in our paper. These algorithms have great sample efficiency, but a higher computational complexity. 
- sample_budget: The maximum allowed function evaluations. 
- n_repeat: Repeat the algorithm this many times. Mainly for benchmarking purposes. 

To optimize a black-box function with unlimited query budget: 

```
python main_unlimited.py --problem SAT --n 25
```

Arguments: 
- problem: Name of the problem. Supported problems: SAT, XORSAT, SubsetSum. Note that this algorithm performs many function evaluations, so if you would like to define a custom problem, make sure that it is cheap to evaluate and properly vectorized. Also, by default, the algorithm terminates when finding a configuration with `f(x)<1e-3` (`f(x)=0` corresponds to the solution in the three benchmarked problems). You can adjust or disable this by editing the `train_step` function in `optimizer_unlimited_sample.py`. 
- n: Number of input Boolean variables. 

Uses GNA-PT by default. 
