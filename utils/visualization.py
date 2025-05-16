import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import plt_config
matplotlib.use('Agg')  # Use a non-interactive backend for matplotlib
color = plt.rcParams['axes.prop_cycle'].by_key()['color']

problems = ['IsingSparsification', 'ContaminationControl', 'SAT', 'XORSAT', 'SubsetSum']
ns = [24, 25, 25, 25, 25]
algorithms = ['SA', 'TPE', 'BOCS-SA', 'BOCS-SDP', 'COMBO', 'GNA-SA', 'GNA-PT']
labels = ['SA', 'TPE', 'BOCS-SA', 'BOCS-SDP', 'COMBO', 'GNA-SA', 'GNA-PT']
result_dir = '../results'
for idx, problem in enumerate(problems):
    n = ns[idx]
    fig, ax = plt.subplots(figsize=(6, 5))
    fig2, ax2 = plt.subplots(figsize=(3, 1.5))
    for i, algorithm in enumerate(algorithms):
        try:
            E_min_curves = np.load(f'{result_dir}/E_min_curves_{problem}_{n}_{algorithm}.npy')
        except FileNotFoundError:
            continue
        batch, length = E_min_curves.shape
        if length == 180:
            E_min_curves = np.concatenate([
                E_min_curves[:, 0].reshape(batch, 1) * np.ones((batch, 20)),
                E_min_curves
            ], axis=1)  # init sample budget is 20
            length = E_min_curves.shape[1]
        x = np.linspace(0, 200, length+1, dtype=int)[1:]  # sample budget is 200
        E_min_mean = np.mean(E_min_curves, axis=0)
        E_min_std = np.std(E_min_curves, axis=0)
        E_min_min = np.min(E_min_curves, axis=0)
        E_min_max = np.max(E_min_curves, axis=0)

        ax.plot(x, E_min_mean, color=color[i], label=labels[i])
        ax.fill_between(x, E_min_min, E_min_max, color=color[i], alpha=0.2)

        ax2.plot(x, E_min_mean, color=color[i], label=algorithm)

    ax.set_xlabel('Number of Queries m')
    ax.set_ylabel('Objective function f(x)')
    # ax.legend()
    fig.savefig(f'{result_dir}/E_min_curves_{problem}.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{result_dir}/E_min_curves_{problem}.svg', dpi=300, bbox_inches='tight')
    fig.savefig(f'{result_dir}/E_min_curves_{problem}.pdf', dpi=300, bbox_inches='tight')

    plt.close(fig)

    if problem == 'IsingSparsification':
        ax2.set_ylim([0.15, 1])
    elif problem == 'ContaminationControl':
        ax2.set_ylim([21.3, 22])
    elif problem  == 'SAT':
        ax2.set_ylim([1, 3])
    elif problem == 'XORSAT':
        ax2.set_ylim([3, 6])
    elif problem == 'SubsetSum':
        ax2.set_ylim([11.3, 13])

    ax2.set_xlabel('m')
    ax2.set_ylabel('f(x)')

    fig2.savefig(f'{result_dir}/E_min_curves_zoomed_{problem}.png', dpi=300, bbox_inches='tight')
    fig2.savefig(f'{result_dir}/E_min_curves_zoomed_{problem}.svg', dpi=300, bbox_inches='tight')
    fig2.savefig(f'{result_dir}/E_min_curves_zoomed_{problem}.pdf', dpi=300, bbox_inches='tight')

    plt.close(fig2)
