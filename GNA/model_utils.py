import numpy as np
import torch
import torch.nn as nn
from torch.distributions.binomial import Binomial
from torch.func import vmap, grad, functional_call


def dec2bin(x, bits):
    # credit to Tiana
    # https://stackoverflow.com/questions/55918468/convert-integer-to-pytorch-tensor-of-binary-bits
    # mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).to(torch.get_default_dtype())


def bin2dec(b, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1, dtype=torch.int64, device=b.device)
    return torch.sum(mask * b, -1)


@torch.no_grad()
def sample(model, n, batch=10000, max_unique=1000, *args):
    # Assuming a starting token 0
    samples = torch.zeros(1, 1, dtype=bool)
    sample_count = torch.tensor([batch], dtype=torch.int64)

    for i in range(n):
        logits = model.forward(samples, *args)  # (seq, batch, phys_dim)
        probs = logits[-1].softmax(dim=-1)  # (batch, phys_dim)

        if len(sample_count) < max_unique:
            distribution = Binomial(total_count=sample_count, probs=probs[:, 0])
            zero_count = distribution.sample()  # (batch, )
            one_count = sample_count - zero_count
            sample_count = torch.cat([zero_count, one_count], dim=0)
            mask = sample_count > 0

            n_unique = samples.shape[1]
            samples = torch.cat([torch.cat([samples, torch.zeros(1, n_unique, dtype=bool)], dim=0),
                                 torch.cat([samples, torch.ones(1, n_unique, dtype=bool)], dim=0)], dim=1)
            samples = samples.T[mask].T  # (seq, batch), with updated batch
            sample_count = sample_count[mask]  # (batch, )
        else:
            # do not generate new branches
            sampled_spins = torch.multinomial(probs, 1).bool()  # (batch, 1)
            samples = torch.cat([samples, sampled_spins.T], dim=0)
    return samples[1:], sample_count / batch  # (n, batch), (batch, )


@torch.no_grad()
def sample_without_weight(model, n, batch=1000, *args):
    samples = torch.zeros((1, batch), dtype=bool)

    for i in range(n):
        logits = model.forward(samples, *args)  # (seq, batch, phys_dim)
        probs = logits[-1].softmax(dim=-1)  # (batch, phys_dim)
        sampled_spins = torch.multinomial(probs, 1) .bool()  # (batch, 1)
        samples = torch.cat([samples, sampled_spins.T], dim=0)

    return samples[1:]


def compute_prob(model, samples, *args):
    n, batch = samples.shape
    spin_idx = samples.to(torch.int64)
    samples = torch.cat([torch.zeros((1, batch), dtype=bool), samples], dim=0)  # (n+1, batch)
    n_idx = torch.arange(n).reshape(n, 1)
    batch_idx = torch.arange(batch).reshape(1, batch)

    logits = model.forward(samples, *args)[:-1]  # (n, batch, phys_dim)
    log_prob = torch.nn.functional.log_softmax(logits, dim=-1)  # (n, batch, phys_dim)
    log_prob = log_prob[n_idx, batch_idx, spin_idx].sum(dim=0)  # (batch, )

    return log_prob


def compute_loss(model, problem, samples, sample_weight, beta):
    log_p = compute_prob(model, samples, beta)
    E = problem.E(samples.T).float()
    F = E + log_p / beta
    multiplier = (sample_weight * (F - (sample_weight * F).sum())).detach()
    loss = (multiplier * log_p).sum()
    return loss


def compute_grad(model, problem, samples, sample_weight, beta):
    with torch.no_grad():
        log_p = compute_prob(model, samples, beta)
        E = problem.E(samples.T).float()
        F = E + log_p / beta

    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    n, batch = samples.shape
    samples = torch.cat([torch.zeros((1, batch), dtype=bool), samples], dim=0)  # (n+1, batch)
    n_idx = torch.arange(n)

    def single_logp(params, buffers, sample_i):
        logits = functional_call(model, (params, buffers), args=(sample_i.unsqueeze(1), beta), strict=True)[:-1]
        log_prob = torch.nn.functional.log_softmax(logits, dim=-1).squeeze(1)  # (n, phys_dim)
        spin_idx_i = sample_i[1:].to(torch.int64)
        log_prob = log_prob[n_idx, spin_idx_i].sum()
        return log_prob

    grad_single = grad(single_logp, argnums=0)
    batched_Ok = vmap(grad_single, in_dims=(None, None, 1))(params, buffers, samples)
    batched_flattened_Ok = torch.cat([g.reshape(batch, -1) for g in batched_Ok.values()], dim=1)  # (batch, n_params)
    # mean_Ok = (batched_flattened_Ok * sample_weight.reshape(batch, 1)).sum(dim=0)  # (n_params, )
    # batched_flattened_Ok = batched_flattened_Ok - mean_Ok  # (batch, n_params)
    Fk = (sample_weight.reshape(batch, 1)
          * (F - (sample_weight * F).sum()).unsqueeze(1)
          * batched_flattened_Ok).sum(dim=0)  # (n_params, )

    pointer = 0
    for name, param in params.items():
        size = param.numel()
        param.grad = Fk[pointer:pointer + size].reshape(param.shape)
        pointer += size
    return


def compute_grad_SR(model, problem, samples, sample_weight, beta, lambd=1e-4):
    with torch.no_grad():
        log_p = compute_prob(model, samples, beta)
        E = problem.E(samples.T).float()
        F = E + log_p / beta
        F = F - (sample_weight * F).sum()

    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    n, batch = samples.shape
    samples = torch.cat([torch.zeros((1, batch), dtype=bool), samples], dim=0)  # (n+1, batch)
    n_idx = torch.arange(n)

    def single_logp(params, buffers, sample_i):
        logits = functional_call(model, (params, buffers), args=(sample_i.unsqueeze(1), beta), strict=True)[:-1]
        log_prob = torch.nn.functional.log_softmax(logits, dim=-1).squeeze(1)  # (n, phys_dim)
        spin_idx_i = sample_i[1:].to(torch.int64)
        log_prob = log_prob[n_idx, spin_idx_i].sum()
        return log_prob

    grad_single = grad(single_logp, argnums=0)
    batched_Ok = vmap(grad_single, in_dims=(None, None, 1))(params, buffers, samples)
    batched_flattened_Ok = torch.cat([g.reshape(batch, -1) for g in batched_Ok.values()], dim=1)  # (batch, n_params)
    mean_Ok = (batched_flattened_Ok * sample_weight.reshape(batch, 1)).sum(dim=0)  # (n_params, )
    batched_flattened_Ok = batched_flattened_Ok - mean_Ok  # (batch, n_params)
    # Fk = (sample_weight.reshape(batch, 1)
    #       * (F - (sample_weight * F).sum()).unsqueeze(1)
    #       * batched_flattened_Ok).sum(dim=0)  # (n_params, )
    # Skk = torch.einsum('b,bi,bj->ij', sample_weight, batched_flattened_Ok, batched_flattened_Ok)  # (n_params, n_params)
    # Skk += lambd * torch.eye(Skk.shape[0], device=Skk.device, dtype=Skk.dtype)

    # Using the SR identity in the paper:
    # A simple linear algebra identity to optimize large-scale neural network quantum states

    S = torch.einsum('ai,bi,b->ab', batched_flattened_Ok, batched_flattened_Ok, sample_weight) \
        + lambd * torch.eye(batch)  # (batch, batch)
    f1 = torch.linalg.solve(S, F)  # (batch, )
    gradients = torch.einsum('bi,b,b->i', batched_flattened_Ok, sample_weight, f1)  # (n_params, )

    pointer = 0
    for name, param in params.items():
        size = param.numel()
        param.grad = gradients[pointer:pointer + size].reshape(param.shape)
        pointer += size
    return
