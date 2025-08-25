import time
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

def _pick_device(gpu_index: int):
    if gpu_index == -1 or not torch.cuda.is_available():
        if gpu_index != -1 and not torch.cuda.is_available():
            print("WARNING: CUDA not available, using CPU (will be very slow).")
        return torch.device("cpu")
    return torch.device("cuda", gpu_index)

def _to_tensor(x, device, dtype=torch.float):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)

def _exp_dirichlet_like(s: int, n: int, m: int, M: int):
    rng = np.random.default_rng()
    base = rng.gamma(shape=1.0, scale=1.0, size=(m, s))
    base = base / base.mean(axis=1, keepdims=True)
    w = np.repeat(base, M, axis=0)
    if s == n:
        w_m = base
    elif n % s == 0:
        w_m = np.repeat(base, n // s, axis=1)
    else:
        tile = int(np.ceil(n / s))
        tmp = np.repeat(base, tile, axis=1)[:, :n]
        w_m = tmp
    return w, w_m

def _concat_noise(w, z_dim: int, m: int, M: int):
    if z_dim <= 0:
        return w
    z = np.random.normal(loc=0.0, scale=1.0, size=(m * M, z_dim))
    return np.concatenate([w, z], axis=1)

def _select_columns_k_grouped(t: torch.Tensor, k: int, p: int):
    nm, pk = t.shape
    assert pk == p * k
    return t.view(nm, p, k).reshape(nm * p, k)

def _uniform_pick_k(num: int, k: int, device):
    return torch.randint(low=0, high=k, size=(num, 1), device=device, dtype=torch.long)

class GeneratorNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, out_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        return self.out(x)

def _likelihood_factory(dist_name: str):
    dist_name = dist_name.strip()
    if dist_name == "Gaussian location":
        def _transform(theta): return theta
        def _build(mean, scale): return torch.distributions.Normal(loc=mean, scale=scale)
        return _transform, _build
    if dist_name == "Gaussian scale":
        def _transform(theta): return torch.exp(theta)
        def _build(scale, mean): return torch.distributions.Normal(loc=mean, scale=scale)
        return _transform, _build
    if dist_name == "Poisson":
        def _transform(theta): return torch.exp(theta)
        def _build(rate, _): return torch.distributions.Poisson(rate)
        return _transform, _build
    if dist_name == "Gamma rate":
        def _transform(theta): return torch.exp(theta)
        def _build(rate, shape): return torch.distributions.Gamma(concentration=shape, rate=rate)
        return _transform, _build
    if dist_name == "Gamma shape":
        def _transform(theta): return torch.exp(theta)
        def _build(shape, rate): return torch.distributions.Gamma(concentration=shape, rate=rate)
        return _transform, _build
    if dist_name == "Binomial":
        def _transform(theta): return torch.sigmoid(theta)
        def _build(probs, total_count): return torch.distributions.Binomial(total_count=total_count, probs=probs)
        return _transform, _build
    if dist_name == "Uniform":
        def _transform(theta): return torch.exp(theta)
        def _build(high_delta, low): return torch.distributions.Uniform(low=low, high=low + high_delta)
        return _transform, _build
    if dist_name == "Weibull scale":
        def _transform(theta): return torch.exp(theta)
        def _build(scale, concentration): return torch.distributions.Weibull(scale=scale, concentration=concentration)
        return _transform, _build
    raise ValueError(f"Unknown distribution: {dist_name}")

def FGBMLE_Loading(
    Option, Save_file, Gpu_ind, N1, M, Num_it, Verb, Tol, LrDecay, Boot_size,
    Tau0, P, Q, Dist, S, n1, Zm, Hidden_size, Param, Lr, X, Y, Lrpower
):
    device = _pick_device(int(Gpu_ind))
    if device.type == "cpu":
        print("WARNING: CPU computing will be very slow!")

    p = int(P)
    k = int(Q)
    dist_name = str(Dist)
    s = int(S)
    n = int(n1)
    z_dim = int(Zm)
    hidden = int(Hidden_size)
    param = float(Param)
    lr0 = float(Lr)
    lr = lr0
    option = str(Option).strip()

    N = int(N1)
    m = int(M)
    max_iter_train = int(Num_it)
    verbose = int(Verb) == 1
    tol = float(Tol)
    lr_decay_flag = int(LrDecay) == 1
    lr_power = float(Lrpower)
    bsz = int(Boot_size)

    in_dim = s + z_dim
    out_dim = p * k
    net = GeneratorNet(in_dim=in_dim, hidden=hidden, out_dim=out_dim).to(device)
    net.load_state_dict(torch.load(Save_file, map_location=device))
    print("Successfully loaded trained Generator!")

    transform_param, build_dist = _likelihood_factory(dist_name)

    if option == "Train":
        pass

    if option == "Sample":
        net.eval()
        t_start = time.perf_counter()
        with torch.no_grad():
            tau0 = _to_tensor(Tau0, device=device, dtype=torch.float)
            tau0 = tau0 / tau0.sum()
            comp_counts = torch.round(bsz * tau0).long()
            diff = bsz - comp_counts.sum()
            if diff != 0:
                idx = torch.argmax(comp_counts)
                comp_counts[idx] += diff
            comp_ids = torch.repeat_interleave(torch.arange(k, device=device), comp_counts)
            comp_ids = comp_ids[:bsz]
            sample_ids = torch.repeat_interleave(comp_ids, p)
            w_np, _ = _exp_dirichlet_like(s=s, n=n, m=bsz, M=1)
            g_input_np = _concat_noise(w_np, z_dim=z_dim, m=bsz, M=1)
            g_input = _to_tensor(g_input_np, device)
            raw = net(g_input).view(bsz, p, k)
            flat = raw.reshape(bsz * p, k)
            theta = torch.gather(flat, 1, sample_ids.view(-1, 1)).view(bsz, p)
            theta_trans = transform_param(theta).cpu().numpy()
            Theta_dist = pd.DataFrame(theta_trans)
        print("Generation of Bootstrap samples Done!")
        gen_time = time.perf_counter() - t_start
        return Theta_dist, gen_time, p

    raise ValueError("Option must be 'Train' or 'Sample'")
