import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time

def _pick_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def _to_tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.float)
    return torch.as_tensor(np.asarray(x), dtype=torch.float, device=device)

def _weights(s, n, m, M):
    rng = np.random.default_rng()
    base = rng.gamma(1.0, 1.0, size=(m, s))
    base = base / base.mean(axis=1, keepdims=True)
    w = np.repeat(base, M, axis=0)
    if s == n:
        w_m = base
    elif n % s == 0:
        w_m = np.repeat(base, n // s, axis=1)
    else:
        t = int(np.ceil(n / s))
        w_m = np.repeat(base, t, axis=1)[:, :n]
    return w, w_m

def _make_input(z_dim, m, M, w):
    if z_dim <= 0:
        return w
    z = np.random.normal(0.0, 1.0, size=(m * M, z_dim))
    return np.concatenate([w, z], axis=1)

def FGBMLE_Sampling(fit, Boot_size):
    device = _pick_device()
    net = fit[0].to(device).eval()
    tau = torch.as_tensor(fit[1], dtype=torch.float, device=device)
    p = int(fit[2])
    k = int(fit[3])
    dist = str(fit[4])
    s = int(fit[5])
    n = int(fit[6])
    zdim = int(fit[7])
    param = float(fit[11]) if len(fit) > 11 else None
    bsz = int(Boot_size)

    t0 = time.perf_counter()
    with torch.no_grad():
        tau = tau / tau.sum()
        comp_ids = torch.multinomial(tau, num_samples=bsz, replacement=True)
        sample_ids = torch.repeat_interleave(comp_ids, p)
        w, _ = _weights(s=s, n=n, m=bsz, M=1)
        g_in = _to_tensor(_make_input(zdim, bsz, 1, w), device)
        out = net(g_in).view(bsz, p, k).reshape(bsz * p, k)
        theta = torch.gather(out, 1, sample_ids.view(-1, 1).to(device)).view(bsz, p).cpu().numpy()
        if dist == "Gaussian location":
            theta_df = pd.DataFrame(theta)
        elif dist in {"Gaussian scale", "Poisson", "Gamma rate", "Gamma shape", "Weibull scale"}:
            theta_df = pd.DataFrame(np.exp(theta))
        elif dist == "Binomial":
            theta_df = pd.DataFrame(1.0 / (1.0 + np.exp(-theta)))
        elif dist == "Uniform":
            if param is None:
                raise ValueError("param is required for Uniform distribution.")
            theta_df = pd.DataFrame(np.exp(theta) + param)
        else:
            raise ValueError(f"Unknown distribution: {dist}")
    elapsed = time.perf_counter() - t0
    return theta_df, elapsed, p
