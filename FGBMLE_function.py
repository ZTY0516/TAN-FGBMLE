import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import sys

def _device_from_index(gpu_index):
    if gpu_index == -1:
        print("WARNING: CPU computing would be very slow!")
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda", int(gpu_index))
    print("WARNING: CPU computing would be very slow!")
    return torch.device("cpu")

def _to_tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.float)
    return torch.as_tensor(np.asarray(x), dtype=torch.float, device=device)

def _dir_like(s, n, m, M):
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

def _make_input(zdim, m, M, w):
    if zdim <= 0:
        return w
    z = np.random.normal(0.0, 1.0, size=(m * M, zdim))
    return np.concatenate([w, z], axis=1)

class _Gen(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.a = nn.ReLU()
        self.l1 = nn.Linear(in_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, out_dim)
    def forward(self, x):
        x = self.a(self.l1(x))
        x = self.a(self.l2(x))
        x = self.a(self.l3(x))
        return self.out(x)

def FGBMLE_train(X, Y, S, Hidden_size, Num_it, L, M, Q, Gpu_ind, N1, n1, P, Dist,
                 Param, Lr0, Verb, Tol, LrDecay, Lrpower, Save, Save_path):
    device = _device_from_index(Gpu_ind)

    s = int(S)
    n = int(n1)
    m = int(M)
    zm = int(Q)
    N = int(N1)
    p = int(P)
    k = int(L)
    save_flag = int(Save)
    param = float(Param)
    dist = str(Dist)
    iters = int(Num_it)
    hidden = int(Hidden_size)
    lr0 = float(Lr0)
    lr = lr0
    verbose = int(Verb)
    tol = float(Tol)
    decay = int(LrDecay)
    lr_pow = float(Lrpower)

    X = _to_tensor(X, device)
    Y = _to_tensor(Y, device).view(n, 1)

    if k == 1:
        em0 = 0
        method = "GMS Algorithm 1"
        print("Training G via GMS Algorithm 1!")
    else:
        em0 = 1
        method = "Two-stage Algorithm"
        print("Training G via Two-stage Algorithm")

    net = _Gen(s + zm, hidden, p * k).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    losses = torch.zeros(iters)

    t0 = time.perf_counter()
    for it in range(iters):
        if decay == 1:
            lr = lr0 / ((it + 1.0) ** lr_pow)
            for g in opt.param_groups:
                g["lr"] = lr

        w_np, w_m_np = _dir_like(s=s, n=n, m=N, M=m)
        g_in = _to_tensor(_make_input(zm, N, m, w_np), device)

        raw = net(g_in)
        if dist == "Gaussian location":
            out_param = raw
        elif dist in {"Gaussian scale", "Poisson", "Gamma rate", "Gamma shape", "Uniform", "Weibull scale"}:
            out_param = torch.exp(raw)
        elif dist == "Binomial":
            out_param = torch.sigmoid(raw)
        else:
            raise ValueError(f"Unknown distribution: {dist}")

        grouped = out_param.view(N * m, p, k).reshape(N * m * p, k)
        comp_idx = torch.randint(0, k, (N * m * p, 1), device=device)
        Theta = torch.gather(grouped, 1, comp_idx).view(N, m, p)

        proj = torch.matmul(Theta, X.transpose(0, 1)).reshape(N, n * m)
        order = torch.arange(0, n * m, device=device).view(m, n).transpose(1, 0).reshape(-1)
        result = torch.index_select(proj, 1, order).view(N, n, m)

        w_m = _to_tensor(w_m_np, device).view(N, n)

        if dist == "Gaussian location":
            d = torch.distributions.Normal(loc=result, scale=param)
            dens = torch.exp(d.log_prob(Y))
            score = w_m * torch.log(dens.mean(dim=2))
        elif dist == "Gaussian scale":
            d = torch.distributions.Normal(loc=param, scale=result)
            dens = torch.exp(d.log_prob(Y))
            score = w_m * torch.log(dens.mean(dim=2))
        elif dist == "Poisson":
            d = torch.distributions.Poisson(rate=result)
            dens = torch.exp(d.log_prob(Y)) + 1e-20
            score = w_m * torch.log(dens.mean(dim=2))
        elif dist == "Gamma rate":
            d = torch.distributions.Gamma(concentration=param, rate=result)
            dens = torch.exp(d.log_prob(Y))
            score = w_m * torch.log(dens.mean(dim=2))
        elif dist == "Gamma shape":
            d = torch.distributions.Gamma(concentration=result, rate=param)
            dens = torch.exp(d.log_prob(Y))
            score = w_m * torch.log(dens.mean(dim=2))
        elif dist == "Binomial":
            d = torch.distributions.Binomial(total_count=param, probs=result)
            dens = torch.exp(d.log_prob(Y))
            score = w_m * torch.log(dens.mean(dim=2))
        elif dist == "Uniform":
            d = torch.distributions.Uniform(low=param, high=result)
            dens = torch.exp(d.log_prob(Y))
            score = w_m * torch.log(dens.mean(dim=2))
        elif dist == "Weibull scale":
            d = torch.distributions.Weibull(scale=result, concentration=param)
            dens = torch.exp(d.log_prob(Y))
            score = w_m * torch.log(dens.mean(dim=2))

        loss = -score.sum(dim=1).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        losses[it] = loss.item()
        if (it + 1) % 10 == 0 and verbose == 1:
            pct = (it + 1) * 100.0 / iters
            bar_len = max(int(pct / 5) - 1, 0)
            bar = "-" * bar_len + ">"
            bar = (bar + " " * (20 - len(bar)))[:20]
            elapsed = time.perf_counter() - t0
            print(
                f"\r[{it+1}/{iters}] Progress: [{bar}] {pct:5.1f}% "
                f"Current/Initial Loss: {losses[it]:.2f}/{losses[0]:.0f}, Method: {method}, "
                f"Learning rate: {lr:.6f}, Training time: {elapsed:.2f}",
                end=""
            )
            sys.stdout.flush()

    gen_time = time.perf_counter() - t0

    em_t0 = time.perf_counter()
    max_em = 5000
    tau = torch.ones(k, device=device) / k
    if em0 == 1:
        pz_buf = np.zeros((n, k))
        Tau_hist = np.zeros((max_em, k))
        it = 1
        sub_m = 20
        Tau_hist[0, :] = tau.cpu().numpy()
        y_tile = torch.cat([Y.view(1, n)] * sub_m, dim=0)
        while it <= max_em:
            for j in range(k):
                W, _ = _dir_like(s, n, m=sub_m, M=1)
                g_in = _to_tensor(_make_input(zm, sub_m, 1, W), device)
                with torch.no_grad():
                    out = net(g_in).view(sub_m, p * k)
                    out = torch.index_select(out, 1, torch.arange(0, k * p, device=device)
                                             .view(k, p).transpose(1, 0).reshape(-1))
                    Theta_j = out.view(sub_m, p, k)[:, :, j]
                    proj_j = torch.matmul(Theta_j, X.transpose(0, 1))
                if dist == "Gaussian location":
                    d = torch.distributions.Normal(loc=proj_j, scale=param)
                    prob = torch.exp(d.log_prob(y_tile)).cpu().numpy()
                elif dist == "Gaussian scale":
                    d = torch.distributions.Normal(loc=param, scale=proj_j)
                    prob = torch.exp(d.log_prob(y_tile)).cpu().numpy()
                elif dist == "Poisson":
                    d = torch.distributions.Poisson(rate=torch.exp(proj_j))
                    prob = torch.exp(d.log_prob(y_tile)).cpu().numpy() + 1e-20
                elif dist == "Gamma rate":
                    d = torch.distributions.Gamma(concentration=param, rate=torch.exp(proj_j))
                    prob = torch.exp(d.log_prob(y_tile)).cpu().numpy()
                elif dist == "Gamma shape":
                    d = torch.distributions.Gamma(concentration=torch.exp(proj_j), rate=param)
                    prob = torch.exp(d.log_prob(y_tile)).cpu().numpy()
                elif dist == "Binomial":
                    d = torch.distributions.Binomial(total_count=param, probs=torch.sigmoid(proj_j))
                    prob = torch.exp(d.log_prob(y_tile)).cpu().numpy()
                elif dist == "Uniform":
                    d = torch.distributions.Uniform(low=param, high=torch.exp(proj_j))
                    prob = torch.exp(d.log_prob(y_tile)).cpu().numpy()
                elif dist == "Weibull scale":
                    d = torch.distributions.Weibull(scale=torch.exp(proj_j), concentration=param)
                    prob = torch.exp(d.log_prob(y_tile)).cpu().numpy()
                pz_buf[:, j] = prob.mean(axis=0).reshape(n, -1).mean(axis=1)
            denom = np.sum(pz_buf, axis=1, keepdims=True)
            pz = pz_buf / np.maximum(denom, 1e-12)
            tau = torch.as_tensor(pz.mean(axis=0), dtype=torch.float, device=device)
            Tau_hist[it, :] = tau.cpu().numpy()
            if np.max(np.abs(Tau_hist[it, :] - Tau_hist[it - 1, :])) < tol:
                break
            it += 1
    em_time = time.perf_counter() - em_t0

    X_copy = X.cpu().numpy()
    Y_copy = Y.cpu().numpy()

    if save_flag == 1:
        torch.save(net.state_dict(), Save_path)

    return net, tau.cpu().numpy(), p, k, dist, s, n, zm, gen_time, em_time, hidden, param, lr, N, m, X_copy, Y_copy, tol, decay
