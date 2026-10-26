import numpy as np
import torch
from scipy.io import loadmat
from model import myLSTM
from runtime import tensor, require, fit, finish


def run(a):
    if a.demo:
        rng = np.random.default_rng(a.seed)
        series = []
        for i in range(16):
            t = np.arange(48)
            f = np.column_stack(
                [t / 60 + 0.03 * rng.normal(size=48) for _ in range(23)]
            )
            series.append((f, (60 + i - t)[:, None], t[:, None]))
    else:
        series = []
        rel = "battery_final/varying_LR_PINN_GE/1st_test_fold/"
        for batch in ["B1", "B2", "B3"]:
            key = "trajectory_cell_" + batch
            for c in loadmat(require(a.data_root, rel + key + ".mat"))[key].ravel():
                f = np.delete(c[:, :24], 13, axis=1)
                series.append((f, c[:, [-1]], np.arange(len(c))[:, None]))
    order = np.random.default_rng(a.seed).permutation(len(series))
    split = int(0.8 * len(series))
    train = [series[i] for i in order[:split]]
    test = [series[i] for i in order[split:]]
    # Use complete-cell splits; scaler statistics come exclusively from training cells.
    from sklearn.preprocessing import MinMaxScaler

    sx = MinMaxScaler().fit(np.concatenate([x[0] for x in train]))
    sy = MinMaxScaler().fit(np.concatenate([x[1] for x in train]))
    st = MinMaxScaler().fit(np.concatenate([x[2] for x in train]))

    def pack(data):
        out = []
        for f, y, t in data:
            for start in range(0, len(f) - 31, 32):
                sl = slice(start, start + 32)
                out.append(
                    (sx.transform(f[sl]), sy.transform(y[sl]), st.transform(t[sl]))
                )
                if len(out) >= a.max_samples:
                    break
            if len(out) >= a.max_samples:
                break
        if not out:
            raise ValueError("Need trajectories with at least 32 time steps")
        return [tensor(np.stack([x[j] for x in out])) for j in range(3)]

    x, y, t = pack(train)
    xv, yv, tv = pack(test)
    model = myLSTM(23, 32, [33, 16, 1], 1)
    # Chain rule: normalized dRUL/dt = -range(time) / range(RUL).
    slope = -float(st.data_range_[0] / max(sy.data_range_[0], 1e-12))

    def loss(idx):
        z = t[idx].detach().requires_grad_(True)
        pred = model(x[idx], z, torch.full((len(idx),), 32, dtype=torch.long))
        dy = torch.autograd.grad(pred, z, torch.ones_like(pred), create_graph=True)[0]
        ddy = torch.autograd.grad(dy, z, torch.ones_like(dy), create_graph=True)[0]
        mse = (pred - y[idx]).square().mean()
        physics = (dy - slope).square().mean()
        gradient = ddy.square().mean()
        return 0.9 * mse + 0.1 * physics + 0.01 * gradient, {
            "data": mse,
            "physics": physics,
            "gradient": gradient,
        }

    history = fit(model, loss, len(x), a)
    model.eval()
    with torch.no_grad():
        pred = model(xv, tv, torch.full((len(xv),), 32, dtype=torch.long)).numpy()
    return finish(
        model,
        a,
        history,
        pred,
        yv.numpy(),
        {
            "task": "RUL prediction",
            "normalized_rul_time_slope": slope,
            "window": 32,
            "weights": [0.9, 0.1, 0.01],
            "full_cross_validation": False,
        },
        {"features": sx, "rul": sy, "time": st},
    )
