import pickle
import numpy as np


def decoder_mean_of_lambdas(mean_lambda_path):
    file = open(mean_lambda_path, "rb")
    values = pickle.load(file)
    lmb = values[0][1]
    count = 0
    value = 0
    means = []
    for e, l in values:
        if lmb == l:
            value += e
            count += 1
        else:
            means.append([value/count, lmb])
            value = 0
            count = 0
            lmb = l
    means.append([value/count, lmb])
    return means


def latent_mean_lambdas(latent_lambda):
    file = open(latent_lambda, "rb")
    values = pickle.load(file)
    means = []
    seen = []
    for t, v in values:
        t = t.item()
        if t in seen:
            continue
        seen.append(t)
        mean = []
        for t2, v2 in values:
            if t == t2.item():
                mean.append(v2.item())
        means.append([np.mean(mean), t])
    return means


def get_lmbda(ts, eps_error):
    v_out = eps_error[-1][0]
    for v, t in eps_error:
        if ts <= t:
            v_out = v
            break
    return v_out


def get_mean(lmb, mean_vals):
    mean_out = mean_vals[0][0]
    for v, l in mean_vals:
        if lmb >= l:
            mean_out = v
            break
    return mean_out


def lambda_t(t, lambda_ts_val):
    for ts, alpha_ts, lambda_ts in lambda_ts_val:
        if t <= ts:
            return lambda_ts


def mean_shift_latent(t, mean_shift_val):
    for ts, mean_shift_ts in mean_shift_val:
        if t <= ts:
            return mean_shift_ts


def std_shift_latent(t, std_shift_val):
    for ts, std_shift_ts in std_shift_val:
        if t <= ts:
            return std_shift_ts
