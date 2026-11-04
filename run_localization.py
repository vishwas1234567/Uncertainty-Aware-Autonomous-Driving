"""
Localization experiment: dead reckoning vs NDT-only vs plain PF vs NDT-PF (ours), over several seeds.
Run:  python run_localization.py [--plot]
"""
import os, sys
import numpy as np
from localization import (make_world, simulate_scan, NDTMap, ndt_align,
                          NDTParticleFilter, wrap)

DT, SPEED = 0.1, 5.0  # 5 m/s * 0.1 s = 0.5 m = map path resolution


def drive(seed, map_pts, path, heading, methods=("dr", "ndt", "pf", "ndtpf"),
          init_offset=(1.0, -1.0, 0.08), init_std=(2.0, 2.0, 0.15), faults=None):
    faults = faults or {}
    world_pts = faults.get("world_pts", map_pts)          # what the lidar really sees (may differ from the HD map)
    rng = np.random.default_rng(1000 + seed)
    ndt = NDTMap(map_pts)
    step = 0.5
    stride = max(1, int(round(SPEED * DT / step)))
    idxs = np.arange(0, len(path), stride)
    truth = np.c_[path[idxs], heading[idxs]]
    yaw_rate = wrap(np.diff(truth[:, 2], prepend=truth[0, 2])) / DT
    yaw_rate[0] = 0

    init = truth[0] + np.array(init_offset)  # imperfect initial guess
    est = {m: [init.copy()] for m in methods}
    pf = NDTParticleFilter(ndt, init, init_std=init_std, n=300, seed=seed, use_injection=False)
    ours = NDTParticleFilter(ndt, init, init_std=init_std, n=300, seed=seed, use_injection=True)
    dr, nd = init.copy(), init.copy()
    bias = faults.get("gyro_bias", 0.01)  # systematic gyro bias (rad/s) -> dead reckoning drifts
    scores = []

    for k in range(1, len(truth)):
        v = SPEED * 1.02 + rng.normal(0, 0.10)
        w = yaw_rate[k] + bias + rng.normal(0, 0.02)
        for pose in (dr, nd):
            pose[0] += v * DT * np.cos(pose[2] + 0.5 * w * DT)
            pose[1] += v * DT * np.sin(pose[2] + 0.5 * w * DT)
            pose[2] = wrap(pose[2] + w * DT)
        t_now = k * DT
        bo = faults.get("blackout")
        if bo and bo[0] <= t_now < bo[1]:
            scan = np.zeros((0, 2))                                  # sensor dropout
        else:
            scan = simulate_scan(truth[k], world_pts, rng, sigma=faults.get("sigma", 0.10),
                                 outlier_frac=faults.get("outlier_frac", 0.10))
        if "ndt" in methods:
            nd, _ = ndt_align(ndt, scan, nd, iters=8)
        if "pf" in methods:
            pf.predict(v, w, DT); pf.update(scan)
        if "ndtpf" in methods:
            ours.predict(v, w, DT); ours.update(scan)
        scores.append(ours.last_score)
        est["dr"].append(dr.copy()); est["ndt"].append(nd.copy())
        est["pf"].append(pf.est.copy()); est["ndtpf"].append(ours.est.copy())
    out = {m: np.array(est[m]) for m in methods}
    out["scores"] = np.array(scores)
    return truth, out


def errors(truth, e):
    pos = np.linalg.norm(e[:, :2] - truth[:, :2], axis=1)
    yaw = np.abs(wrap(e[:, 2] - truth[:, 2]))
    return pos, np.degrees(yaw)


if __name__ == "__main__":
    map_pts, path, heading, total = make_world(seed=0)
    print(f"HD map: {len(map_pts)} points, route length {total:.0f} m")
    names = {"dr": "Dead reckoning", "ndt": "NDT only", "pf": "PF (NDT likelihood)",
             "ndtpf": "NDT-PF (ours)"}
    scenarios = {"nominal init (1.4 m, 5 deg off)": dict(init_offset=(1.0, -1.0, 0.08), init_std=(2.0, 2.0, 0.15)),
                 "hard init (5 m, 20 deg off)": dict(init_offset=(4.0, -3.0, 0.35), init_std=(4.0, 4.0, 0.35))}
    n_seeds = 5
    last = None
    for sname, kw in scenarios.items():
        agg = {m: [] for m in names}
        for seed in range(n_seeds):
            truth, est = drive(seed, map_pts, path, heading, **kw)
            if last is None:
                last = (truth, est)
            for m in names:
                pos, yaw = errors(truth, est[m])
                agg[m].append((np.sqrt((pos ** 2).mean()), np.percentile(pos, 95), pos.max(),
                               np.sqrt((yaw ** 2).mean()), float(pos[-50:].mean() < 1.0)))
        print(f"\nScenario: {sname}  ({n_seeds} seeds, mean of per-run metrics)")
        print(f"{'method':24s} {'RMSE pos [m]':>13s} {'p95 [m]':>9s} {'max [m]':>9s} {'RMSE yaw [deg]':>15s} {'converged':>10s}")
        for m, label in names.items():
            a = np.array(agg[m]).mean(0)
            print(f"{label:24s} {a[0]:13.2f} {a[1]:9.2f} {a[2]:9.2f} {a[3]:15.2f} {a[4]*100:9.0f}%")

    if "--plot" in sys.argv:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        truth, est = last
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        ax[0].scatter(map_pts[::3, 0], map_pts[::3, 1], s=1, c="lightgray")
        ax[0].plot(truth[:, 0], truth[:, 1], "k", lw=1, label="truth")
        ax[0].plot(est["dr"][:, 0], est["dr"][:, 1], "r--", lw=1, label="dead reckoning")
        ax[0].plot(est["ndtpf"][:, 0], est["ndtpf"][:, 1], "g", lw=1, label="NDT-PF")
        ax[0].set_aspect("equal"); ax[0].legend(); ax[0].set_title("Trajectory over HD map")
        t = np.arange(len(truth)) * DT
        for m, c in (("dr", "r"), ("ndt", "b"), ("pf", "orange"), ("ndtpf", "g")):
            ax[1].plot(t, errors(truth, est[m])[0], c, label=names[m])
        ax[1].set_yscale("log"); ax[1].set_xlabel("time [s]"); ax[1].set_ylabel("position error [m]")
        ax[1].legend(); ax[1].set_title("Localization error")
        plt.tight_layout(); os.makedirs("figures", exist_ok=True); plt.savefig("figures/localization_results.png", dpi=120)
        print("saved figures/localization_results.png")
