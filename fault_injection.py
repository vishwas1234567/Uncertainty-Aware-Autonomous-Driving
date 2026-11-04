"""
Fault-injection campaigns.
  (A) Localisation: scan blackout, high outlier rate, gyro bias, high sensor noise, map change.
  (B) Decision making: perception faults (latency, dropout, speed bias, circulating-traffic timing error)
      fed to the behaviour tree, evaluated with and without a degraded-mode supervisor.
Run: python fault_injection.py   (writes results/fault_injection.json)
"""
import json, os, copy
import numpy as np
import behavior_tree as bt
import run_localization as rl
from localization import make_world

# ------------------------------------------------------------------ (A) localisation
def localisation_campaign(n_seeds=5):
    map_pts, path, heading, total = make_world(seed=0)
    # map change: delete every third 12 m building block on both sides (construction / demolished facade)
    s_map = np.cumsum(np.r_[0, np.linalg.norm(np.diff(path, axis=0), axis=1)])
    nearest = np.argmin(((map_pts[:, None, :] - path[None, ::4, :]) ** 2).sum(-1), axis=1) * 4
    blk = (s_map[nearest] // 12.0).astype(int)
    changed = map_pts[blk % 3 != 0]
    changed2 = map_pts[blk % 3 == 0]
    faults = {
        "baseline": {},
        "blackout_5s": dict(blackout=(20.0, 25.0)),
        "outliers_40pct": dict(outlier_frac=0.40),
        "gyro_bias_0.03": dict(gyro_bias=0.03),
        "sensor_noise_0.3m": dict(sigma=0.30),
        "map_change_33pct": dict(world_pts=changed),
        "blackout_20s": dict(blackout=(15.0, 35.0)),
        "outliers_70pct": dict(outlier_frac=0.70),
        "map_change_67pct": dict(world_pts=changed2),
        "severe_combo": dict(blackout=(15.0, 30.0), outlier_frac=0.5, gyro_bias=0.05, world_pts=changed),
    }
    res = {}
    for name, f in faults.items():
        rows = []
        for seed in range(n_seeds):
            truth, est = rl.drive(seed, map_pts, path, heading, methods=("dr", "ndt", "pf", "ndtpf"), faults=f)
            for m in ("ndt", "ndtpf"):
                pos, _ = rl.errors(truth, est[m])
                rows.append((m, np.sqrt((pos ** 2).mean()), pos.max(), float(pos[-50:].mean() < 1.0)))
            # divergence detector quality: score<0.25 for >=5 consecutive steps vs true error>2 m
            pos, _ = rl.errors(truth, est["ndtpf"]); low = np.r_[est["scores"][:1], est["scores"]] < 0.25
            run = np.convolve(low, np.ones(5), "same") >= 5
            bad = pos > 2.0
            rows.append(("detect", float((run & bad).sum()), float(bad.sum()), float((run & ~bad).sum()), float((~bad).sum())))
        out = {}
        for m in ("ndt", "ndtpf"):
            r = np.array([x[1:] for x in rows if x[0] == m]); out[m] = dict(rmse=float(r[:, 0].mean()), max=float(r[:, 1].mean()), converged=float(r[:, 2].mean()))
        d = np.array([x[1:5] for x in rows if x[0] == "detect"]).sum(0)
        out["detector"] = dict(steps_err_gt_2m=int(d[1]), recall_when_err_gt_2m=(float(d[0] / d[1]) if d[1] else None),
                               false_alarm_rate=float(d[2] / d[3]))
        res[name] = out
        print(f"{name:20s} NDT rmse {out['ndt']['rmse']:6.2f} max {out['ndt']['max']:6.2f} conv {out['ndt']['converged']*100:3.0f}% | "
              f"NDT-PF rmse {out['ndtpf']['rmse']:6.2f} max {out['ndtpf']['max']:6.2f} conv {out['ndtpf']['converged']*100:3.0f}% | "
              f"det recall {('n/a' if out['detector']['recall_when_err_gt_2m'] is None else format(out['detector']['recall_when_err_gt_2m'], '.2f'))} FA {out['detector']['false_alarm_rate']:.3f}")
    return res


# ------------------------------------------------------------------ (B) decision making
class Perceived:
    """What the behaviour tree believes. Built from the true world + a fault model; true world is used for physics."""
    def __init__(self, w, fault, hist, rng):
        for k, v in w.__dict__.items():
            setattr(self, k, copy.copy(v))
        self.degraded_flag = False
        kind = fault.get("kind")
        if kind == "latency":                                   # stale object list (positions from `lag` seconds ago)
            lag = int(fault["lag"] / 0.1); old = hist[max(0, len(hist) - 1 - lag)]
            self.lead_s, self.onc_s = old["lead_s"], old["onc_s"]
            self.degraded_flag = True
        elif kind == "dropout":                                 # oncoming car not detected during a window
            t0, t1 = fault["window"]
            if t0 <= w.t < t1:
                self.onc_s = 1e6; self.degraded_flag = True
        elif kind == "speed_bias":                              # oncoming speed under-estimated
            self.onc_v = w.onc_v * fault["factor"]
        elif kind == "cv_timing":                               # circulating traffic timing error (s)
            self.cv_times = w.cv_times + rng.normal(0, fault["sigma"], len(w.cv_times))


def run_faulted(seed, fault, supervisor):
    rng = np.random.default_rng(10_000 + seed)
    w, tree = bt.World(seed), bt.build_tree(); hist = []
    while w.s < bt.ROUTE_END and w.t < 150:
        hist.append(dict(lead_s=w.lead_s, onc_s=w.onc_s))
        pw = Perceived(w, fault, hist, rng)
        bb = bt.bb_from_world(pw); bb["cmd"] = dict(speed=0.0, lane=0)
        bb["degraded"] = supervisor and pw.degraded_flag
        tree.tick(bb)
        # commands act on the TRUE world; overtake bookkeeping (counters) lives on the true world
        w.step(bb["cmd"])
    return dict(done=w.s >= bt.ROUTE_END, coll=w.collisions > 0)


def decision_campaign(n=200):
    faults = {
        "none": dict(),
        "latency_0.5s": dict(kind="latency", lag=0.5),
        "latency_1.0s": dict(kind="latency", lag=1.0),
        "oncoming_dropout_3s": dict(kind="dropout", window=(8.0, 11.0)),
        "oncoming_speed_x0.6": dict(kind="speed_bias", factor=0.6),
        "cv_timing_sigma_1.5s": dict(kind="cv_timing", sigma=1.5),
    }
    res = {}
    for name, f in faults.items():
        res[name] = {}
        for sup in (False, True):
            r = [run_faulted(s, f, sup) for s in range(n)]
            res[name]["supervisor" if sup else "no_supervisor"] = dict(
                collision_rate=float(np.mean([x["coll"] for x in r])), completion=float(np.mean([x["done"] for x in r])))
        a, b = res[name]["no_supervisor"], res[name]["supervisor"]
        print(f"{name:24s} collisions {a['collision_rate']*100:5.1f}% -> {b['collision_rate']*100:5.1f}% (with supervisor) | completion {a['completion']*100:5.1f}% / {b['completion']*100:5.1f}%")
    return res


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    print("== (A) localisation faults, 5 seeds ==")
    A = localisation_campaign()
    print("\n== (B) decision-making faults, 200 episodes each ==")
    B = decision_campaign()
    json.dump(dict(localisation=A, decision=B), open("results/fault_injection.json", "w"), indent=1)
