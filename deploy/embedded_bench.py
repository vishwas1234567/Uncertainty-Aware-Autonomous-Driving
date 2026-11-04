"""
Per-module latency benchmark (host CPU). Used to size the real-time budget before porting to an embedded
target: run the same script ON the target to get target numbers (none are reported for hardware not tested).
Run: python deploy/embedded_bench.py  -> results/latency.json
"""
import os, sys, json, time, platform
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from localization import make_world, simulate_scan, NDTMap, NDTParticleFilter, ndt_align
from planning import LatticePlanner, FState, Obstacle, optimise_trajectory
from control import AdaptiveMPC, RefPath
import behavior_tree as bt


def timeit(fn, n=60, warm=5):
    for _ in range(warm): fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e3)
    ts = np.array(ts)
    return dict(p50_ms=float(np.percentile(ts, 50)), p95_ms=float(np.percentile(ts, 95)), max_ms=float(ts.max()))


def main():
    rng = np.random.default_rng(0)
    pts, path, heading, _ = make_world(0)
    ndt = NDTMap(pts); pose = np.array([*path[200], heading[200]]); scan = simulate_scan(pose, pts, rng)
    res = {"host": platform.processor() or platform.machine(), "python": platform.python_version(),
           "numpy": np.__version__, "cpu_count": os.cpu_count()}
    for n in (100, 300, 1000):
        pf = NDTParticleFilter(ndt, pose, n=n, use_injection=False)
        res[f"pf_update_{n}_particles"] = timeit(lambda: (pf.predict(5.0, 0.0, 0.1), pf.update(scan)))
    res["ndt_align_6_iters"] = timeit(lambda: ndt_align(ndt, scan, pose + [0.5, 0.3, 0.03], iters=6))
    pf = NDTParticleFilter(ndt, pose, n=300, use_injection=True)
    res["ndt_pf_full_cycle_300"] = timeit(lambda: (pf.predict(5.0, 0.0, 0.1), pf.update(scan)))
    pl = LatticePlanner(); obs = [Obstacle(60, 0, 5.0), Obstacle(140, 3.5, 0.0)]; st = FState(0, 10, 0, 0, 0, 0)
    res["lattice_plan"] = timeit(lambda: pl.plan(st, obs), n=40)
    tr, _ = pl.plan(st, obs)
    res["qp_trajectory_opt"] = timeit(lambda: optimise_trajectory(tr, obs), n=40)
    ref = RefPath(); mpc = AdaptiveMPC()
    r = dict(x=0.0, y=0.5, th=0.0, kappa=0.0, vN=np.full(15, 10.0), kN=np.zeros(15))
    res["mpc_step"] = timeit(lambda: mpc.command((0.0, 0.5, 0.0), 10.0, r), n=200)
    w = bt.World(0); tree = bt.build_tree()
    def tick():
        bb = bt.bb_from_world(w); bb["cmd"] = dict(speed=0.0, lane=0); tree.tick(bb)
    res["bt_tick"] = timeit(tick, n=500)
    budget = {"localisation_10Hz": ("ndt_pf_full_cycle_300", 100.0), "planning_2Hz": ("lattice_plan", 500.0),
              "control_20Hz": ("mpc_step", 50.0), "decision_10Hz": ("bt_tick", 100.0)}
    res["budget_check"] = {k: dict(module=m, p95_ms=res[m]["p95_ms"], budget_ms=b, ok=res[m]["p95_ms"] < b) for k, (m, b) in budget.items()}
    os.makedirs("results", exist_ok=True); json.dump(res, open("results/latency.json", "w"), indent=1)
    for k, v in res.items():
        if isinstance(v, dict) and "p50_ms" in v:
            print(f"{k:28s} p50 {v['p50_ms']:8.2f} ms   p95 {v['p95_ms']:8.2f} ms")
    for k, v in res["budget_check"].items():
        print(f"budget {k:20s} p95 {v['p95_ms']:7.2f} / {v['budget_ms']:.0f} ms  {'OK' if v['ok'] else 'EXCEEDED'}")


if __name__ == "__main__":
    main()
