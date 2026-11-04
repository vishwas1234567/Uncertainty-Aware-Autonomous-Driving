"""Run: python -m tests.test_all   (fast regression tests for every module)"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import behavior_tree as bt
from localization import make_world, simulate_scan, NDTMap, ndt_align
from planning import LatticePlanner, FState, Obstacle, optimise_trajectory, collides, check_limits
from control import RefPath, Vehicle, Stanley, AdaptiveMPC, tracking_errors


def test_bt_structure_and_safety():
    tree = bt.build_tree()
    assert tree.children[-1].name == "cruise"                       # root is total (never FAILURE)
    for seed in range(40):
        res, _ = bt.run_episode(seed)
        assert res["done"] and res["collisions"] == 0, seed


def test_bt_never_overtakes_into_oncoming():
    p = dict(lead_s=30, lead_v=4, onc_s=120, onc_v=12, cv_times=[])
    res, log = bt.run_episode(0, params=p, trace=True)
    assert res["collisions"] == 0 and res["done"]


def test_ndt_recovers_offset():
    pts, path, hd, _ = make_world(0); rng = np.random.default_rng(1)
    ndt = NDTMap(pts); pose = np.array([*path[300], hd[300]]); scan = simulate_scan(pose, pts, rng)
    est, _ = ndt_align(ndt, scan, pose + [0.8, -0.6, 0.06], iters=10)
    assert np.linalg.norm(est[:2] - pose[:2]) < 0.3


def test_planner_collision_free_and_feasible():
    pl = LatticePlanner(); obs = [Obstacle(50, 0, 4.0), Obstacle(120, 3.5, 0.0)]
    tr, n = pl.plan(FState(0, 10, 0, 0, 0, 0), obs)
    assert n > 0 and not collides(tr.t, tr.s, tr.l, tr.sd, obs)
    op = optimise_trajectory(tr, obs)
    assert not collides(op.t, op.s, op.l, op.sd, obs) and check_limits(op)["ok"]
    assert op.jerk_rms() <= tr.jerk_rms() + 1e-9                     # optimisation does not worsen jerk


def test_emergency_when_boxed_in():
    pl = LatticePlanner(); obs = [Obstacle(15, 0, 0.0), Obstacle(15, 3.5, 0.0)]
    tr, n = pl.plan(FState(0, 10, 0, 0, 0, 0), obs)
    assert tr.kind == "emergency" and tr.sd[-1] < 0.5


def test_controllers_converge():
    ref = RefPath()
    for c in (Stanley(), AdaptiveMPC(adapt=True)):
        x, y, th, _ = ref.at(0); veh = Vehicle(x, y + 1.0, th, 10.0); s = 0.0
        for _ in range(300):
            s, l, _ = ref.project(veh.x, veh.y, s); xr, yr, thr, k = ref.at(s)
            r = dict(x=xr, y=yr, th=thr, kappa=k, vN=np.full(15, 10.0), kN=np.array([ref.at(s + 1.0 * q)[3] for q in range(1, 16)]))
            d = c.command((veh.x, veh.y, veh.th), veh.v, r); yr_ = veh.step(d, 0.0)
            if hasattr(c, "estimate"): c.estimate(veh.v, yr_, d, 0.05)
        assert abs(l) < 0.1, c.name


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
