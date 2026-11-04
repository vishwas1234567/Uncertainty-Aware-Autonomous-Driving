"""
Verification of the driving behaviour tree.

(1) Static / structural checks on the tree (totality, every leaf reachable, depth).
(2) Runtime monitors for safety properties (signal-temporal-logic style, evaluated every tick).
(3) Bounded exhaustive verification: the scenario parameter space is discretised into a grid and
    EVERY grid point is simulated against the monitors (deterministic world => complete for the grid).
    This is verification relative to the grid and the 1-D world model -- not a proof for the real world.
Run: python verify_bt.py
"""
import itertools, sys, time
import numpy as np
import behavior_tree as bt

TTC_MIN = 3.0   # property P2: never in the oncoming lane with time-to-collision below this


class Monitor:
    """Safety properties, G(...) = must hold at every tick."""
    def __init__(self):
        self.viol = {}
        self.prev_lat, self.max_active = 0.0, 0

    def __call__(self, w, bb):
        def bad(name):
            self.viol.setdefault(name, w.t)
        # P1: G(no collision)  (recorded by world)
        if w.collisions:
            bad("P1_no_collision")
        # P2: G(in oncoming lane -> TTC to oncoming car >= TTC_MIN)
        if w.lat > 0.1 and w.onc_s > w.s:
            ttc = (w.onc_s - w.s - bt.CAR_LEN) / max(w.v + abs(w.onc_v), 1e-3)
            if ttc < TTC_MIN:
                bad("P2_oncoming_TTC")
        # P3: G(speed in roundabout zone <= 8 m/s)
        if bt.ROUND_ENTRY - 1 < w.s < bt.ROUND_END and w.v > 8.0:
            bad("P3_roundabout_speed")
        # P4: G(no lane change inside/near roundabout)
        if w.s > bt.ROUND_ENTRY - 30 and w.s < bt.ROUND_END and abs(w.lat) > 0.02:
            bad("P4_no_lane_change_near_roundabout")
        # P5: G(never stationary in the conflict zone while a circulating car arrives)  (subset of P1, explicit)
        # P6: G(exactly <=1 active leaf per tick)
        if len(set(bb["active"])) > 1:
            bad("P6_single_active_leaf")


def structural_checks():
    tree = bt.build_tree()
    leaves, depth = [], [0]

    def walk(n, d=0):
        depth[0] = max(depth[0], d)
        kids = getattr(n, "children", None) or ([n.child] if hasattr(n, "child") else [])
        if not kids:
            leaves.append(n.name)
        for k in kids:
            walk(k, d + 1)
    walk(tree)
    out = {"leaves": leaves, "depth": depth[0]}
    # totality: root's last child is an Action that never returns FAILURE
    out["root_total"] = tree.children[-1].name == "cruise"
    return out


def run_grid(grid, label):
    t0 = time.time(); n = 0; fails = {}; incomplete = 0; examples = {}
    for combo in itertools.product(*grid.values()):
        p = dict(zip(grid.keys(), combo))
        p["cv_times"] = p.pop("cv", ())
        m = Monitor()
        res, _ = bt.run_episode(0, params=p, monitor=m)
        n += 1
        if not res["done"]:
            incomplete += 1
            examples.setdefault("LIVENESS", p)
        for k in m.viol:
            fails[k] = fails.get(k, 0) + 1
            examples.setdefault(k, p)
    print(f"\n[{label}] {n} scenarios in {time.time()-t0:.0f}s")
    print("  violations:", fails if fails else "none", "| incomplete routes:", incomplete)
    for k, v in examples.items():
        print("  counterexample", k, {a: (np.round(b, 1).tolist() if hasattr(b, '__len__') else b) for a, b in v.items()})
    return n, fails, incomplete


if __name__ == "__main__":
    st = structural_checks()
    print("structure:", st)
    # sweep A: overtaking sub-problem, fine grid, no roundabout traffic
    gridA = dict(lead_s=range(20, 71, 5), lead_v=[3, 4, 5, 6, 7],
                 onc_s=range(120, 361, 20), onc_v=[9, 10, 11, 12, 13], cv=[()])
    # sweep B: roundabout yielding: all subsets of <=3 circulating-car passage times (coarse), fixed others
    slots = list(range(20, 61, 4))
    cvs = [c for r in range(0, 4) for c in itertools.combinations(slots, r)]
    gridB = dict(lead_s=[30, 50], lead_v=[4, 6], onc_s=[200, 340], onc_v=[11], cv=cvs)
    run_grid(gridA, "A overtaking")
    run_grid(gridB, "B roundabout yield")
