"""
Behavior-tree decision making for overtaking, roundabout entry and yielding.

Part 1: a small BT library (Selector / Sequence with optional memory / Condition / Action / Guard).
Part 2: a 1-D route world (lane 0 = ego lane, lane 1 = oncoming lane) and the driving tree.
Part 3: randomised batch evaluation (python behavior_tree.py).
"""
import os, sys
from enum import Enum
import numpy as np


# ------------------------------------------------------------------ BT library
class Status(Enum):
    SUCCESS, FAILURE, RUNNING = 1, 2, 3


class Node:
    def __init__(self, name):
        self.name = name

    def tick(self, bb):
        raise NotImplementedError

    def reset(self):
        pass


class Condition(Node):
    def __init__(self, name, fn):
        super().__init__(name); self.fn = fn

    def tick(self, bb):
        return Status.SUCCESS if self.fn(bb) else Status.FAILURE


class Action(Node):
    """fn(bb) -> Status. Writes its commands (target speed / lane) into the blackboard."""
    def __init__(self, name, fn):
        super().__init__(name); self.fn = fn

    def tick(self, bb):
        st = self.fn(bb)
        if st == Status.RUNNING:
            bb["active"].append(self.name)
        return st


class Sequence(Node):
    """memory=True: resumes at the running child (used for committed manoeuvres)."""
    def __init__(self, name, children, memory=False):
        super().__init__(name); self.children, self.memory, self.i = children, memory, 0

    def tick(self, bb):
        start = self.i if self.memory else 0
        for k in range(start, len(self.children)):
            st = self.children[k].tick(bb)
            if st == Status.RUNNING:
                self.i = k
                return st
            if st == Status.FAILURE:
                self.reset(); return st
        self.reset()
        return Status.SUCCESS

    def reset(self):
        self.i = 0
        for c in self.children:
            c.reset()


class Selector(Node):
    """Reactive priority selector; lower-priority branches are reset when a higher one takes over."""
    def __init__(self, name, children):
        super().__init__(name); self.children = children

    def tick(self, bb):
        for k, c in enumerate(self.children):
            st = c.tick(bb)
            if st != Status.FAILURE:
                for other in self.children[k + 1:]:
                    other.reset()
                return st
        return Status.FAILURE

    def reset(self):
        for c in self.children:
            c.reset()


class Guard(Node):
    """Runs child only while cond holds; otherwise FAILURE (abort condition for a manoeuvre)."""
    def __init__(self, name, cond, child):
        super().__init__(name); self.cond, self.child = cond, child

    def tick(self, bb):
        if not self.cond(bb):
            self.child.reset(); return Status.FAILURE
        return self.child.tick(bb)

    def reset(self):
        self.child.reset()


# ------------------------------------------------------------------ world
V_DES, A_MAX, B_COMF = 12.0, 2.5, 3.0
CAR_LEN = 4.5
ROUND_ENTRY, ROUND_END = 300.0, 345.0
YIELD_LINE = ROUND_ENTRY - 3.0   # stop line (ego front)
ROUTE_END = 420.0


class World:
    def __init__(self, seed, params=None):
        r = np.random.default_rng(seed)
        self.t = 0.0
        self.s, self.v, self.lat = 0.0, 0.0, 0.0          # lat: 0 = lane 0, 1 = lane 1
        self.lead_s = r.uniform(30, 60)
        self.lead_v = r.uniform(3.0, 7.0)
        self.lead_v0 = self.lead_v                        # lead's desired speed (it follows ego with IDM if ego is ahead)
        self.onc_s = r.uniform(180, 320)                  # oncoming car in lane 1
        self.onc_v = -r.uniform(9.0, 13.0)
        n_cv = r.integers(2, 5)                           # circulating vehicles: times they pass conflict point
        self.cv_times = np.sort(r.uniform(18, 60, n_cv))
        if params is not None:                            # explicit scenario (used by formal/exhaustive checks)
            self.lead_s, self.lead_v = params["lead_s"], params["lead_v"]
            self.lead_v0 = self.lead_v
            self.onc_s, self.onc_v = params["onc_s"], -abs(params["onc_v"])
            self.cv_times = np.array(sorted(params["cv_times"]), float)
        self.collisions = 0
        self.min_gap = 1e9
        self.yield_wait = 0.0
        self.overtakes = 0

    # tick the world with commands from the BT
    def step(self, cmd, dt=0.1):
        tv, tl = cmd["speed"], cmd["lane"]
        a = np.clip((tv - self.v) / 0.6, -B_COMF * 1.8, A_MAX)
        self.v = max(0.0, self.v + a * dt)
        self.s += self.v * dt
        self.lat += np.clip(tl - self.lat, -dt / 2.0, dt / 2.0)   # 2 s per lane change
        # lead car: IDM follower when ego is ahead of it in lane 0 (so it never rear-ends a stopped ego)
        gap = self.s - self.lead_s - CAR_LEN
        if self.lat < 0.9 and gap > -CAR_LEN:
            dv = self.lead_v - self.v
            s_star = 2.0 + self.lead_v * 1.2 + self.lead_v * dv / (2 * np.sqrt(2.0 * 3.0))
            acc = 2.0 * (1 - (self.lead_v / self.lead_v0) ** 4 - (s_star / max(gap, 0.1)) ** 2)
            self.lead_v = max(0.0, self.lead_v + np.clip(acc, -6.0, 2.0) * dt)
        self.lead_s += self.lead_v * dt
        self.onc_s += self.onc_v * dt
        self.t += dt
        # safety metrics
        in0, in1 = self.lat < 0.9, self.lat > 0.1
        if in0 and abs(self.lead_s - self.s) < CAR_LEN:
            self.collisions += 1
        if in1 and abs(self.onc_s - self.s) < CAR_LEN:
            self.collisions += 1
        if in0 and self.lead_s > self.s:
            self.min_gap = min(self.min_gap, self.lead_s - self.s - CAR_LEN)
        if in1 and self.onc_s > self.s:
            self.min_gap = min(self.min_gap, self.onc_s - self.s - CAR_LEN)
        # collision with circulating traffic: ego inside conflict zone while a cv passes
        if ROUND_ENTRY - 1.0 < self.s < ROUND_ENTRY + CAR_LEN:   # physical conflict zone
            for tc in self.cv_times:
                if abs(self.t - tc) < 1.0:
                    self.collisions += 1
        if self.v < 0.3 and YIELD_LINE - 8 < self.s < ROUND_ENTRY:
            self.yield_wait += dt


def bb_from_world(w):
    return dict(w=w, active=[])


# ------------------------------------------------------------------ conditions / actions
def lead_gap(bb):
    w = bb["w"]
    return w.lead_s - w.s - CAR_LEN if w.lat < 0.9 and w.lead_s > w.s else 1e9


def blocked_by_slow_lead(bb):
    """Slow lead close ahead AND the whole pass (incl. return to lane 0) finishes before the roundabout
    approach zone starts -- the roundabout branch has priority and must never interrupt a pass."""
    w = bb["w"]
    if bb.get("degraded"):                       # perception supervisor flagged a fault: no overtaking
        return False
    if not (lead_gap(bb) < 28 and w.lead_v < 0.75 * V_DES):
        return False
    _, d_ego = predict_pass(w)
    return w.s + d_ego + 15.0 < YIELD_LINE - 60.0


def raw_lead_gap(w):
    return w.lead_s - w.s - CAR_LEN


def predict_pass(w):
    """Kinematic prediction of the overtake: returns (duration, ego distance travelled).
    Ego accelerates at A_MAX towards V_DES, lead keeps its speed; done when the ego is
    2 car lengths + 12 m ahead, plus 2 s for the lane change back."""
    t, se, ve, sl = 0.0, w.s, w.v, w.lead_s
    while se < sl + 2 * CAR_LEN + 12 and t < 40:
        ve = min(V_DES, ve + A_MAX * 0.1)
        se += ve * 0.1; sl += w.lead_v * 0.1; t += 0.1
    t += 2.0
    return t, se - w.s + 2.0 * V_DES


def overtake_safe(bb, k=1.3, margin=40.0):
    """Oncoming car must stay clear for the whole predicted manoeuvre (+30% time margin, +40 m) -- start
    condition. The abort test uses k=1.0, margin=15 m: hysteresis, so the tree cannot chatter between lanes."""
    w = bb["w"]
    gap = w.onc_s - w.s
    if gap < -(2 * CAR_LEN + 5.0):                       # oncoming car fully past us (bumper clearance + margin)
        return True
    t, d_ego = predict_pass(w)
    return gap > 0 and gap - d_ego - abs(w.onc_v) * k * t > margin


def merge_back_safe(w):
    """Can we return to lane 0 behind the lead without hitting it? (2.5 s lane change + 8 m reserve)"""
    return raw_lead_gap(w) > max(w.v - w.lead_v, 0.0) * 2.5 + 8.0


def oncoming_unsafe_abort(bb):
    """Abort only while still behind the lead AND merging back is itself safe; otherwise we are committed."""
    w = bb["w"]
    behind = w.s < w.lead_s - CAR_LEN - 1.0
    return behind and not overtake_safe(bb, k=1.0, margin=15.0) and merge_back_safe(w)


def go_left(bb):
    bb["cmd"] = dict(speed=min(V_DES, bb["w"].lead_v + 5.0 + max(0, lead_gap(bb)) * 0.0 + 4.0), lane=1)
    return Status.SUCCESS if bb["w"].lat > 0.95 else Status.RUNNING


def pass_lead(bb):
    w = bb["w"]
    bb["cmd"] = dict(speed=V_DES, lane=1)
    return Status.SUCCESS if w.s > w.lead_s + CAR_LEN + 8 else Status.RUNNING


def return_right(bb):
    w = bb["w"]
    bb["cmd"] = dict(speed=V_DES, lane=0)
    if w.lat < 0.05:
        w.overtakes += 1
        return Status.SUCCESS
    return Status.RUNNING


def approaching_roundabout(bb):
    w = bb["w"]
    in_zone = 0 < YIELD_LINE - w.s < 60 or YIELD_LINE < w.s < ROUND_END
    return in_zone and w.lat < 0.02          # never preempt a lane change / overtake in progress


def before_yield_line(bb):
    return bb["w"].s < ROUND_ENTRY - 0.5          # commit point: after this we are in the zone


def arrival_time(w):
    """Time to reach the conflict point from current (s, v) assuming accel A_ENTER (kinematic)."""
    d = max(ROUND_ENTRY - w.s, 0.0)
    a = 1.5
    return w.t + (-w.v + np.sqrt(w.v ** 2 + 2 * a * d)) / a


def conflict_ahead(bb):
    """Predicted arrival at the conflict point overlaps (+-3 s) a circulating vehicle."""
    w = bb["w"]
    t_arr = arrival_time(w)
    d = ROUND_ENTRY - w.s
    d_nr = w.v * 0.6 + w.v ** 2 / (2 * 4.5)                   # distance needed to stop (lag + braking)
    win = 4.5 if d < d_nr + 12.0 else 3.0                     # wider window near the point of no return
    return bool(np.any((w.cv_times > t_arr - win) & (w.cv_times < t_arr + win)))


def yield_action(bb):
    """Brake to a stop at the yield line (v = sqrt(2 b d)) and hold until the gap opens."""
    w = bb["w"]
    d = max(YIELD_LINE - w.s - 2.0, 0.0)          # aim 2 m short: actuator lag (0.6 s) makes the car creep forward
    v = min(V_DES * 0.6, np.sqrt(2 * 2.0 * d))
    bb["cmd"] = dict(speed=follow_limit(w, v if d > 0.3 else 0.0), lane=0)
    return Status.RUNNING


def follow_limit(w, v):
    """Car-following speed cap applied to every longitudinal command (not only in cruise)."""
    g = lead_gap_w(w)
    if g < 60:
        v = min(v, max(w.lead_v + 0.4 * (g - (6 + 1.2 * w.v)), 0.0))
    return v


def lead_gap_w(w):
    return w.lead_s - w.s - CAR_LEN if w.lat < 0.9 and w.lead_s > w.s else 1e9


def enter_roundabout(bb):
    w = bb["w"]
    bb["cmd"] = dict(speed=follow_limit(w, 6.0 if w.s < ROUND_END else V_DES), lane=0)
    return Status.RUNNING


def cruise(bb):
    """Adaptive cruise: free-flow V_DES, or follow the lead with a time-gap controller."""
    w = bb["w"]
    g = lead_gap(bb)
    v = V_DES
    if g < 60:
        v = min(V_DES, w.lead_v + 0.4 * (g - (6 + 1.2 * w.v)))
    bb["cmd"] = dict(speed=max(v, 0.0), lane=0)
    return Status.RUNNING


def build_tree():
    overtake = Guard("overtake_guard", lambda bb: not oncoming_unsafe_abort(bb),
                     Sequence("overtake", [
                         Condition("blocked_by_slow_lead", blocked_by_slow_lead),
                         Condition("overtake_safe", overtake_safe),
                         Action("go_left", go_left),
                         Action("pass_lead", pass_lead),
                         Action("return_right", return_right)], memory=True))
    roundabout = Sequence("roundabout", [
        Condition("approaching_roundabout", approaching_roundabout),
        Selector("yield_or_enter", [
            Sequence("yield", [Condition("before_yield_line", before_yield_line),
                               Condition("conflict_ahead", conflict_ahead),
                               Action("yield", yield_action)]),
            Action("enter_roundabout", enter_roundabout)])])
    return Selector("root", [roundabout, overtake, Action("cruise", cruise)])


# ------------------------------------------------------------------ simulation
def run_episode(seed, trace=False, params=None, monitor=None, tree=None):
    w, tree = World(seed, params), (tree or build_tree())
    log = []
    while w.s < ROUTE_END and w.t < 150:
        bb = bb_from_world(w)
        bb["cmd"] = dict(speed=0.0, lane=0)
        tree.tick(bb)
        w.step(bb["cmd"])
        if monitor:
            monitor(w, bb)
        if trace:
            log.append((w.t, w.s, w.v, w.lat, bb["active"][-1] if bb["active"] else "-"))
    done = w.s >= ROUTE_END
    return dict(done=done, time=w.t, collisions=w.collisions, min_gap=w.min_gap,
                overtakes=w.overtakes, yield_wait=w.yield_wait), log


if __name__ == "__main__":
    n = 300
    res = [run_episode(s)[0] for s in range(n)]
    ok = np.array([r["done"] for r in res]); col = np.array([r["collisions"] > 0 for r in res])
    print(f"{n} randomised episodes (random lead speed/position, oncoming car, circulating traffic)")
    print(f"  completed route        : {ok.mean()*100:.1f}%")
    print(f"  episodes with collision: {col.mean()*100:.1f}%")
    print(f"  overtakes performed    : {np.mean([r['overtakes'] for r in res]):.2f} per episode "
          f"({np.mean([r['overtakes'] > 0 for r in res])*100:.0f}% of episodes)")
    print(f"  mean route time        : {np.mean([r['time'] for r in res if r['done']]):.1f} s")
    print(f"  min gap (median / 5th) : {np.median([r['min_gap'] for r in res]):.1f} m / "
          f"{np.percentile([r['min_gap'] for r in res], 5):.1f} m")
    print(f"  episodes that had to wait at yield line: {np.mean([r['yield_wait'] > 0.5 for r in res])*100:.0f}%")
    if "--plot" in sys.argv:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _, log = run_episode(3, trace=True)
        L = np.array([(a, b, c, d) for a, b, c, d, _ in log]); names = [e for *_, e in log]
        fig, ax = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
        ax[0].plot(L[:, 0], L[:, 1]); ax[0].set_ylabel("s [m]")
        ax[1].plot(L[:, 0], L[:, 2]); ax[1].set_ylabel("speed [m/s]")
        uniq = list(dict.fromkeys(names))
        ax[2].step(L[:, 0], [uniq.index(n_) for n_ in names]); ax[2].set_yticks(range(len(uniq)))
        ax[2].set_yticklabels(uniq); ax[2].set_xlabel("time [s]"); ax[2].set_ylabel("active leaf")
        plt.tight_layout(); os.makedirs("figures", exist_ok=True); plt.savefig("figures/bt_episode.png", dpi=120); print("saved figures/bt_episode.png")
