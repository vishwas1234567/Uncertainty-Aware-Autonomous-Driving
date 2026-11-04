"""
Motion planning in the Frenet frame.

1. Space-time lattice: a 3-D lattice of terminal conditions (lateral offset l_f x terminal speed v_f x
   horizon T, plus "follow-the-lead" terminals) is expanded with quintic (lateral) / quartic or quintic
   (longitudinal) jerk-optimal polynomials, checked against moving obstacles in (s, l, t) space-time,
   and ranked with a cost function (Werling et al. 2010 style).
2. Trajectory optimisation: the winning lattice trajectory is re-optimised as a bounded least-squares
   QP (minimum jerk + deviation from the lattice solution, with obstacle-induced corridor bounds on
   l(t) and s(t)). Dynamic limits are checked afterwards and reported.
"""
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import lsq_linear

CAR_L, CAR_W = 4.5, 2.0
A_MAX, A_MIN, A_LAT_MAX, V_MAX = 3.0, -6.0, 3.5, 20.0


@dataclass
class Obstacle:
    s: float; l: float; v: float = 0.0          # constant-speed along the road, fixed lateral position
    def at(self, t): return self.s + self.v * t, self.l


@dataclass
class FState:                                   # Frenet state (time derivatives)
    s: float; sd: float; sdd: float; l: float; ld: float; ldd: float


@dataclass
class Traj:
    t: np.ndarray; s: np.ndarray; l: np.ndarray
    sd: np.ndarray; ld: np.ndarray; sdd: np.ndarray; ldd: np.ndarray
    cost: float = 0.0; kind: str = ""
    def jerk_rms(self):
        j = np.gradient(np.hypot(self.sdd, self.ldd), self.t)
        return float(np.sqrt(np.mean(j ** 2)))


def quintic(x0, v0, a0, x1, v1, a1, T):
    A = np.array([[T ** 3, T ** 4, T ** 5], [3 * T ** 2, 4 * T ** 3, 5 * T ** 4], [6 * T, 12 * T ** 2, 20 * T ** 3]])
    b = np.array([x1 - (x0 + v0 * T + 0.5 * a0 * T ** 2), v1 - (v0 + a0 * T), a1 - a0])
    c3, c4, c5 = np.linalg.solve(A, b)
    return np.array([x0, v0, 0.5 * a0, c3, c4, c5])


def quartic(x0, v0, a0, v1, a1, T):
    A = np.array([[3 * T ** 2, 4 * T ** 3], [6 * T, 12 * T ** 2]])
    b = np.array([v1 - (v0 + a0 * T), a1 - a0])
    c3, c4 = np.linalg.solve(A, b)
    return np.array([x0, v0, 0.5 * a0, c3, c4, 0.0])


def poly_eval(c, t):
    x = c[0] + c[1] * t + c[2] * t ** 2 + c[3] * t ** 3 + c[4] * t ** 4 + c[5] * t ** 5
    v = c[1] + 2 * c[2] * t + 3 * c[3] * t ** 2 + 4 * c[4] * t ** 3 + 5 * c[5] * t ** 4
    a = 2 * c[2] + 6 * c[3] * t + 12 * c[4] * t ** 2 + 20 * c[5] * t ** 3
    return x, v, a


def collides(t, s, l, sd, obstacles, ms=3.0, ml=0.35):
    for o in obstacles:
        so = o.s + o.v * t
        ds = np.abs(s - so); dl = np.abs(l - o.l)
        # speed-dependent longitudinal buffer (0.5 s headway behind obstacle)
        buf = ms + 0.5 * np.maximum(sd, 0.0) * (s < so)
        if np.any((ds < CAR_L + buf) & (dl < CAR_W + ml)):
            return True
    return False


class LatticePlanner:
    def __init__(self, v_des=12.0, road=(-1.75, 5.25), dt=0.2, kappa_ref=None):
        self.v_des, self.road, self.dt = v_des, road, dt
        self.lf = np.unique(np.r_[np.arange(-0.75, 4.4, 0.875), 0.0, 3.5])     # lateral lattice incl. both lane centres
        self.Ts = (3.0, 4.0, 5.0)
        self.w = dict(jerk=0.1, T=0.4, lat=1.0, speed=1.0, lane=0.6, ldev=0.05)
        self.lane_centres = np.array([0.0, 3.5])

    def candidates(self, st: FState, obstacles):
        out = []
        for T in self.Ts:
            t = np.arange(0, T + 1e-9, self.dt)
            # longitudinal terminals: velocity keeping lattice, plus follow-lead for every obstacle ahead in lane
            lon = []
            for vf in np.linspace(0.0, self.v_des, 7):
                lon.append(("vel", quartic(st.s, st.sd, st.sdd, vf, 0.0, T), None))
            for o in obstacles:
                if o.s > st.s and o.v < self.v_des:
                    sf = o.s + o.v * T - (CAR_L + 6.0 + 1.2 * o.v)
                    if sf > st.s:
                        lon.append(("follow", quintic(st.s, st.sd, st.sdd, sf, o.v, 0.0, T), o))
            for lf in self.lf:
                cl = quintic(st.l, st.ld, st.ldd, lf, 0.0, 0.0, T)
                lv, ldv, lda = poly_eval(cl, t)
                for kind, cs, _ in lon:
                    s, sd, sdd = poly_eval(cs, t)
                    if np.any(sd < -0.1) or np.any(sdd > A_MAX) or np.any(sdd < A_MIN):
                        continue
                    if np.any(np.abs(lda) > 2.5) or np.any(np.abs(sd ** 2 * 0 + lda) > A_LAT_MAX):
                        continue
                    if np.any(lv < self.road[0] + 0.5) or np.any(lv > self.road[1] - 0.5):
                        continue
                    if collides(t, s, lv, sd, obstacles):
                        continue
                    j_l = np.gradient(lda, t); j_s = np.gradient(sdd, t)
                    J = (self.w["jerk"] * (np.trapezoid(j_l ** 2, t) + np.trapezoid(j_s ** 2, t))
                         + self.w["T"] * T
                         + self.w["speed"] * (sd[-1] - self.v_des) ** 2 * 0.25
                         + self.w["lane"] * np.min(np.abs(lf - self.lane_centres)) ** 2
                         + self.w["ldev"] * lf ** 2 * 0.2)
                    out.append(Traj(t, s, lv, sd, ldv, sdd, lda, J, f"{kind}|lf={lf:.2f}|T={T:.0f}"))
        return out

    def plan(self, st: FState, obstacles):
        c = self.candidates(st, obstacles)
        if not c:
            return self.emergency(st), 0
        best = min(c, key=lambda x: x.cost)
        return best, len(c)

    def emergency(self, st: FState):
        T = max(st.sd / 5.0, 1.0)
        t = np.arange(0, T + 1e-9, self.dt)
        cs = quartic(st.s, st.sd, st.sdd, 0.0, 0.0, T)
        s, sd, sdd = poly_eval(cs, t)
        l = np.full_like(t, st.l)
        return Traj(t, s, l, sd, np.zeros_like(t), sdd, np.zeros_like(t), 1e6, "emergency")


# ---------------------------------------------------------------- trajectory optimisation
def optimise_trajectory(tr: Traj, obstacles, w_jerk=1.0, w_dev=0.02, road=(-1.75, 5.25)):
    """
    Bounded least-squares QP on the sample vectors l(t) and s(t):
        min  w_jerk*||D3 x||^2 + w_dev*||x - x_lattice||^2     s.t.  x_lo <= x <= x_hi,  x(0..2) fixed
    Bounds: lateral corridor from obstacle sides chosen by the lattice solution; longitudinal upper
    bound s(t) <= s_obstacle_rear(t) for obstacles that the lattice solution stays behind.
    """
    t = tr.t; n = len(t); dt = t[1] - t[0]
    D3 = np.zeros((n - 3, n))
    for i in range(n - 3):
        D3[i, i:i + 4] = [-1, 3, -3, 1]
    D3 /= dt ** 3
    s_hi = np.full(n, np.inf); l_lo = np.full(n, road[0] + 0.5); l_hi = np.full(n, road[1] - 0.5)
    for o in obstacles:
        so = o.s + o.v * t
        close = np.abs(tr.s - so) < CAR_L + 12
        if not close.any():
            continue
        behind = tr.s < so
        if behind.all() and (np.abs(tr.l - o.l) < CAR_W + 0.5).any():
            s_hi = np.minimum(s_hi, so - CAR_L - 3.0)                 # stays behind: bound s
        elif close.any():                                              # passes beside: bound l by side
            side = np.sign(np.mean(tr.l[close]) - o.l)
            if side > 0:  l_lo[close] = np.maximum(l_lo[close], o.l + CAR_W + 0.35)
            else:         l_hi[close] = np.minimum(l_hi[close], o.l - CAR_W - 0.35)

    def solve(x0vec, lo, hi, fixed=3):
        lo = lo.copy(); hi = hi.copy()
        lo[:fixed] = x0vec[:fixed] - 1e-6; hi[:fixed] = x0vec[:fixed] + 1e-6
        lo = np.minimum(lo, x0vec - 1e-6 + 0) if False else lo
        hi = np.maximum(hi, lo + 1e-6)
        A = np.vstack([np.sqrt(w_jerk) * D3, np.sqrt(w_dev) * np.eye(n)])
        b = np.concatenate([np.zeros(n - 3), np.sqrt(w_dev) * x0vec])
        x0c = np.clip(x0vec, lo, hi)
        return lsq_linear(A, b, bounds=(lo, hi), method="bvls", lsmr_tol="auto").x

    l_new = solve(tr.l, l_lo, l_hi)
    s_new = solve(tr.s, np.full(n, -np.inf), s_hi)
    s_new = np.maximum.accumulate(s_new)
    sd = np.gradient(s_new, dt); sdd = np.gradient(sd, dt)
    ld = np.gradient(l_new, dt); ldd = np.gradient(ld, dt)
    return Traj(t, s_new, l_new, sd, ld, sdd, ldd, tr.cost, tr.kind + "|opt")


def check_limits(tr: Traj):
    """Dynamic feasibility report for a trajectory."""
    return dict(max_acc=float(tr.sdd.max()), max_dec=float(tr.sdd.min()), max_lat_acc=float(np.abs(tr.ldd).max()),
                v_max=float(tr.sd.max()), jerk_rms=tr.jerk_rms(),
                ok=bool(tr.sdd.max() <= A_MAX + 0.3 and tr.sdd.min() >= A_MIN - 0.3 and np.abs(tr.ldd).max() <= 3.0 and tr.sd.min() >= -0.2))
