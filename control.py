"""
Vehicle model, reference path and trajectory-tracking controllers.
  Pure pursuit, Stanley, and a time-varying linear MPC (condensed bounded least squares) with online
  adaptation of the steering-effectiveness gain (NLMS on gyro yaw rate).
"""
import numpy as np
from scipy.optimize import lsq_linear

WHEELBASE, DELTA_MAX, DELTA_RATE = 2.7, 0.5, 0.7


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class RefPath:
    """Smooth S-curve reference road, dense (ds = 0.1 m); Frenet <-> Cartesian conversion."""
    def __init__(self, length=700.0, ds=0.1):
        self.ds = ds
        self.s = np.arange(0, length, ds)
        self.th = 0.30 * np.sin(2 * np.pi * self.s / 260.0)
        self.x = np.cumsum(np.cos(self.th)) * ds
        self.y = np.cumsum(np.sin(self.th)) * ds
        self.kappa = np.gradient(self.th, ds)

    def at(self, s):
        s = np.clip(s, 0, self.s[-1] - 1e-3)
        return (np.interp(s, self.s, self.x), np.interp(s, self.s, self.y),
                np.interp(s, self.s, self.th), np.interp(s, self.s, self.kappa))

    def to_cart(self, s, l):
        x, y, th, _ = self.at(s)
        return x - l * np.sin(th), y + l * np.cos(th)

    def project(self, x, y, hint=None):
        i0, i1 = (0, len(self.s)) if hint is None else (max(int((hint - 20) / self.ds), 0), min(int((hint + 40) / self.ds), len(self.s)))
        d2 = (self.x[i0:i1] - x) ** 2 + (self.y[i0:i1] - y) ** 2
        i = i0 + int(np.argmin(d2))
        th = self.th[i]
        l = -np.sin(th) * (x - self.x[i]) + np.cos(th) * (y - self.y[i])
        s = self.s[i] + np.cos(th) * (x - self.x[i]) + np.sin(th) * (y - self.y[i])
        return s, l, i


class Vehicle:
    """Kinematic bicycle + first-order steering/acceleration actuators, unknown steering gain, latency."""
    def __init__(self, x, y, th, v, k_true=1.0, tau_steer=0.15, tau_acc=0.25, latency=0.0, dt=0.05):
        self.x, self.y, self.th, self.v = x, y, th, v
        self.delta, self.acc = 0.0, 0.0
        self.k_true, self.tau_s, self.tau_a, self.dt = k_true, tau_steer, tau_acc, dt
        self.buf = [(0.0, 0.0)] * int(round(latency / dt))

    def step(self, delta_cmd, acc_cmd):
        self.buf.append((delta_cmd, acc_cmd)); d_c, a_c = self.buf.pop(0)
        d_c = np.clip(self.k_true * d_c, -DELTA_MAX, DELTA_MAX)
        dd = np.clip((d_c - self.delta) / self.tau_s, -DELTA_RATE, DELTA_RATE)
        self.delta += dd * self.dt
        self.acc += (np.clip(a_c, -6, 3) - self.acc) / self.tau_a * self.dt
        self.v = max(0.0, self.v + self.acc * self.dt)
        self.x += self.v * np.cos(self.th) * self.dt
        self.y += self.v * np.sin(self.th) * self.dt
        self.th = wrap(self.th + self.v / WHEELBASE * np.tan(self.delta) * self.dt)
        return self.v / WHEELBASE * np.tan(self.delta)          # true yaw rate (gyro, before noise)


def tracking_errors(x, y, th, xr, yr, thr):
    ey = -np.sin(thr) * (x - xr) + np.cos(thr) * (y - yr)
    return ey, wrap(th - thr)


class PurePursuit:
    name = "Pure pursuit"
    def command(self, pose, v, ref, **_):
        x, y, th = pose
        Ld = max(4.0, 0.9 * v)
        px, py = ref["xs"], ref["ys"]
        d = np.hypot(px - x, py - y)
        j = int(np.argmax(d >= Ld)) if (d >= Ld).any() else len(d) - 1
        alpha = wrap(np.arctan2(py[j] - y, px[j] - x) - th)
        return float(np.clip(np.arctan2(2 * WHEELBASE * np.sin(alpha), Ld), -DELTA_MAX, DELTA_MAX))


class Stanley:
    name = "Stanley"
    def __init__(self, k=1.5):
        self.k = k
    def command(self, pose, v, ref, **_):
        x, y, th = pose
        ey, eth = tracking_errors(x, y, th, ref["x"], ref["y"], ref["th"])
        ff = np.arctan(WHEELBASE * ref["kappa"])
        return float(np.clip(ff - eth - np.arctan2(self.k * ey, v + 1.0), -DELTA_MAX, DELTA_MAX))


class AdaptiveMPC:
    """
    Error model (about the reference):  e_y' = v e_th,  e_th' = (k v / L) u,  u = delta - delta_ff.
    Time-varying in v (speed profile of the reference), k = steering gain estimated online (NLMS).
    Condensed QP:  min sum q_y e_y^2 + q_th e_th^2 + r u^2 + r_d du^2   s.t. |delta_ff + u| <= delta_max.
    """
    def __init__(self, N=15, dt=0.1, adapt=True, qy=1.0, qth=2.0, r=1.0, rd=20.0, mu=0.05):
        self.N, self.dt, self.adapt, self.mu = N, dt, adapt, mu
        self.q, self.r, self.rd = (qy, qth), r, rd
        self.k_hat, self.u_prev, self.delta_filt = 1.0, 0.0, 0.0
        self.name = "Adaptive MPC" if adapt else "MPC (fixed model)"

    def estimate(self, v, yaw_rate_meas, delta_cmd, dt):
        """NLMS update of the steering gain from gyro yaw rate (regressor = commanded steer through the known lag)."""
        self.delta_filt += (delta_cmd - self.delta_filt) / 0.15 * dt
        phi = v / WHEELBASE * np.tan(self.delta_filt)
        if self.adapt and abs(phi) > 0.01 and v > 2.0:
            self.k_hat += self.mu * phi * (yaw_rate_meas - self.k_hat * phi) / (1e-3 + phi ** 2)
            self.k_hat = float(np.clip(self.k_hat, 0.4, 1.6))

    def command(self, pose, v, ref, **_):
        N, dt = self.N, self.dt
        x, y, th = pose
        ey, eth = tracking_errors(x, y, th, ref["x"], ref["y"], ref["th"])
        vs, kap = ref["vN"], ref["kN"]                      # reference speed / curvature along horizon (N,)
        ff = np.arctan(WHEELBASE * kap) / self.k_hat       # feed-forward steer needed with estimated gain
        # condensed prediction  X = Phi x0 + Gam u   (X stacks x_1..x_N)
        Phi = np.zeros((2 * N, 2)); Gam = np.zeros((2 * N, N))
        P_prev, G_prev = np.eye(2), np.zeros((2, N))
        for k in range(N):
            A = np.array([[1.0, dt * vs[k]], [0.0, 1.0]])
            B = np.array([0.0, dt * self.k_hat * vs[k] / WHEELBASE])
            P_prev = A @ P_prev
            G_prev = A @ G_prev
            G_prev[:, k] = B
            Phi[2 * k:2 * k + 2] = P_prev
            Gam[2 * k:2 * k + 2] = G_prev
        x0 = np.array([ey, eth])
        sq = np.sqrt(np.tile(self.q, N))
        rows = [sq[:, None] * Gam, np.sqrt(self.r) * np.eye(N)]
        rhs = [-(sq * (Phi @ x0)), np.zeros(N)]
        D = np.eye(N) - np.eye(N, k=-1)
        rows.append(np.sqrt(self.rd) * D); e0 = np.zeros(N); e0[0] = np.sqrt(self.rd) * self.u_prev
        rhs.append(e0)
        lo, hi = -DELTA_MAX - ff, DELTA_MAX - ff
        sol = lsq_linear(np.vstack(rows), np.concatenate(rhs), bounds=(lo, hi), method="bvls")
        u = sol.x[0]
        self.u_prev = float(u)
        return float(np.clip(ff[0] + u, -DELTA_MAX, DELTA_MAX))


class SpeedController:
    """Feed-forward acceleration + PI on speed + along-track position error."""
    def __init__(self, kp=1.2, ki=0.15, ks=0.25):
        self.kp, self.ki, self.ks, self.I = kp, ki, ks, 0.0
    def command(self, v, v_ref, a_ref, s_err, dt):
        e = v_ref - v
        self.I = float(np.clip(self.I + e * dt, -3, 3))
        return float(np.clip(a_ref + self.kp * e + self.ki * self.I + self.ks * s_err, -6, 3))
