"""
Closed-loop evaluation: lattice planner -> QP trajectory optimisation -> tracking controller -> vehicle.
Run: python closed_loop.py [--plot]   (writes results/closed_loop.json and figures/closed_loop.png)
"""
import sys, json, time, os
import numpy as np
from planning import LatticePlanner, FState, Obstacle, optimise_trajectory, check_limits, collides, CAR_L, CAR_W
from control import (RefPath, Vehicle, PurePursuit, Stanley, AdaptiveMPC, SpeedController, wrap, WHEELBASE)

DT, REPLAN = 0.05, 0.5


def make_scenario():
    return [Obstacle(45.0, 0.0, 5.0), Obstacle(180.0, 0.0, 0.0)]     # slow lead + parked car in the ego lane


def traj_to_cart(tr, ref, dt=0.05):
    """Resample a Frenet trajectory on a dt grid and convert to Cartesian reference signals."""
    t = np.arange(0, tr.t[-1] + 1e-9, dt)
    s = np.interp(t, tr.t, tr.s); l = np.interp(t, tr.t, tr.l)
    sd = np.interp(t, tr.t, tr.sd); sdd = np.interp(t, tr.t, tr.sdd)
    x, y = ref.to_cart(s, l)
    dx, dy = np.gradient(x, dt), np.gradient(y, dt)
    th = np.unwrap(np.arctan2(dy, dx)); v = np.hypot(dx, dy)
    kappa = np.gradient(th, dt) / np.maximum(v, 0.5)
    return dict(t=t, s=s, l=l, x=x, y=y, th=th, v=v, kappa=kappa, sd=sd, sdd=sdd)


def run(controller, k_true=1.0, latency=0.0, pos_noise=0.0, yaw_noise=0.0, seed=0, v0=8.0, t_end=60.0,
        use_opt=True, record=False):
    rng = np.random.default_rng(seed)
    ref = RefPath(); planner = LatticePlanner(v_des=12.0)
    obs = make_scenario()
    x0, y0, th0, _ = ref.at(0.0)
    veh = Vehicle(x0, y0, th0, v0, k_true=k_true, latency=latency, dt=DT)
    sp = SpeedController(); ctrl = controller
    if hasattr(ctrl, "k_hat"): ctrl.k_hat, ctrl.u_prev, ctrl.delta_filt = 1.0, 0.0, 0.0
    t, s_hint = 0.0, 0.0
    traj = None; t_plan = -1e9; cart = None
    st = FState(0.0, v0, 0.0, 0.0, 0.0, 0.0)
    ey_log, es_log, clr_log, lat_acc, plan_ms, cand_n, log = [], [], [], [], [], [], []
    collisions, infeasible, replans = 0, 0, 0
    yaw_rate = 0.0; delta_cmd = 0.0
    while t < t_end:
        s_true, l_true, ihint = ref.project(veh.x, veh.y, s_hint); s_hint = s_true
        if s_true > 330.0:
            break
        # ---- measurement with injected localisation error
        mx = veh.x + rng.normal(0, pos_noise); my = veh.y + rng.normal(0, pos_noise)
        mth = veh.th + rng.normal(0, yaw_noise)
        # ---- replan
        if traj is None or t - t_plan >= REPLAN - 1e-9:
            tau = t - t_plan
            if traj is not None:
                k = min(int(round(tau / planner.dt)), len(traj.t) - 1)
                st = FState(traj.s[k], traj.sd[k], traj.sdd[k], traj.l[k], traj.ld[k], traj.ldd[k])
                s_m, l_m, _ = ref.project(mx, my, s_hint)
                if abs(l_m - st.l) > 1.0 or abs(s_m - st.s) > 4.0:        # tracking lost: restart from measurement
                    st = FState(s_m, veh.v, 0.0, l_m, 0.0, 0.0)
            else:
                st = FState(s_true, veh.v, 0.0, l_true, 0.0, 0.0)
            t0 = time.perf_counter()
            tr, n = planner.plan(st, [Obstacle(o.s, o.l, o.v) for o in obs])
            if use_opt:
                op = optimise_trajectory(tr, obs)
                if not collides(op.t, op.s, op.l, op.sd, obs) and check_limits(op)["ok"]:
                    tr = op
                else:
                    infeasible += 1
            plan_ms.append((time.perf_counter() - t0) * 1e3); cand_n.append(n)
            traj, t_plan = tr, t
            # shift so that traj.t starts at plan time; obstacles in the plan already start at plan time
            cart = traj_to_cart(traj, ref); replans += 1
            plan_obs_t = t
        # ---- reference at current time
        tau = min(t - t_plan, cart["t"][-1] - 2.0)
        i = int(round(tau / DT))
        r = {k: cart[k][i] for k in ("x", "y", "th", "kappa", "v", "s", "sd", "sdd")}
        idx = np.minimum(i + np.round(0.1 / DT * np.arange(1, 16)).astype(int), len(cart["t"]) - 1)
        r["vN"], r["kN"] = cart["v"][idx], cart["kappa"][idx]
        j = np.minimum(i + np.arange(0, 120, 6), len(cart["t"]) - 1)
        r["xs"], r["ys"] = cart["x"][j], cart["y"][j]
        # ---- control
        delta_cmd = ctrl.command((mx, my, mth), veh.v, r)
        s_m, _, _ = ref.project(mx, my, s_hint)
        a_cmd = sp.command(veh.v, r["sd"] * 1.0 if False else r["v"], r["sdd"], r["s"] - s_m, DT)
        yaw_rate = veh.step(delta_cmd, a_cmd)
        if hasattr(ctrl, "estimate"):
            ctrl.estimate(veh.v, yaw_rate + rng.normal(0, 0.005), delta_cmd, DT)
        # ---- world update + metrics (ground truth)
        for o in obs:
            o.s += o.v * DT
        t += DT
        s_tr, l_tr, _ = ref.project(veh.x, veh.y, s_hint)
        ey, _ = (lambda e: e)((-np.sin(r["th"]) * (veh.x - r["x"]) + np.cos(r["th"]) * (veh.y - r["y"]), 0))
        ey_log.append(ey); es_log.append(s_tr - r["s"]); lat_acc.append(abs(veh.v * yaw_rate))
        c = 1e9
        for o in obs:
            gs, gl = abs(s_tr - o.s) - CAR_L, abs(l_tr - o.l) - CAR_W
            c = min(c, max(gs, gl))
            if gs < 0 and gl < 0:
                collisions += 1
        clr_log.append(c)
        if record:
            log.append((t, veh.x, veh.y, veh.v, s_tr, l_tr, delta_cmd, ey, getattr(ctrl, "k_hat", 1.0)))
    ey = np.array(ey_log)
    out = dict(controller=ctrl.name, completed=bool(s_hint > 325.0), time=round(t, 1),
               rms_ey=float(np.sqrt(np.mean(ey ** 2))), max_ey=float(np.max(np.abs(ey))),
               collision_steps=int(collisions), min_clearance=float(np.min(clr_log)),
               max_lat_acc=float(np.max(lat_acc)), plan_ms_mean=float(np.mean(plan_ms)),
               plan_ms_p95=float(np.percentile(plan_ms, 95)), replans=replans, opt_rejected=infeasible,
               k_hat_final=float(getattr(ctrl, "k_hat", 1.0)))
    return (out, np.array(log), obs) if record else out


CONDITIONS = {
    "nominal": dict(),
    "gain_0.7+latency_0.1s": dict(k_true=0.7, latency=0.10),
    "loc_noise_0.3m": dict(pos_noise=0.3, yaw_noise=0.01),
    "all_faults": dict(k_true=0.7, latency=0.10, pos_noise=0.3, yaw_noise=0.01),
}


def controllers():
    return [PurePursuit(), Stanley(), AdaptiveMPC(adapt=False), AdaptiveMPC(adapt=True)]


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True); os.makedirs("figures", exist_ok=True)
    results = {}
    print(f"{'condition':24s} {'controller':20s} {'done':>5s} {'RMS ey':>7s} {'max ey':>7s} {'coll':>5s} {'clr[m]':>7s} {'plan ms':>8s}")
    for cname, kw in CONDITIONS.items():
        results[cname] = []
        for c in controllers():
            r = run(c, **kw); results[cname].append(r)
            print(f"{cname:24s} {r['controller']:20s} {str(r['completed']):>5s} {r['rms_ey']:7.3f} {r['max_ey']:7.2f} "
                  f"{r['collision_steps']:5d} {r['min_clearance']:7.2f} {r['plan_ms_mean']:8.1f}")
    json.dump(results, open("results/closed_loop.json", "w"), indent=1)
    if "--plot" in sys.argv:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ref = RefPath()
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        for c in controllers():
            r, log, obs = run(c, record=True, **CONDITIONS["all_faults"])
            ax[0].plot(log[:, 4], log[:, 5], label=c.name)
            ax[1].plot(log[:, 0], log[:, 7], label=c.name)
            if c.name == "Adaptive MPC": ax[2].plot(log[:, 0], log[:, 8], "g")
        for o0 in make_scenario():
            ax[0].plot([o0.s, o0.s + 4.5], [o0.l, o0.l], "k", lw=6, alpha=.4)
        ax[0].set_xlabel("s [m]"); ax[0].set_ylabel("l [m]"); ax[0].set_title("Path in Frenet frame (all faults)"); ax[0].legend(fontsize=7)
        ax[1].set_xlabel("t [s]"); ax[1].set_ylabel("tracking error e_y [m]"); ax[1].set_title("Lateral tracking error"); ax[1].legend(fontsize=7)
        ax[2].axhline(0.7, ls="--", c="r"); ax[2].set_xlabel("t [s]"); ax[2].set_ylabel("estimated steering gain"); ax[2].set_title("Adaptive MPC gain estimate (true 0.7)")
        plt.tight_layout(); plt.savefig("figures/closed_loop.png", dpi=130); print("saved figures/closed_loop.png")
