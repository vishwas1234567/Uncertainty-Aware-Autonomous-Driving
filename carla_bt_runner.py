"""
Runs the behaviour tree (behavior_tree.py, unchanged) inside CARLA 0.9.x.

STATUS: written against the CARLA Python API but NOT yet executed in this project (no CARLA server was
available during development). Treat as an integration skeleton to be validated; see the report, Sec. 'Planned'.

The BT only reads attributes of a world object (s, v, lat, t, lead_s, lead_v, onc_s, onc_v, cv_times).
CarlaAdapter fills those from CARLA actors; the BT output (target speed, target lane) is converted to
carla.VehicleControl with a PI speed loop and pure-pursuit steering from control.py.

Usage:  python carla_bt_runner.py --host localhost --port 2000 --town Town03 --seed 0
"""
import argparse, math, time
import numpy as np
import behavior_tree as bt
from control import wrap, WHEELBASE

try:
    import carla
except ImportError:                       # allows importing this file (and unit-testing the adapter) without CARLA
    carla = None


class CarlaAdapter:
    """Maps CARLA actor states to the scalar world model used by the behaviour tree."""
    def __init__(self, ego, lead, oncoming, route, cv_times=()):
        self.ego, self.lead, self.onc, self.route = ego, lead, oncoming, route   # route: (N,2) polyline, left-hand lane = +l
        self.cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1))]
        self.cv_times = np.array(cv_times, float)
        self.t0 = time.time(); self.lat = 0.0
        self.t = self.s = self.v = self.lead_s = self.lead_v = self.onc_s = self.onc_v = 0.0
        self.collisions = 0; self.overtakes = 0

    def station(self, loc):
        d = np.hypot(self.route[:, 0] - loc.x, self.route[:, 1] - loc.y); i = int(np.argmin(d))
        return float(self.cum[i]), i

    def update(self):
        self.t = time.time() - self.t0
        for name, actor in (("ego", self.ego), ("lead", self.lead), ("onc", self.onc)):
            s, i = self.station(actor.get_location())
            vel = actor.get_velocity(); fwd = actor.get_transform().get_forward_vector()
            v = vel.x * fwd.x + vel.y * fwd.y
            if name == "ego":
                self.s, self.v = s, v
                tang = self.route[min(i + 1, len(self.route) - 1)] - self.route[i]
                nrm = np.array([-tang[1], tang[0]]) / (np.linalg.norm(tang) + 1e-9)
                loc = actor.get_location()
                self.lat = float(np.clip(np.dot([loc.x - self.route[i, 0], loc.y - self.route[i, 1]], nrm) / 3.5, 0, 1))
            elif name == "lead":
                self.lead_s, self.lead_v = s, v
            else:
                self.onc_s, self.onc_v = s, -abs(v)           # oncoming: station decreases with time

    def command_to_control(self, cmd, steer):
        ctrl = carla.VehicleControl()
        err = cmd["speed"] - self.v
        ctrl.throttle = float(np.clip(0.4 * err, 0, 1)); ctrl.brake = float(np.clip(-0.3 * err, 0, 1))
        ctrl.steer = float(np.clip(steer / 0.5, -1, 1))
        return ctrl


def pure_pursuit_steer(ego_tf, target_xy):
    yaw = math.radians(ego_tf.rotation.yaw); loc = ego_tf.location
    alpha = wrap(math.atan2(target_xy[1] - loc.y, target_xy[0] - loc.x) - yaw)
    Ld = max(4.0, math.hypot(target_xy[0] - loc.x, target_xy[1] - loc.y))
    return math.atan2(2 * WHEELBASE * math.sin(alpha), Ld)


def main():
    if carla is None:
        raise SystemExit("carla package not installed (pip install carla==0.9.15 matching your server)")
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost"); ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--town", default="Town04"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--duration", type=float, default=90.0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    client = carla.Client(a.host, a.port); client.set_timeout(10.0)
    world = client.load_world(a.town); m = world.get_map()
    settings = world.get_settings(); settings.synchronous_mode = True; settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    bp = world.get_blueprint_library()
    start = m.get_spawn_points()[0]
    wp = m.get_waypoint(start.location)
    # reference polyline = centre of the ego lane for 420 m, left lane offset handled via lat in [0, 1]
    route, w = [], wp
    for _ in range(420):
        route.append([w.transform.location.x, w.transform.location.y]); w = w.next(1.0)[0]
    route = np.array(route)
    actors = []
    try:
        def spawn(model, tf):
            actor = world.spawn_actor(bp.find(model), tf); actors.append(actor); return actor
        ego = spawn("vehicle.tesla.model3", start)
        lead_tf = m.get_waypoint(start.location).next(rng.uniform(30, 60))[0].transform; lead_tf.location.z += 0.3
        lead = spawn("vehicle.audi.a2", lead_tf)
        lead.set_target_velocity(carla.Vector3D(0, 0, 0))
        onc_wp = (wp.get_left_lane() or wp).next(rng.uniform(180, 320))[0]
        onc_tf = onc_wp.transform; onc_tf.rotation.yaw += 180; onc_tf.location.z += 0.3
        onc = spawn("vehicle.audi.tt", onc_tf)
        adapter = CarlaAdapter(ego, lead, onc, route, cv_times=np.sort(rng.uniform(18, 60, 3)))
        tree = bt.build_tree(); lead_v = rng.uniform(3, 7); onc_v = rng.uniform(9, 13)
        while adapter.t < a.duration and adapter.s < bt.ROUTE_END:
            world.tick()
            # simple scripted traffic: constant speed along each actor's heading
            for actor, speed in ((lead, lead_v), (onc, onc_v)):
                f = actor.get_transform().get_forward_vector(); actor.set_target_velocity(carla.Vector3D(f.x * speed, f.y * speed, 0))
            adapter.update()
            bb = bt.bb_from_world(adapter); bb["cmd"] = dict(speed=0.0, lane=0); tree.tick(bb)
            i = adapter.station(ego.get_location())[1]; tgt = route[min(i + 12, len(route) - 1)]
            ego.apply_control(adapter.command_to_control(bb["cmd"], pure_pursuit_steer(ego.get_transform(), tgt)))
            print(f"t={adapter.t:5.1f} s={adapter.s:6.1f} v={adapter.v:4.1f} lat={adapter.lat:.2f} leaf={bb['active'][-1] if bb['active'] else '-'}")
    finally:
        settings.synchronous_mode = False; world.apply_settings(settings)
        for actor in actors: actor.destroy()


if __name__ == "__main__":
    main()
