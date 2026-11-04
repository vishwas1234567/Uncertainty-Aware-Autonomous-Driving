"""
HD-map localization: NDT scan alignment + particle filter (NDT-PF).

Components
----------
make_world()          synthetic HD map (road with irregular building walls) + ground-truth drive
NDTMap                Normal Distributions Transform grid built from the pre-mapped HD map points
ndt_align()           Newton scan-to-map alignment (local refinement)
NDTParticleFilter     particle filter whose measurement likelihood is the NDT score, plus
                      "score-gated NDT injection" and score-triggered cloud inflation (recovery)

Only numpy is required.
"""
import numpy as np


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def transform_batch(poses, pts):
    """poses (P,3), pts (N,2) in vehicle frame -> (P,N,2) in map frame."""
    c = np.cos(poses[:, 2])[:, None]
    s = np.sin(poses[:, 2])[:, None]
    x = c * pts[None, :, 0] - s * pts[None, :, 1] + poses[:, 0:1]
    y = s * pts[None, :, 0] + c * pts[None, :, 1] + poses[:, 1:2]
    return np.stack([x, y], axis=-1)


# --------------------------------------------------------------------------- world
def make_world(seed=0, step=0.5):
    """Closed road with irregular building setbacks. Returns map points and the drive path."""
    rng = np.random.default_rng(seed)
    phi = np.linspace(0, 2 * np.pi, 4000, endpoint=False)
    r = 60 + 12 * np.sin(3 * phi) + 6 * np.cos(2 * phi)
    xy = np.c_[r * np.cos(phi), r * np.sin(phi)]
    seg = np.linalg.norm(np.diff(np.vstack([xy, xy[:1]]), axis=0), axis=1)
    arc = np.concatenate([[0], np.cumsum(seg)])
    total = arc[-1]
    s_grid = np.arange(0, total, step)
    px = np.interp(s_grid, arc, np.append(xy[:, 0], xy[0, 0]))
    py = np.interp(s_grid, arc, np.append(xy[:, 1], xy[0, 1]))
    path = np.c_[px, py]
    d = np.gradient(path, axis=0)
    heading = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    normal = np.c_[-np.sin(heading), np.cos(heading)]

    block = 12.0  # metres per building block, each with its own setback
    n_blocks = int(total // block) + 1
    pts = []
    for side in (+1, -1):
        offs = rng.uniform(7.0, 15.0, n_blocks + 1)
        for i, s in enumerate(s_grid):
            k = int(s // block)
            off = offs[k]
            pts.append(path[i] + side * off * normal[i])
            if i > 0 and int(s_grid[i - 1] // block) != k:  # facade connecting two setbacks
                for o in np.arange(min(offs[k - 1], off), max(offs[k - 1], off), step):
                    pts.append(path[i] + side * o * normal[i])
    # sparse poles near the kerb add extra structure
    for i in range(0, len(s_grid), 14):
        pts.append(path[i] + rng.choice([-1, 1]) * 4.5 * normal[i])
    pts = np.array(pts)
    return pts, path, heading, total


def simulate_scan(pose, map_pts, rng, n=80, rng_max=25.0, sigma=0.10, outlier_frac=0.10):
    """Lidar-like scan: random visible map points within range, in the vehicle frame."""
    d = map_pts - pose[:2]
    dist = np.linalg.norm(d, axis=1)
    cand = np.where(dist < rng_max)[0]
    if len(cand) < 5:
        return np.zeros((0, 2))
    sel = rng.choice(cand, size=min(n, len(cand)), replace=False)
    c, s = np.cos(-pose[2]), np.sin(-pose[2])
    R = np.array([[c, -s], [s, c]])
    local = (d[sel] @ R.T) + rng.normal(0, sigma, (len(sel), 2))
    k = int(outlier_frac * len(local))
    if k:
        local[:k] = rng.uniform(-rng_max, rng_max, (k, 2))
    return local


# --------------------------------------------------------------------------- NDT
class NDTMap:
    """Grid of Gaussians (mean, inverse covariance) fitted to the HD map points."""

    def __init__(self, pts, cell=4.0, min_pts=4, reg=0.15):
        self.cell = cell
        self.origin = pts.min(0) - cell
        self.dims = np.ceil((pts.max(0) + cell - self.origin) / cell).astype(int) + 1
        self.mean = np.zeros((*self.dims, 2))
        self.icov = np.zeros((*self.dims, 2, 2))
        self.valid = np.zeros(self.dims, bool)
        idx = np.floor((pts - self.origin) / cell).astype(int)
        flat = idx[:, 0] * self.dims[1] + idx[:, 1]
        order = np.argsort(flat)
        flat_s, pts_s = flat[order], pts[order]
        bounds = np.flatnonzero(np.diff(flat_s)) + 1
        for chunk, f in zip(np.split(pts_s, bounds), flat_s[np.r_[0, bounds]]):
            if len(chunk) < min_pts:
                continue
            ix, iy = divmod(int(f), int(self.dims[1]))
            mu = chunk.mean(0)
            C = np.cov(chunk.T) + reg * np.eye(2)  # reg keeps wall cells from being degenerate
            self.mean[ix, iy] = mu
            self.icov[ix, iy] = np.linalg.inv(C)
            self.valid[ix, iy] = True

    def score(self, poses, scan):
        """Sum over scan points of exp(-0.5 d^T C^-1 d) for each pose -> (P,)."""
        if len(scan) == 0:
            return np.zeros(len(poses))
        w = transform_batch(poses, scan)
        idx = np.floor((w - self.origin) / self.cell).astype(int)
        inb = (idx >= 0).all(-1) & (idx[..., 0] < self.dims[0]) & (idx[..., 1] < self.dims[1])
        ix = np.clip(idx[..., 0], 0, self.dims[0] - 1)
        iy = np.clip(idx[..., 1], 0, self.dims[1] - 1)
        ok = inb & self.valid[ix, iy]
        d = w - self.mean[ix, iy]
        m = np.einsum("pni,pnij,pnj->pn", d, self.icov[ix, iy], d)
        return np.where(ok, np.exp(-0.5 * m), 0.0).sum(1)


def ndt_align(ndt, scan, init, iters=10, damping=1e-2, max_step=(1.0, 1.0, 0.1)):
    """Maximise the NDT score with damped Newton steps (finite-difference derivatives)."""
    pose = np.array(init, float)
    eps = np.array([0.05, 0.05, 0.005])
    max_step = np.array(max_step)
    for _ in range(iters):
        # build batch: centre, +-e_i, and four combos per pair
        batch = [pose]
        for i in range(3):
            for sgn in (+1, -1):
                p = pose.copy(); p[i] += sgn * eps[i]; batch.append(p)
        pairs = [(0, 1), (0, 2), (1, 2)]
        for i, j in pairs:
            for si, sj in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                p = pose.copy(); p[i] += si * eps[i]; p[j] += sj * eps[j]; batch.append(p)
        f = ndt.score(np.array(batch), scan)
        f0 = f[0]
        g = np.array([(f[1 + 2 * i] - f[2 + 2 * i]) / (2 * eps[i]) for i in range(3)])
        H = np.zeros((3, 3))
        for i in range(3):
            H[i, i] = (f[1 + 2 * i] - 2 * f0 + f[2 + 2 * i]) / eps[i] ** 2
        for k, (i, j) in enumerate(pairs):
            b = 7 + 4 * k
            H[i, j] = H[j, i] = (f[b] - f[b + 1] - f[b + 2] + f[b + 3]) / (4 * eps[i] * eps[j])
        # maximise f: need negative-definite H; otherwise fall back to gradient ascent
        Hn = -H + damping * np.eye(3) * max(1.0, np.abs(np.diag(H)).max())
        try:
            if np.all(np.linalg.eigvalsh(Hn) > 0):
                step = np.linalg.solve(Hn, g)
            else:
                step = g / (np.abs(g).max() + 1e-9) * max_step * 0.3
        except np.linalg.LinAlgError:
            break
        step = np.clip(step, -max_step, max_step)
        new = pose + step
        if ndt.score(new[None], scan)[0] < f0:  # backtrack once
            new = pose + 0.3 * step
            if ndt.score(new[None], scan)[0] < f0:
                break
        pose = new
        if np.abs(step).max() < 1e-3:
            break
    pose[2] = wrap(pose[2])
    return pose, ndt.score(pose[None], scan)[0]


# --------------------------------------------------------------------------- PF
class NDTParticleFilter:
    """
    Particle filter with an NDT likelihood.

    Two extras beyond a textbook PF (the project's own contribution):
      * score-gated NDT injection: after each update, run a Newton NDT alignment from the weighted
        mean; if it scores clearly better than the mean, the worst particles are re-seeded around
        the refined pose (and the estimate is the refined pose). This fixes particle-set
        impoverishment/quantisation error without trusting NDT blindly (it only acts when the
        score confirms it).
      * score-triggered inflation: if the normalised score stays low for several steps (map
        mismatch, drift, kidnapping) the process noise is inflated so the cloud can re-acquire.
    """

    def __init__(self, ndt, init_pose, init_std=(2.0, 2.0, 0.15), n=300, seed=0,
                 use_injection=True, temper=0.5):
        self.ndt, self.n, self.rng = ndt, n, np.random.default_rng(seed)
        self.p = np.array(init_pose) + self.rng.normal(0, 1, (n, 3)) * np.array(init_std)
        self.w = np.full(n, 1.0 / n)
        self.use_injection, self.temper = use_injection, temper
        self.low_count, self.inflate = 0, 1.0
        self.est = np.array(init_pose, float)
        self.last_score = 0.0

    def predict(self, v, w, dt):
        s = self.inflate
        vn = v + self.rng.normal(0, 0.20 * s, self.n)
        wn = w + self.rng.normal(0, 0.04 * s, self.n)
        th = self.p[:, 2]
        self.p[:, 0] += vn * dt * np.cos(th + 0.5 * wn * dt)
        self.p[:, 1] += vn * dt * np.sin(th + 0.5 * wn * dt)
        self.p[:, 2] = wrap(th + wn * dt)

    def update(self, scan, n_ref=None):
        sc = self.ndt.score(self.p, scan)
        n = max(len(scan), 1)
        norm = sc / n  # 0..1 : fraction of points explained
        logw = self.temper * np.log(sc + 1e-3 * n) + np.log(self.w + 1e-300)
        logw -= logw.max()
        self.w = np.exp(logw); self.w /= self.w.sum()
        ess = 1.0 / np.sum(self.w ** 2)
        if ess < self.n / 2:
            self._resample()
        self.est = self._mean()
        self.last_score = float(self.ndt.score(self.est[None], scan)[0] / n)

        if self.use_injection:
            ref, rs = ndt_align(self.ndt, scan, self.est, iters=6)
            ref_norm = rs / n
            if ref_norm > self.last_score + 0.02 and np.linalg.norm(ref[:2] - self.est[:2]) < 3.0:
                k = self.n // 5
                worst = np.argsort(self.w)[:k]
                self.p[worst] = ref + self.rng.normal(0, 1, (k, 3)) * np.array([0.25, 0.25, 0.02])
                self.w[worst] = self.w.max()
                self.w /= self.w.sum()
                self.est = ref
                self.last_score = float(ref_norm)
        # recovery logic
        if self.last_score < 0.25:
            self.low_count += 1
        else:
            self.low_count = 0
        self.inflate = 3.0 if self.low_count >= 5 else 1.0
        return self.est

    def _resample(self):
        pos = (self.rng.random() + np.arange(self.n)) / self.n
        idx = np.minimum(np.searchsorted(np.cumsum(self.w), pos), self.n - 1)
        self.p = self.p[idx]
        self.w = np.full(self.n, 1.0 / self.n)

    def _mean(self):
        x = np.sum(self.w * self.p[:, 0]); y = np.sum(self.w * self.p[:, 1])
        th = np.arctan2(np.sum(self.w * np.sin(self.p[:, 2])), np.sum(self.w * np.cos(self.p[:, 2])))
        return np.array([x, y, th])
