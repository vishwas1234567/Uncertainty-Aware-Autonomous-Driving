"""
Public-dataset utilities: KITTI odometry loader (velodyne .bin, poses, calib) and trajectory metrics
(ATE with SE(2)/SE(3) alignment, RPE). NOTE: the loader is validated here on synthetic data written in the
KITTI file format (tests/test_datasets.py); it has NOT yet been run on the real KITTI dataset.

Expected layout:  <root>/sequences/00/velodyne/000000.bin ... ,  <root>/poses/00.txt
"""
import os
import numpy as np


def load_velodyne(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)          # x, y, z, intensity


def load_poses(path):
    """KITTI poses: each line = 12 numbers of a 3x4 [R|t] camera-to-world matrix. Returns (N,4,4)."""
    P = np.loadtxt(path).reshape(-1, 3, 4)
    T = np.tile(np.eye(4), (len(P), 1, 1)); T[:, :3, :4] = P
    return T


def scan_to_2d(points, z_min=-1.5, z_max=1.5, rng_max=40.0, n=120, rng=None):
    """Project a 3-D lidar scan to the 2-D point set used by the NDT-PF (height slab, range gate, subsample)."""
    rng = rng or np.random.default_rng(0)
    m = (points[:, 2] > z_min) & (points[:, 2] < z_max) & (np.hypot(points[:, 0], points[:, 1]) < rng_max)
    p = points[m, :2]
    return p[rng.choice(len(p), min(n, len(p)), replace=False)] if len(p) else p


def ate_rmse(est_xy, gt_xy, align=True):
    """Absolute trajectory error (RMSE, m) after optional rigid alignment (Umeyama without scale)."""
    est, gt = np.asarray(est_xy, float), np.asarray(gt_xy, float)
    if align:
        mu_e, mu_g = est.mean(0), gt.mean(0)
        H = (est - mu_e).T @ (gt - mu_g)
        U, _, Vt = np.linalg.svd(H)
        D = np.diag([1, np.sign(np.linalg.det(Vt.T @ U.T))])
        R = Vt.T @ D @ U.T
        est = (R @ (est - mu_e).T).T + mu_g
    return float(np.sqrt(np.mean(np.sum((est - gt) ** 2, axis=1))))


def rpe_translation(est_xy, gt_xy, delta=10):
    """Relative pose error: translation drift over `delta` frames (mean, m)."""
    de = est_xy[delta:] - est_xy[:-delta]; dg = gt_xy[delta:] - gt_xy[:-delta]
    return float(np.mean(np.linalg.norm(de - dg, axis=1)))


def kitti_sequence(root, seq="00"):
    vdir = os.path.join(root, "sequences", seq, "velodyne")
    files = sorted(f for f in os.listdir(vdir) if f.endswith(".bin"))
    poses = load_poses(os.path.join(root, "poses", f"{seq}.txt"))
    return [os.path.join(vdir, f) for f in files], poses
