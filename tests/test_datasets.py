"""Run: python -m tests.test_datasets  (validates the KITTI-format loader and the metrics on synthetic data)."""
import os, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import load_velodyne, load_poses, scan_to_2d, ate_rmse, rpe_translation, kitti_sequence


def main():
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as root:
        os.makedirs(f"{root}/sequences/00/velodyne"); os.makedirs(f"{root}/poses")
        pts = rng.normal(0, 10, (2000, 4)).astype(np.float32); pts[:, 2] = rng.uniform(-2, 2, 2000)
        for i in range(3):
            pts.tofile(f"{root}/sequences/00/velodyne/{i:06d}.bin")
        P = np.array([np.r_[np.eye(3).ravel().reshape(3, 3), [[i * 1.0], [0], [0]]].reshape(-1) if False else
                      np.hstack([np.eye(3), [[i * 1.0], [0.0], [0.0]]]).ravel() for i in range(3)])
        np.savetxt(f"{root}/poses/00.txt", P)
        files, poses = kitti_sequence(root)
        assert len(files) == 3 and poses.shape == (3, 4, 4) and abs(poses[2, 0, 3] - 2.0) < 1e-9
        cloud = load_velodyne(files[0]); assert cloud.shape == (2000, 4)
        assert 0 < len(scan_to_2d(cloud)) <= 120
    gt = np.c_[np.linspace(0, 100, 200), 5 * np.sin(np.linspace(0, 6, 200))]
    th = 0.3; R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    est = gt @ R.T + [3.0, -2.0]
    assert ate_rmse(est, gt, align=True) < 1e-9                      # pure rigid offset is removed by alignment
    assert ate_rmse(est, gt, align=False) > 1.0
    noisy = gt + rng.normal(0, 0.2, gt.shape)
    assert 0.15 < ate_rmse(noisy, gt) < 0.35 and rpe_translation(noisy, gt) > 0
    print("test_datasets: OK")


if __name__ == "__main__":
    main()
