"""
Reproduce everything: tests -> experiments -> figures -> results/*.log (+ results/*.json).
  python run_all.py --quick   tests + localisation + BT batch + closed loop + latency  (~2 min)
  python run_all.py --full    additionally BT grid verification and fault-injection campaigns (~20 min)
Independent experiments run in parallel processes; each log is the exact stdout of its script.
"""
import os, subprocess, sys, time

JOBS_QUICK = [
    ("localization", [sys.executable, "-u", "run_localization.py", "--plot"]),
    ("bt_batch", [sys.executable, "-u", "behavior_tree.py", "--plot"]),
    ("closed_loop", [sys.executable, "-u", "closed_loop.py", "--plot"]),
    ("latency", [sys.executable, "-u", "deploy/embedded_bench.py"]),
]
JOBS_FULL = [
    ("bt_verification", [sys.executable, "-u", "verify_bt.py"]),
    ("fault_injection", [sys.executable, "-u", "fault_injection.py"]),
]


def main():
    os.makedirs("results", exist_ok=True); os.makedirs("figures", exist_ok=True)
    for t in ("tests.test_all", "tests.test_datasets"):
        r = subprocess.run([sys.executable, "-m", t], capture_output=True, text=True)
        open(f"results/{t.split('.')[-1]}.log", "w").write(r.stdout + r.stderr)
        print(("PASS " if r.returncode == 0 else "FAIL ") + t)
        if r.returncode:
            sys.exit(r.stderr)
    jobs = JOBS_QUICK + (JOBS_FULL if "--full" in sys.argv else [])
    procs = []
    for name, cmd in jobs:
        f = open(f"results/{name}.log", "w")
        procs.append((name, subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT), f, time.time()))
    for name, p, f, t0 in procs:
        p.wait(); f.close()
        print(f"{'done' if p.returncode == 0 else 'FAILED'} {name} ({time.time() - t0:.0f}s) -> results/{name}.log")


if __name__ == "__main__":
    main()
