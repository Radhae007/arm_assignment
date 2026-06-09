import os
import sys
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from planning import dh_target as dh

def fk(q):
    """
    Calls the DH model to get the transform, and returns position and rotation.
    """
    # TODO: Use dh.fk(q) to get the 4x4 matrix, then extract position and rotation
    T = dh.fk(q)
    return T[:3, 3].copy(), T[:3, :3].copy()
    
    # Placeholder:
    # return np.zeros(3), np.eye(3)

def print_fk_row(label, q, pos):
    deg = np.degrees(q)
    print(f"{label:16s} | q_deg: [{deg[0]:+8.3f} {deg[1]:+8.3f} {deg[2]:+8.3f}] | EE XYZ: [{pos[0]:+.6f} {pos[1]:+.6f} {pos[2]:+.6f}] m")

def run_known_poses():
    print("Checking known poses:")
    poses = [
        (np.zeros(3), "zeros"),
        (np.array([0.0, 0.6, -0.8]), "ready values"),
        (np.array([0.0, 1.0, -1.0]), "forward values"),
        (np.array([1.2, 0.6, -0.8]), "left values")
    ]
    
    for q, label in poses:
        pos, _ = fk(q)
        print_fk_row(label, q, pos)

def run_sweep(n):
    limits = dh.JOINT_LIMITS
    rng = np.random.default_rng(0)
    qs = rng.uniform(limits[:, 0], limits[:, 1], size=(n, 3))

    positions = np.empty((n, 3))
    for i, q in enumerate(qs):
        positions[i], _ = fk(q)

    finite = np.isfinite(positions).all(axis=1)
    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)

    print(f"\nRandom sweep: {n} configurations")
    print(f"  finite EE positions: {finite.sum()}/{n}")
    print(f"  x range: [{mins[0]:+.6f}, {maxs[0]:+.6f}] m")
    print(f"  y range: [{mins[1]:+.6f}, {maxs[1]:+.6f}] m")
    print(f"  z range: [{mins[2]:+.6f}, {maxs[2]:+.6f}] m")
