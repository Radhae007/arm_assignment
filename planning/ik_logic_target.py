import os
import sys
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from planning import dh_target as dh

class IKSolution:
    def __init__(self, q, converged, iterations, position_error):
        self.q = q
        self.converged = converged
        self.iterations = iterations
        self.position_error = position_error

def clamp_to_limits(q, limits):
    return np.clip(q, limits[:, 0], limits[:, 1])

def wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi

def solve(target, q0=None, elbow="up"):
    """
    Analytical inverse kinematics for the 3-DOF spatial SO101 arm.
    
    Parameters:
    - target: list or np.array of [x, y, z] target coordinates in meters.
    - q0: optional initial guess (ignored in this closed-form analytical solver).
    - elbow: "up" (preferred configuration) or "down".
    """
    # 1. Clean up input target array first to map [x, y, z]
    target = np.asarray(target, dtype=float).reshape(3)
    limits = dh.JOINT_LIMITS
    
    x = target[0]
    y = target[1]
    z = target[2]

    # 2. Extract accurate structural dimensions from standard DH parameters
    # d1 = Height of shoulder axis above world origin (0.0595 m)
    # L2 = Link length of upper arm (a2 = 0.1175 m)
    # L3 = Forearm length combined with fixed tool offset (a3 + a_tcp = 0.0950 + 0.0100 = 0.1050 m)
    d1 = dh.DH_PARAMS[0, 1]
    L2 = dh.DH_PARAMS[1, 0]
    L3 = dh.DH_PARAMS[2, 0] + 0.01

    # 3. Initialize joint array [q1, q2, q3]
    q = np.zeros(3)

    # 4. Calculate Joint 1: Base Yaw (q1 / Python index 0)
    # Splitting the spatial problem by looking at the target position from top-down
    q[0] = np.arctan2(y, x)

    # 5. Project coordinates onto a 2D vertical reach plane (r, s)
    # r = horizontal radial distance outward from the base column center
    # s = vertical height relative to the shoulder lift pitching center axis
    r = np.sqrt(x**2 + y**2)
    s = z - d1

    D_sq = r**2 + s**2
    D = np.sqrt(D_sq)

    # Boundary check: Ensure target is within the absolute reach limits
    if D > (L2 + L3) or D < abs(L2 - L3):
        return IKSolution(q=q, converged=False, iterations=1, position_error=999.0)

    # 6. Calculate Joint 3: Elbow Flex (q3 / Python index 2) via the Law of Cosines
    cos_q3 = (D_sq - L2**2 - L3**2) / (2.0 * L2 * L3)
    cos_q3 = np.clip(cos_q3, -1.0, 1.0) # Prevent floating point noise from creating NaNs
    
    # Branch configuration selection
    if elbow == "up":
        q[2] = -np.arccos(cos_q3) # Mountain configuration (Safe)
    else:
        q[2] = np.arccos(cos_q3)  # Valley configuration (Elbow-Down)

    # 7. Calculate Joint 2: Shoulder Lift (q2 / Python index 1)
    gamma = np.arctan2(s, r) # Angle from shoulder center up to the target coordinate
    
    cos_alpha = (L2**2 + D_sq - L3**2) / (2.0 * L2 * D)
    cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
    alpha = np.arccos(cos_alpha) # Interior geometric offset angle

    # Apply configuration branch adjustments to match DH alignment standard
    if elbow == "up":
        q[1] = gamma + alpha
    else:
        q[1] = gamma - alpha

    # 8. Bound-check constraints and keep angles clean
    q[0] = wrap_to_pi(q[0])
    q[1] = wrap_to_pi(q[1])
    q[2] = wrap_to_pi(q[2])
    
    # q = clamp_to_limits(q, limits)

    # 9. Compute validation metrics against 3D Forward Kinematics
    current_fk_pos = dh.position(q)
    err = float(np.linalg.norm(target - current_fk_pos))
    
    converged = err < 1e-3 # Converged if under a 1 millimeter tracking threshold
    return IKSolution(q=q, converged=converged, iterations=1, position_error=err)

def print_solution(target, sol, fk_pos):
    p = fk_pos
    print(f"target      = [{target[0]:+.6f} {target[1]:+.6f} {target[2]:+.6f}] m")
    print(f"converged   = {sol.converged}")
    print(f"iterations  = {sol.iterations}")
    print(f"q           = [{sol.q[0]:+.6f} {sol.q[1]:+.6f} {sol.q[2]:+.6f}] rad")
    print(f"q_deg       = [{np.degrees(sol.q[0]):+.2f} {np.degrees(sol.q[1]):+.2f} {np.degrees(sol.q[2]):+.2f}] deg")
    print(f"dh_fk(q)    = [{p[0]:+.6f} {p[1]:+.6f} {p[2]:+.6f}] m")
    print(f"pos_error   = {sol.position_error * 1000:.6f} mm")

def run_sweep(n):
    limits = dh.JOINT_LIMITS
    rng = np.random.default_rng(0)
    qs = rng.uniform(limits[:, 0], limits[:, 1], size=(n, 3))
    errors = []
    for q_true in qs:
        target = dh.position(q_true)
        # Sweeping using elbow="up" as preferred by standard workspace configurations
        sol = solve(target, elbow="up")
        errors.append(sol.position_error)

    errors = np.array(errors)
    print(f"Sweep: {n} pure-DH targets")
    print(f"  max error  = {errors.max() * 1000:.6f} mm")
    print(f"  mean error = {errors.mean() * 1000:.6f} mm")
    print(f"  all < 1 mm = {'YES' if np.all(errors < 1e-3) else 'NO'}")