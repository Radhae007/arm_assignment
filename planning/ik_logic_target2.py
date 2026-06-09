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

def solve(target, q0=None, elbow="down"):
    """
    Analytical inverse kinematics for the 3-DOF arm.
    """
    # 1. Clean up input target array first
    target = np.asarray(target, dtype=float).reshape(3)
    limits = dh.JOINT_LIMITS
    
    x = target[0]
    y = target[1]
    si = target[2] # Total orientation frame angle (phi)

    # 2. Extract Link Lengths from DH parameters
    L1 = np.sqrt(dh.DH_PARAMS[0, 0]**2 + dh.DH_PARAMS[0, 1]**2) # a1 + d1
    L2 = np.sqrt(dh.DH_PARAMS[1, 0]**2 + dh.DH_PARAMS[1, 1]**2) # a2 + d2
    L3 = np.sqrt(dh.DH_PARAMS[2, 0]**2 + dh.DH_PARAMS[2, 1]**2) + dh.EE_OFFSET # a3 + d3

    # 3. Initialize joint array
    q = np.zeros(3)

    # 4. Find the wrist center position (decoupling)
    x1 = x - L3 * np.cos(si)
    y1 = y - L3 * np.sin(si)

    # 5. Calculate Joint 2 (Python index 1) via Cosine Rule
    # Added defensive clipping so floating-point noise doesn't cause nan errors
    cos_q1 = (x1**2 + y1**2 - L1**2 - L2**2) / (2 * L1 * L2)
    cos_q1 = np.clip(cos_q1, -1.0, 1.0) 
    
    # Select elbow configuration based on parameter string
    if elbow == "down":
        q[1] = -np.arccos(cos_q1) + np.pi/2  # Negative angle changes elbow orientation
    else:
        q[1] = np.arccos(cos_q1)  # Elbow Up standard root
    

    # 6. Calculate Joint 1 (Python index 0)
    a = np.arctan2(y1, x1) 
    b = np.arctan2(L2 * np.sin(q[1]), L1 + L2 * np.cos(q[1]))
    # if a > 0:
    #     q[0] = a - b
    # else : 
    #     q[0] = -a + np.pi/2 -b 
    q[0] = a - b
    
    # 7. Calculate Joint 3 (Python index 2)
    q[2] = si - q[0] - q[1]

    # Optional but highly recommended: Keep angles clean and bound checked
    # q[0] = wrap_to_pi(q[0])
    # q[1] = wrap_to_pi(q[1])
    # q[2] = wrap_to_pi(q[2])
    
    # Clamp values to workspace safety constraints
    q = clamp_to_limits(q, limits)

    # 8. Compute accuracy and return validation metrics
    # Assuming dh.position(q) calculates the forward kinematics position matrix/vector
    current_fk_pos = dh.position(q)
    err = float(np.linalg.norm(target[:2] - current_fk_pos[:2])) # Position error check
    
    # Determine convergence tracking
    converged = err < 1e-3 # Converged if under 1 millimeter error threshold
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
        sol = solve(target, elbow="down")
        errors.append(sol.position_error)

    errors = np.array(errors)
    print(f"Sweep: {n} pure-DH targets")
    print(f"  max error  = {errors.max() * 1000:.6f} mm")
    print(f"  mean error = {errors.mean() * 1000:.6f} mm")
    print(f"  all < 1 mm = {'YES' if np.all(errors < 1e-3) else 'NO'}")