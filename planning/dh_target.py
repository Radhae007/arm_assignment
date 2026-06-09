import numpy as np

JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex")

# FIX 1: Explicitly structure columns as [a, d, alpha] to match standard DH convention
# Row 1: a=0,       d=0.0595, alpha=+pi/2
# Row 2: a=0.1175,  d=0,      alpha=0
# Row 3: a=0.0950,  d=0,      alpha=0
DH_PARAMS = np.array([
    [0. ,      0.0595,  1.57079632679], 
    [0.1175,   0.0,     0.0],
    [0.0950,   0.0,     0.0],
], dtype=float)

EE_OFFSET = 0.0100

JOINT_LIMITS = np.array([
    [-1.91986, 1.91986],
    [-1.7453293, 1.7453293],
    [-1.69, 1.69],
], dtype=float)

# Tool Center Point Translation Matrix
Tacp = np.array([[1, 0, 0, EE_OFFSET],
                 [0, 1, 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]])


def dh_transform(a, d, alpha, theta):
    ctheta = np.cos(theta)
    stheta = np.sin(theta)
    calpha = np.cos(alpha)
    salpha = np.sin(alpha)
    
    T = np.array([[ctheta, -stheta*calpha,  stheta*salpha, a*ctheta],
                  [stheta,  ctheta*calpha, -ctheta*salpha, a*stheta],
                  [0,       salpha,         calpha,        d], 
                  [0,       0,              0,             1]])
    return T


def fk(q):
    # FIX 3: Initialize with an Identity Matrix to multiply from Base -> Tip
    Fk = np.eye(4)
    
    # FIX 2: Use enumerate to pull the correct single joint angle 'q[i]' for each link
    for i, param in enumerate(DH_PARAMS):
        a = param[0]
        d = param[1]
        alpha = param[2]
        
        T = dh_transform(a, d, alpha, q[i])
        Fk = np.dot(Fk, T)  # Chain transforms forward
        
    # Apply the final tool center point offset at the end of the chain
    Fk = np.dot(Fk, Tacp)
    return Fk


def position(q):
    return fk(q)[:3, 3].copy()


def print_table():
    print("Teaching DH table")
    print("  i  joint            theta     d (m)    a (m)    alpha (rad)")
    for i, (name, (a, d, alpha)) in enumerate(zip(JOINT_NAMES, DH_PARAMS), start=1):
        print(f"  {i:<2} {name:<15} q{i:<7} {d:8.4f} {a:8.4f} {alpha:12.6f}")
    print(f"  end-effector offset along final x-axis: {EE_OFFSET:.4f} m\n")

# --- Verification Test ---
if __name__ == "__main__":
    print_table()
    
    # Test using the 'ready' keyframe from documentation: [0, 0.6, -0.8] rad
    q_ready = np.array([0.0, 0.6, -0.8])
    pos = position(q_ready)
    
    print("Validation Check (Ready Keyframe):")
    print(f"Calculated TCP Position: X={pos[0]:.4f}m, Y={pos[1]:.4f}m, Z={pos[2]:.4f}m")
    print("Expected Target Position: X=0.1998m,  Y=0.0000m,  Z=0.1049m")