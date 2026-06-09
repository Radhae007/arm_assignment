import os
import sys
import numpy as np
import csv

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from planning import dh_target as dh
from planning import fk_logic_target as fk_logic
from planning import ik_logic_target as ik_logic

def validate(filename):
    fk_errors = []
    ik_errors = []
    
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        # Find indices
        header = [h.strip() for h in header]
        q1_idx = header.index('q1')
        q2_idx = header.index('q2')
        q3_idx = header.index('q3')
        x_idx = header.index('x')
        y_idx = header.index('y')
        z_idx = header.index('z')
        
        for row in reader:
            if not row: continue
            q_true = np.array([float(row[q1_idx]), float(row[q2_idx]), float(row[q3_idx])])
            pos_true = np.array([float(row[x_idx]), float(row[y_idx]), float(row[z_idx])])
            
            # Test FK
            pos_fk, _ = fk_logic.fk(q_true)
            fk_err = np.linalg.norm(pos_fk - pos_true)
            fk_errors.append(fk_err)
            
            # Test IK
            sol_down = ik_logic.solve(pos_true, elbow="down")
            sol_up = ik_logic.solve(pos_true, elbow="up")
            
            ik_err = min(sol_down.position_error, sol_up.position_error)
            ik_errors.append(ik_err)
            
    fk_errors = np.array(fk_errors)
    ik_errors = np.array(ik_errors)
    
    print("\n--- Validation Results against _target_sol.py files ---")
    print(f"Total targets tested: {len(fk_errors)}")
    print(f"FK Max Error : {fk_errors.max()*1000:.6f} mm")
    print(f"FK Mean Error: {fk_errors.mean()*1000:.6f} mm")
    print(f"IK Max Error : {ik_errors.max()*1000:.6f} mm")
    print(f"IK Mean Error: {ik_errors.mean()*1000:.6f} mm")
    
if __name__ == "__main__":
    csv_file = os.path.join(ROOT_DIR, "docs", "targets.csv")
    validate(csv_file)
