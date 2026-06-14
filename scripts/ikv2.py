import argparse
import os
import sys
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from planning import dh_target as dh
from planning import ik_logic_target as ik_logicv2

def main():
    parser = argparse.ArgumentParser(description="Teaching DH pure IK for SO101 3-DOF")
    parser.add_argument("--target", nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument("--seed", nargs=3, type=float, metavar=("Q1", "Q2", "Q3"))
    parser.add_argument("--elbow", choices=("down", "up"), default="down")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--n", type=int, default=200)
    args = parser.parse_args()

    dh.print_table()

    if args.sweep:
        ik_logicv2.run_sweep(args.n)
        return

    if args.target is None:
        parser.error("--target is required unless --sweep is used")

    sol = ik_logicv2.solve(
        np.array(args.target, dtype=float),
        np.array(args.seed, dtype=float) if args.seed else None,
        elbow=args.elbow,
    )
    fk_pos = dh.position(sol.q)
    ik_logicv2.print_solution(np.array(args.target, dtype=float), sol, fk_pos)

if __name__ == "__main__":
    main()
