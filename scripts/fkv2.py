import argparse
import os
import sys
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from planning import fk_logic_target as fk_logicv2


def main():
    parser = argparse.ArgumentParser(description="SO101 3-DOF Pure DH FK checks")
    parser.add_argument("--q", nargs=3, type=float, metavar=("Q1", "Q2", "Q3"), help="Joint angles in radians")
    parser.add_argument("--sweep", action="store_true", help="Run a random FK sweep")
    parser.add_argument("--n", type=int, default=500, help="Number of sweep samples")
    args = parser.parse_args()

    from planning import dh_target as dh
    dh.print_table()

    if args.q is not None:
        q = np.array(args.q, dtype=float)
        pos, rot = fk_logicv2.fk(q)
        print("Single configuration results:")
        fk_logicv2.print_fk_row("input", q, pos)
        print("EE rotation matrix:")
        print(np.array2string(rot, precision=6, suppress_small=True))
        return

    fk_logicv2.run_known_poses()

    if args.sweep:
        fk_logicv2.run_sweep(args.n)

if __name__ == "__main__":
    main()
