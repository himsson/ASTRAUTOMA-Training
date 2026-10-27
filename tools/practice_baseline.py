"""Замер практики без обучения: навигатор против точного планировщика."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if __name__ == "__main__":
    from kia_core.neural import practice as P, nav_plan
    ag = nav_plan.build_agent()
    nav_plan.load(ag)
    pool = P.Pool(10, 512)
    try:
        a = P.evaluate(ag, pool, rounds=2)
        e = P.evaluate(ag, pool, rounds=2, expert=True)
        keys = ("overall", "flown_success", "aborted", "rocket_limited", "worst_pair")
        print("навигатор:", {k: round(a[k], 3) for k in keys})
        print("потолок:  ", {k: round(e[k], 3) for k in keys})
        print("слабые пары:", sorted(a["pairs"].items(), key=lambda x: x[1])[:6])
    finally:
        pool.close()
