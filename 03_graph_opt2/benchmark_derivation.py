import os

os.environ["WICK_PROFILE"] = "0"
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/pywick-mpl")
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
import json
from pathlib import Path
from time import perf_counter

from derive import derive_equation


def run(args):
    result = dict(
        max_rank=args.rank,
        generation=args.generation,
        max_order=4,
        unordered_clusters=not args.ordered,
        delta_mode="map",
        equations=[],
    )
    ranks = range(args.rank + 1) if args.projection is None else (args.projection,)
    for rank in ranks:
        equation, stats = derive_equation(
            args.rank,
            rank,
            unordered=not args.ordered,
            generation=args.generation,
            delta_mode="map",
            deadline=perf_counter() + args.timeout,
            progress=lambda value: print(json.dumps(value), flush=True),
        )
        result["equations"].append(stats)
        result["derivation_seconds"] = sum(s["seconds"] for s in result["equations"])
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, choices=(2, 3, 4), required=True)
    parser.add_argument("--projection", type=int)
    parser.add_argument(
        "--generation", choices=("orbit", "slots", "labelled"), default="orbit"
    )
    parser.add_argument(
        "--ordered", action="store_true", help="Keep ordered T products"
    )
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.projection is not None and not 0 <= args.projection <= args.rank:
        parser.error("projection must be between 0 and rank")
    run(args)
