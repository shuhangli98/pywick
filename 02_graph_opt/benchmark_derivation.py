import os

os.environ["WICK_PROFILE"] = "0"
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/pywick-mpl")
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
import argparse
import json
from pathlib import Path
import platform
from time import perf_counter
import igraph
from derive import derive_equation


def run(args):
    result = dict(
        max_rank=args.rank,
        method={2: "CCSD", 3: "CCSDT", 4: "CCSDTQ"}[args.rank],
        max_order=4,
        unordered_clusters=args.unordered,
        delta_mode=args.delta_mode,
        timeout_per_equation_seconds=args.timeout,
        backend="igraph " + igraph.__version__ + " / BLISS",
        python=platform.python_version(),
        equations=[],
    )
    start = perf_counter()
    ranks = list(range(args.rank + 1)) if args.projection is None else [args.projection]
    for rank in ranks:
        print(f'[derive] {result["method"]} projection rank {rank}', flush=True)

        def progress(info):
            print(json.dumps(info), flush=True)

        try:
            equation, stats = derive_equation(
                args.rank,
                rank,
                unordered=args.unordered,
                progress=progress,
                deadline=perf_counter() + args.timeout,
                delta_mode=args.delta_mode,
            )
        except TimeoutError as error:
            stats = dict(error.args[0], status="timeout")
        result["equations"].append(stats)
        result["wall_seconds"] = perf_counter() - start
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, choices=(2, 3, 4), required=True)
    parser.add_argument("--projection", type=int)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--delta-mode", choices=("copy", "map"), default="copy")
    parser.add_argument(
        "--unordered", action="store_true", help="Use commuting-T multinomial expansion"
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.projection is not None and not 0 <= args.projection <= args.rank:
        parser.error("projection must be between 0 and rank")
    run(args)
