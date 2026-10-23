import os

os.environ["WICK_PROFILE"] = "0"
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/pywick-mpl")
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import platform
import subprocess
import sys
import tempfile
from time import perf_counter

HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parent / "02_graph_opt"


def worker(args):
    source = REFERENCE if args.worker == "reference" else HERE
    sys.path.insert(0, str(source))
    from derive import derive_equation
    from graph_canonical import GraphCanonicalizer, cluster_symmetries
    import igraph

    result = dict(
        method=args.worker,
        max_rank=args.rank,
        max_order=4,
        unordered_clusters=True,
        delta_mode="map",
        threads=1,
        python=platform.python_version(),
        platform=platform.platform(),
        igraph=igraph.__version__,
        source=str(source),
        equations=[],
    )
    coefficients = []
    for projection in range(args.rank + 1):
        print(
            f"[{args.worker}] CC rank {args.rank}, projection {projection}", flush=True
        )
        options = dict(
            unordered=True,
            delta_mode="map",
            deadline=perf_counter() + args.timeout,
            progress=lambda stats: print(json.dumps(stats), flush=True),
        )
        if args.worker != "reference":
            options["generation"] = args.worker
        tick = perf_counter()
        equation, stats = derive_equation(args.rank, projection, **options)
        stats["measured_derivation_seconds"] = perf_counter() - tick
        result["equations"].append(stats)
        # Explicit full dictionary, not just term counts or a digest comparison.
        canonicalizer = GraphCanonicalizer(cluster_symmetries(args.rank))
        table = {}
        for term in equation.terms:
            key, sign = canonicalizer.canonicalize(term)
            if sign:
                table[key] = table.get(key, 0) + sign * term.factor
        table = {key: value for key, value in table.items() if value}
        coefficients.append(table)
        result["derivation_seconds"] = sum(
            s["measured_derivation_seconds"] for s in result["equations"]
        )
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    Path(args.coefficients).write_bytes(pickle.dumps(coefficients, protocol=5))


def compare(args):
    result = dict(
        configuration=dict(
            max_order=4,
            unordered_clusters=True,
            delta_mode="map",
            threads=1,
            exact_fraction_comparison=True,
            timing="derivation only; excludes imports and coefficient comparison",
        ),
        comparisons=[],
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pywick-03-") as temporary:
        temporary = Path(temporary)
        for rank in args.ranks:
            runs, tables = {}, {}
            for method in ("reference", "orbit"):
                stats_path = temporary / f"{method}-{rank}.json"
                table_path = temporary / f"{method}-{rank}.pickle"
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    method,
                    "--rank",
                    str(rank),
                    "--timeout",
                    str(args.timeout),
                    "--output",
                    str(stats_path),
                    "--coefficients",
                    str(table_path),
                ]
                subprocess.run(
                    command, check=True, env=dict(os.environ, PYTHONHASHSEED="0")
                )
                runs[method] = json.loads(stats_path.read_text())
                tables[method] = pickle.loads(table_path.read_bytes())
            checks = []
            for projection, (old, new) in enumerate(
                zip(tables["reference"], tables["orbit"])
            ):
                if old != new:
                    missing, extra = old.keys() - new.keys(), new.keys() - old.keys()
                    different = sum(old[k] != new[k] for k in old.keys() & new.keys())
                    raise AssertionError(
                        f"rank={rank}, projection={projection}: "
                        f"missing={len(missing)}, extra={len(extra)}, "
                        f"different coefficients={different}"
                    )
                old_count = runs["reference"]["equations"][projection][
                    "connected_terms"
                ]
                new_count = runs["orbit"]["equations"][projection]["connected_terms"]
                assert old_count == new_count, (old_count, new_count)
                # Digest is retained for reproducibility, after exact comparison above.
                digest = hashlib.sha256(repr(sorted(old.items())).encode()).hexdigest()
                checks.append(
                    dict(
                        projection=projection,
                        exact_match=True,
                        terms=len(old),
                        connected_cardinality=old_count,
                        coefficient_sha256=digest,
                    )
                )
            entry = dict(
                max_rank=rank,
                runs=runs,
                validation=checks,
                speedup=runs["reference"]["derivation_seconds"]
                / runs["orbit"]["derivation_seconds"],
            )
            result["comparisons"].append(entry)
            output.write_text(json.dumps(result, indent=2) + "\n")
            print(
                f"[verified] rank={rank} all coefficients equal; "
                f'{entry["speedup"]:.2f}x derivation speedup',
                flush=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ranks", type=int, nargs="+", choices=(2, 3, 4), default=[3, 4]
    )
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--output", default=str(HERE / "benchmark_comparison.json"))
    parser.add_argument("--worker", choices=("reference", "orbit", "slots"))
    parser.add_argument("--rank", type=int, choices=(2, 3, 4))
    parser.add_argument("--coefficients")
    args = parser.parse_args()
    if args.worker:
        if args.rank is None or not args.coefficients:
            parser.error("worker requires --rank and --coefficients")
        worker(args)
    else:
        compare(args)
