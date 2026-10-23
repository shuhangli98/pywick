import os

os.environ["WICK_PROFILE"] = "0"
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/pywick-mpl")
for variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ[variable] = "1"

import argparse
import importlib.util
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import igraph
import numpy as np
import pyscf
from pyscf import gto, scf, lib
from pyscf.cc import gccsd
from ccsd import GCCSD, generate_equations

HERE = Path(__file__).resolve().parent


def load_previous():
    """Load the actual archived modules without changing our active wick module."""

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    previous_wick = load("previous_wick", HERE.parent / "01_test_opt/wick.py")
    current = sys.modules["wick"]
    try:
        sys.modules["wick"] = previous_wick
        return load("previous_ccsd", HERE.parent / "01_test_opt/ccsd.py")
    finally:
        sys.modules["wick"] = current


class GeneratedDriver(gccsd.GCCSD):
    def __init__(self, mf, generated):
        super().__init__(mf)
        self.generated = generated
        self._keys = self._keys | {"generated"}

    def update_amps(self, t1, t2, eris):
        return self.generated.update_amps(t1, t2)

    def energy(self, t1, t2, eris):
        return float(self.generated.energy(t1, t2))


def timed_updates(fn, t1, t2, repeats):
    fn(t1, t2)
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        fn(t1, t2)
        samples.append(perf_counter() - start)
    return dict(median_seconds=float(np.median(samples)), samples_seconds=samples)


def solve(driver, eris, zeros):
    driver.verbose = 0
    driver.conv_tol = 1e-10
    driver.conv_tol_normt = 1e-8
    driver.max_cycle = 100
    driver.diis = True
    driver.diis_space = 6
    start = perf_counter()
    energy, t1, t2 = driver.kernel(t1=zeros[0].copy(), t2=zeros[1].copy(), eris=eris)
    result = dict(
        seconds=perf_counter() - start,
        cycles=driver.cycles,
        converged=bool(driver.converged),
        e_corr=float(energy),
        e_tot=float(driver.e_tot),
    )
    assert driver.converged, result
    return result, t1, t2


def run(args):
    lib.num_threads(1)
    previous = load_previous()
    old_eq = previous.generate_equations(verbose=False)
    graph_eq = generate_equations()
    result = dict(
        environment=dict(
            python=platform.python_version(),
            numpy=np.__version__,
            pyscf=pyscf.__version__,
            igraph=igraph.__version__,
            threads=1,
            platform=platform.platform(),
        ),
        settings=dict(
            atom="O 0 0 0; H 0 1 0; H 0 0 1",
            basis=args.basis,
            initial_amplitudes="zero",
            diis=True,
            diis_space=6,
            conv_tol=1e-10,
            conv_tol_normt=1e-8,
            max_cycle=100,
            random_seed=20261023,
        ),
        symbolic=dict(
            previous_seconds=old_eq.seconds,
            graph_seconds=graph_eq.seconds,
            stages=graph_eq.stats,
        ),
    )
    mol = gto.M(atom=result["settings"]["atom"], basis=args.basis, verbose=0)
    mf = scf.GHF(mol).run(conv_tol=1e-14)
    assert mf.converged
    reference = gccsd.GCCSD(mf)
    eris = reference.ao2mo()
    baseline = previous.GCCSD(mf, equations=old_eq)
    generated = GCCSD(mf, equations=graph_eq)
    no = generated.nocc
    result["dimensions"] = dict(nocc=no, nvir=generated.nvir)
    result["e_hf"] = float(mf.e_tot)
    zeros = generated.init_amps()
    rng = np.random.default_rng(20261023)
    t1 = rng.normal(scale=0.02, size=zeros[0].shape)
    t2 = rng.normal(scale=0.02, size=zeros[1].shape)
    t2 = t2 - t2.swapaxes(0, 1)
    t2 = t2 - t2.swapaxes(2, 3)
    updates = {
        "previous": baseline.update_amps,
        "graph": generated.update_amps,
        "pyscf": lambda a, b: reference.update_amps(a, b, eris),
    }
    expected = updates["pyscf"](t1, t2)
    checks = {}
    for name, solver in [("previous", baseline), ("graph", generated)]:
        values = updates[name](t1, t2)
        checks[name] = dict(
            random_t1_max_abs=float(np.max(np.abs(values[0] - expected[0]))),
            random_t2_max_abs=float(np.max(np.abs(values[1] - expected[1]))),
            random_energy_abs=abs(
                float(solver.energy(t1, t2)) - reference.energy(t1, t2, eris)
            ),
        )
        assert max(checks[name].values()) < 1e-10, checks
    print("[benchmark] Random-amplitude checks passed", flush=True)
    result["updates"] = {
        name: timed_updates(fn, t1, t2, args.repeats) for name, fn in updates.items()
    }
    drivers = {
        "pyscf": reference,
        "previous": GeneratedDriver(mf, baseline),
        "graph": GeneratedDriver(mf, generated),
    }
    solutions = {}
    for name, driver in drivers.items():
        print(f"[benchmark] Converging {name}", flush=True)
        summary, a, b = solve(driver, eris, zeros)
        result[name] = summary
        solutions[name] = (a, b)
    for name in ("previous", "graph"):
        checks[name].update(
            energy_abs=abs(result[name]["e_corr"] - result["pyscf"]["e_corr"]),
            converged_t1_max_abs=float(
                np.max(np.abs(solutions[name][0] - solutions["pyscf"][0]))
            ),
            converged_t2_max_abs=float(
                np.max(np.abs(solutions[name][1] - solutions["pyscf"][1]))
            ),
        )
        assert checks[name]["energy_abs"] < 1e-9, checks
        assert checks[name]["converged_t1_max_abs"] < 1e-7, checks
        assert checks[name]["converged_t2_max_abs"] < 1e-7, checks
    eps = np.diag(generated.h1e)
    d1 = eps[:no, None] - eps[None, no:]
    d2 = d1[:, None, :, None] + d1[None, :, None, :]
    final = generated.update_amps(*solutions["graph"])
    checks["graph"]["final_residual_max_abs"] = max(
        float(np.max(np.abs((final[0] - solutions["graph"][0]) * d1))),
        float(np.max(np.abs((final[1] - solutions["graph"][1]) * d2))),
    )
    assert checks["graph"]["final_residual_max_abs"] < 1e-7, checks
    result["correctness"] = checks
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", default="6-31g")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", default=str(HERE / "benchmark_ccsd.json"))
    run(parser.parse_args())
