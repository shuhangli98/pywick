import os

os.environ.setdefault("WICK_PROFILE", "0")
import argparse
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

# Set BLAS/OpenMP limits before importing any numerical libraries.
_thread_parser = argparse.ArgumentParser(add_help=False)
_thread_parser.add_argument("--threads", type=int, default=1)
_thread_options, _ = _thread_parser.parse_known_args()
for _variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ[_variable] = str(_thread_options.threads)

import numpy as np
import pyscf
from pyscf import gto, scf, lib
from pyscf.cc import gccsd
from ccsd import GCCSD, generate_equations

HERE = Path(__file__).resolve().parent


def legacy_engine():
    spec = importlib.util.spec_from_file_location(
        "old_wick", HERE.parent / "00_old_slow/wick.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def legacy_timing(order, timeout):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--legacy-worker",
        str(order),
    ]
    start = perf_counter()
    try:
        run = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=dict(os.environ, WICK_PROFILE="0"),
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "timeout_seconds": timeout,
            "wall_seconds": perf_counter() - start,
        }
    if run.returncode:
        raise RuntimeError(run.stderr)
    return json.loads(run.stdout)


class GeneratedDriver(gccsd.GCCSD):
    """Use PySCF's convergence/DIIS driver with independently generated equations."""

    def __init__(self, mf, generated):
        super().__init__(mf)
        self.generated = generated
        self._keys = self._keys | {"generated"}

    def update_amps(self, t1, t2, eris):
        return self.generated.update_amps(t1, t2)

    def energy(self, t1, t2, eris):
        return float(self.generated.energy(t1, t2))


def timed_updates(fn, t1, t2, repeats):
    fn(t1, t2)  # untimed warm-up
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        fn(t1, t2)
        samples.append(perf_counter() - start)
    return {"median_seconds": float(np.median(samples)), "samples_seconds": samples}


def solve(driver, eris, zeros):
    steps = []
    driver.callback = lambda env: steps.append(env["istep"])
    start = perf_counter()
    ecorr, t1, t2 = driver.kernel(t1=zeros[0].copy(), t2=zeros[1].copy(), eris=eris)
    elapsed = perf_counter() - start
    return (
        {
            "seconds": elapsed,
            "cycles": len(steps),
            "converged": bool(driver.converged),
            "e_corr": float(ecorr),
            "e_tot": float(driver.e_tot),
        },
        t1,
        t2,
    )


def run(args):
    lib.num_threads(args.threads)
    result = {
        "environment": {
            "python": platform.python_version(),
            "pyscf": pyscf.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "threads": args.threads,
        },
        "settings": {
            "atom": "O 0 0 0; H 0 1 0; H 0 0 1",
            "basis": args.basis,
            "reference": "GHF / GCCSD",
            "initial_amplitudes": "zero",
            "diis": True,
            "diis_space": 6,
            "conv_tol": 1e-10,
            "conv_tol_normt": 1e-8,
            "max_cycle": 100,
            "random_seed": 20261023,
        },
    }
    equations = generate_equations()
    quadratic = generate_equations(max_order=2, verbose=False)
    result["symbolic"] = {
        "optimized_order4_seconds": equations.seconds,
        "optimized_order2_seconds": quadratic.seconds,
        "order4_terms": {
            name: len(getattr(equations, name).terms)
            for name in ("energy", "singles", "doubles")
        },
    }
    print("[benchmark] Preparing H2O /", args.basis, flush=True)
    mol = gto.M(atom=result["settings"]["atom"], basis=args.basis, verbose=0)
    start = perf_counter()
    mf = scf.GHF(mol).run(conv_tol=1e-14)
    if not mf.converged:
        raise RuntimeError("GHF did not converge")
    result["scf_seconds"] = perf_counter() - start
    result["e_hf"] = float(mf.e_tot)
    reference = gccsd.GCCSD(mf)
    start = perf_counter()
    eris = reference.ao2mo()
    result["pyscf_ao2mo_seconds"] = perf_counter() - start
    start = perf_counter()
    generated = GCCSD(mf, equations=equations)
    result["generated_ao2mo_seconds"] = perf_counter() - start
    result["dimensions"] = {"nocc": generated.nocc, "nvir": generated.nvir}
    no = generated.nocc
    anti = generated.g2e - generated.g2e.swapaxes(2, 3)
    checks = {
        "fock_max_abs": float(np.max(np.abs(generated.h1e - eris.fock))),
        "oovv_max_abs": float(np.max(np.abs(anti[:no, :no, no:, no:] - eris.oovv))),
        "reference_energy_abs": abs(generated.ecore - mf.e_tot),
    }
    zeros = generated.init_amps()
    rng = np.random.default_rng(20261023)
    t1 = rng.normal(scale=0.02, size=zeros[0].shape)
    t2 = rng.normal(scale=0.02, size=zeros[1].shape)
    t2 = t2 - t2.swapaxes(0, 1)
    t2 = t2 - t2.swapaxes(2, 3)
    gupdate = generated.update_amps(t1, t2)
    pupdate = reference.update_amps(t1, t2, eris)
    checks["random_update_t1_max_abs"] = float(np.max(np.abs(gupdate[0] - pupdate[0])))
    checks["random_update_t2_max_abs"] = float(np.max(np.abs(gupdate[1] - pupdate[1])))
    checks["random_energy_abs"] = abs(
        float(generated.energy(t1, t2)) - reference.energy(t1, t2, eris)
    )
    eps = np.diag(generated.h1e)
    d1 = eps[:no, None] - eps[None, no:]
    d2 = d1[:, None, :, None] + d1[None, :, None, :]
    checks["random_residual_t1_max_abs"] = float(
        np.max(np.abs((gupdate[0] - pupdate[0]) * d1))
    )
    checks["random_residual_t2_max_abs"] = float(
        np.max(np.abs((gupdate[1] - pupdate[1]) * d2))
    )
    if max(checks.values()) > 1e-10:
        raise AssertionError(checks)
    print("[benchmark] Random-amplitude checks:", checks, flush=True)
    result["update_timing"] = {
        "generated": timed_updates(generated.update_amps, t1, t2, args.repeats),
        "pyscf": timed_updates(
            lambda a, b: reference.update_amps(a, b, eris), t1, t2, args.repeats
        ),
    }
    optimized = GeneratedDriver(mf, generated)
    for driver in (reference, optimized):
        driver.verbose = 0
        driver.conv_tol = 1e-10
        driver.conv_tol_normt = 1e-8
        driver.max_cycle = 100
        driver.diis = True
        driver.diis_space = 6
    print("[benchmark] Running PySCF GCCSD iterations", flush=True)
    result["pyscf"], pt1, pt2 = solve(reference, eris, zeros)
    print("[benchmark] Running generated GCCSD iterations", flush=True)
    result["generated"], gt1, gt2 = solve(optimized, eris, zeros)
    checks["converged_energy_abs"] = abs(
        result["pyscf"]["e_corr"] - result["generated"]["e_corr"]
    )
    checks["converged_t1_max_abs"] = float(np.max(np.abs(pt1 - gt1)))
    checks["converged_t2_max_abs"] = float(np.max(np.abs(pt2 - gt2)))
    final_update = generated.update_amps(gt1, gt2)
    checks["final_residual_t1_max_abs"] = float(
        np.max(np.abs((final_update[0] - gt1) * d1))
    )
    checks["final_residual_t2_max_abs"] = float(
        np.max(np.abs((final_update[1] - gt2) * d2))
    )
    result["correctness"] = checks
    assert result["pyscf"]["converged"] and result["generated"]["converged"], result
    assert checks["converged_energy_abs"] < 1e-9, checks
    assert (
        max(checks["converged_t1_max_abs"], checks["converged_t2_max_abs"]) < 1e-7
    ), checks
    assert (
        max(checks["final_residual_t1_max_abs"], checks["final_residual_t2_max_abs"])
        < 1e-7
    ), checks
    if not args.skip_legacy:
        for order in (2, 4):
            print(
                f"[benchmark] Archived algorithm, order {order}, timeout {args.legacy_timeout}s",
                flush=True,
            )
            result["symbolic"][f"legacy_order{order}"] = legacy_timing(
                order, args.legacy_timeout
            )
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", default="6-31g")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--legacy-timeout", type=float, default=60)
    parser.add_argument("--skip-legacy", action="store_true")
    parser.add_argument("--output")
    parser.add_argument(
        "--legacy-worker", type=int, choices=(2, 4), help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if args.legacy_worker:
        eq = generate_equations(
            engine=legacy_engine(), max_order=args.legacy_worker, verbose=False
        )
        print(
            json.dumps(
                {
                    "status": "completed",
                    "seconds": eq.seconds,
                    "terms": [
                        len(eq.energy.terms),
                        len(eq.singles.terms),
                        len(eq.doubles.terms),
                    ],
                }
            )
        )
    else:
        run(args)
