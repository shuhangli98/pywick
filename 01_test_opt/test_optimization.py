import os

os.environ["WICK_PROFILE"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
from collections import Counter
import importlib.util
from pathlib import Path
import random

import numpy as np
import pytest
import wick
from ccsd import GCCSD, generate_equations
from pyscf import gto, scf, lib
from pyscf.cc import gccsd

spec = importlib.util.spec_from_file_location(
    "archived_wick", Path(__file__).resolve().parent.parent / "00_old_slow/wick.py"
)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
SPACES = {wick.WickSpaces.Occupied: "ijpq", wick.WickSpaces.Virtual: "abpq"}
OLD_SPACES = {old.WickSpaces(int(space)): labels for space, labels in SPACES.items()}


def collect(expr):
    """Compare raw terms modulo delta-factor ordering, keeping operator order."""
    out = Counter()
    for term in expr.terms:
        scalars, operators = [], []
        for tensor in term.tensors:
            indices = tuple((int(idx.space), idx.label) for idx in tensor.indices)
            if int(tensor.type) == int(wick.TensorTypes.KroneckerDelta):
                indices = tuple(sorted(indices))
            item = (int(tensor.type), tensor.name, indices)
            (operators if int(tensor.type) in (1, 2, 3) else scalars).append(item)
        key = (
            tuple(sorted(scalars)),
            tuple(operators),
            tuple(sorted((int(idx.space), idx.label) for idx in term.contractions)),
        )
        out[key] += term.factor
    return dict(out)


def test_full_contractions_match_archive():
    rng = random.Random(804)
    cases = [
        "1",
        "C[i]",
        "C[i] D[j]",
        "D[a] C[b]",
        "C[a] D[b]",
        "C[i] C[j] D[i] D[j]",
        "C[i] C[j] D[j] D[i]",
        "C[i] D[a] C[b] D[j]",
        "SUM <pq> h[pq] C[p] D[q]",
        "SUM <ijab> h[ijab] C[i] C[j] D[b] D[a] C[a] C[b] D[j] D[i]",
    ]
    for _ in range(300):
        cases.append(
            "0.75 "
            + " ".join(
                f'{rng.choice("CD")}[{rng.choice("ijab")}]'
                for _ in range(rng.randrange(13))
            )
        )
    for text in cases:
        expected = old.WickExpr.parse(text, OLD_SPACES).wick_theorem(0)
        actual = wick.WickExpr.parse(text, SPACES).wick_theorem(0)
        assert collect(actual) == collect(expected), text


@pytest.mark.parametrize(
    "max_unctr,no_ctr", [(-1, False), (2, False), (4, False), (-1, True), (0, True)]
)
def test_other_modes_match_archive(max_unctr, no_ctr):
    text = "SUM <pq> h[pq] C[p] D[q] C[i] D[a] C[b] D[j]"
    actual = wick.WickExpr.parse(text, SPACES).wick_theorem(max_unctr, no_ctr)
    expected = old.WickExpr.parse(text, OLD_SPACES).wick_theorem(max_unctr, no_ctr)
    assert collect(actual) == collect(expected)


def test_vacuum_expectations_against_occupation_bitstrings():
    rng = random.Random(951)
    labels = "ijab"
    # Apply the operator string right-to-left directly to |1100>.
    for _ in range(500):
        ops = [(rng.choice("CD"), rng.randrange(4)) for _ in range(rng.randrange(13))]
        state, value = 3, 1
        for kind, orbital in reversed(ops):
            occupied = bool(state & (1 << orbital))
            if occupied == (kind == "C"):
                value = 0
                break
            value *= (-1) ** ((state & ((1 << orbital) - 1)).bit_count())
            state ^= 1 << orbital
        expected = value if state == 3 else 0
        text = "1 " + " ".join(f"{kind}[{labels[orbital]}]" for kind, orbital in ops)
        terms = wick.WickExpr.parse(text, SPACES).wick_theorem(0).terms
        actual = sum(
            term.factor
            for term in terms
            if all(t.indices[0] == t.indices[1] for t in term.tensors)
        )
        assert actual == expected, text


@pytest.fixture(scope="module")
def full_equations():
    return generate_equations(verbose=False)


def test_quadratic_equations_match_archive():
    expected = generate_equations(engine=old, max_order=2, verbose=False)
    actual = generate_equations(max_order=2, verbose=False)
    # Evaluate both generated programs on the same tensors: dummy labels and
    # delta elimination order may differ without changing the tensor expression.
    rng = np.random.default_rng(31)
    from itertools import product

    ns = {
        "np": np,
        "tOV": rng.normal(size=(2, 3)),
        "tOOVV": rng.normal(size=(2, 2, 3, 3)),
    }
    for name, rank in [("h", 2), ("v", 4)]:
        for spaces in product("OV", repeat=rank):
            ns[name + "".join(spaces)] = rng.normal(
                size=tuple(2 if s == "O" else 3 for s in spaces)
            )
    for name, lhs, shape in [
        ("energy", "E", ()),
        ("singles", "t1new", (2, 3)),
        ("doubles", "t2new", (2, 2, 3, 3)),
    ]:
        assert len(getattr(expected, name).terms) == len(getattr(actual, name).terms)
        values = []
        for eq in (expected, actual):
            context = dict(ns)
            context[lhs] = np.zeros(shape)
            exec(eq.code[name], context)
            values.append(context[lhs])
        np.testing.assert_allclose(*values, atol=1e-11, rtol=1e-11)


def test_full_equations_against_pyscf(full_equations):
    lib.num_threads(1)
    mol = gto.M(atom="O 0 0 0; H 0 1 0; H 0 0 1", basis="sto-3g", verbose=0)
    mf = scf.GHF(mol).run(conv_tol=1e-13)
    assert mf.converged
    generated = GCCSD(mf, equations=full_equations)
    reference = gccsd.GCCSD(mf)
    eris = reference.ao2mo()
    rng = np.random.default_rng(41)
    for scale in (0, 0.02, 0.1):
        t1, t2 = generated.init_amps()
        t1[:] = rng.normal(scale=scale, size=t1.shape)
        t2[:] = rng.normal(scale=scale, size=t2.shape)
        t2 = t2 - t2.swapaxes(0, 1)
        t2 = t2 - t2.swapaxes(2, 3)
        actual = generated.update_amps(t1, t2)
        expected = reference.update_amps(t1, t2, eris)
        np.testing.assert_allclose(actual[0], expected[0], atol=1e-11, rtol=1e-11)
        np.testing.assert_allclose(actual[1], expected[1], atol=1e-11, rtol=1e-11)
        assert abs(generated.energy(t1, t2) - reference.energy(t1, t2, eris)) < 1e-11
    assert full_equations.max_order == 4
    # Nonzero T1 makes the restored third/fourth-order contributions observable.
    truncated = GCCSD(mf, equations=generate_equations(max_order=2, verbose=False))
    low = truncated.update_amps(t1, t2)
    assert np.max(np.abs(actual[1] - low[1])) > 1e-6
