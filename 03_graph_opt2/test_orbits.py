import os

os.environ["WICK_PROFILE"] = "0"
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/pywick-mpl")

from math import factorial
from collections import Counter
from fractions import Fraction

import pytest

from derive import products, derive_equation
from delta_reduce import reduce_disjoint_deltas
from graph_canonical import TermAccumulator, cluster_symmetries
from orbit_contract import iter_orbit_contractions
from wick import WickTensor, TensorTypes


def coefficients(terms):
    accumulator = TermAccumulator(cluster_symmetries())
    for term in terms:
        accumulator.add(term)
    return {k: v for k, v in accumulator.coefficients.items() if v}


@pytest.mark.parametrize("projection", [0, 1, 2])
def test_every_ccsd_product_orbit_coefficients_and_cardinality(projection):
    for _, ranks, term in products(2, projection, unordered=True):
        raw = [reduce_disjoint_deltas(t) for t in term.iter_full_contractions()]
        expected = coefficients(raw)
        connected = [t for t in raw if t.is_connected()]
        for exchange in (False, True):
            orbit = list(
                iter_orbit_contractions(
                    term, exchange_clusters=exchange, connected_only=False
                )
            )
            assert sum(weight for _, weight in orbit) == len(raw)
            actual = coefficients(reduce_disjoint_deltas(t) for t, _ in orbit)
            assert actual == expected
            expected_weight = 1
            for rank in ranks:
                expected_weight *= factorial(rank) ** 2
            if exchange:
                for count in Counter(ranks).values():
                    expected_weight *= factorial(count)
            assert all(weight == expected_weight for _, weight in orbit)
            pruned = list(iter_orbit_contractions(term, exchange_clusters=exchange))
            assert sum(weight for _, weight in pruned) == len(connected)
            assert coefficients(
                reduce_disjoint_deltas(t) for t, _ in pruned
            ) == coefficients(connected)


@pytest.mark.parametrize("projection", [0, 1, 2, 3])
def test_ccsdt_exact_equations(projection):
    old, old_stats = derive_equation(
        3, projection, unordered=True, generation="labelled"
    )
    new, new_stats = derive_equation(3, projection, unordered=True)
    assert coefficients(old.terms) == coefficients(new.terms)
    assert old_stats["connected_terms"] == new_stats["connected_terms"]
    assert all(isinstance(t.factor, Fraction) for t in new.terms)


def test_t4_orbit_size_and_coefficient():
    # Single T4 product: manageable labelled oracle, both T4 slot groups act.
    checked = False
    for _, ranks, term in products(4, 4, unordered=True):
        if ranks != (4,):
            continue
        orbit = list(iter_orbit_contractions(term))
        if not orbit:
            continue
        raw = [reduce_disjoint_deltas(t) for t in term.iter_full_contractions()]
        raw = [t for t in raw if t.is_connected()]
        assert all(weight == factorial(4) ** 2 for _, weight in orbit)
        assert sum(weight for _, weight in orbit) == len(raw)
        assert coefficients(raw) == coefficients(
            reduce_disjoint_deltas(t) for t, _ in orbit
        )
        checked = True
        break
    assert checked


def test_reject_unsupported_operator_structure():
    term = next(t for _, ranks, t in products(2, 1, unordered=True) if ranks == (1,))
    tensor = next(t for t in term.tensors if t.name == "t")
    idx = tensor.indices[0]
    op = next(
        t
        for t in term.tensors
        if t.type == TensorTypes.Annihilation and t.indices == [idx]
    )
    op.type = TensorTypes.Creation
    with pytest.raises(ValueError, match="pure excitation"):
        list(iter_orbit_contractions(term))


def test_compiled_ccsd_matches_labelled_reference():
    import numpy as np
    from itertools import product
    from ccsd import generate_equations

    old = generate_equations(verbose=False, generation="labelled")
    new = generate_equations(verbose=False)
    rng = np.random.default_rng(20261023)
    dimensions = {"O": 3, "V": 4}
    inputs = {"np": np}
    for name, rank in (("h", 2), ("v", 4)):
        for spaces in product("OV", repeat=rank):
            inputs[name + "".join(spaces)] = rng.normal(
                size=tuple(dimensions[s] for s in spaces)
            )
    inputs["tOV"] = rng.normal(size=(3, 4))
    t2 = rng.normal(size=(3, 3, 4, 4))
    t2 = t2 - t2.swapaxes(0, 1)
    inputs["tOOVV"] = t2 - t2.swapaxes(2, 3)
    for name, target, shape in (
        ("energy", "E", ()),
        ("singles", "t1new", (3, 4)),
        ("doubles", "t2new", (3, 3, 4, 4)),
    ):
        actual, expected = dict(inputs), dict(inputs)
        actual[target], expected[target] = np.zeros(shape), np.zeros(shape)
        exec(old.code[name], {}, expected)
        exec(new.code[name], {}, actual)
        np.testing.assert_allclose(
            actual[target], expected[target], atol=1e-11, rtol=1e-12
        )
