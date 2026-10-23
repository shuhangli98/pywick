import os

os.environ.setdefault("WICK_PROFILE", "0")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/pywick-mpl")
from fractions import Fraction
from itertools import permutations, product
import random

import numpy as np
import pytest
from wick import WickIndex, WickSpaces, WickTensor, WickString, WickExpr
from graph_canonical import (
    GraphCanonicalizer,
    SlotGroup,
    TensorSymmetry,
    TermAccumulator,
    cluster_symmetries,
    parity,
)

O, V = WickSpaces.Occupied, WickSpaces.Virtual
SPACES = {O: "ijklmn", V: "abcdef"}
P = lambda s: WickString.parse(s, SPACES)


def test_dummy_renaming_factor_order_and_fixed_externals():
    c = GraphCanonicalizer(cluster_symmetries())
    a, sa = c.canonicalize(P("SUM <jb> h[jb] t[ijab]"))
    b, sb = c.canonicalize(P("SUM <kc> t[ikac] h[kc]"))
    assert (a, sa) == (b, sb)
    b, sb = c.canonicalize(P("SUM <kc> h[kc] t[kiac]"))
    assert a == b and sa == -sb
    assert c.canonicalize(P("h[ia]"))[0] != c.canonicalize(P("h[ja]"))[0]
    assert c.canonicalize(P("h[ia]"))[0] != c.canonicalize(P("h[ai]"))[0]


def test_signed_automorphism_zero():
    sym = {
        ("A", 2): TensorSymmetry((SlotGroup((0, 1)),)),
        ("S", 2): TensorSymmetry((SlotGroup((0, 1), False),)),
    }
    c = GraphCanonicalizer(sym)
    assert c.canonicalize(P("SUM <ij> A[ij] S[ij]"))[1] == 0
    assert c.canonicalize(P("SUM <i> A[ii]"))[1] == 0
    # Two antisymmetric tensors have cancelling signs, so their scalar product is not zero.
    sym[("B", 2)] = sym[("A", 2)]
    assert GraphCanonicalizer(sym).canonicalize(P("SUM <ij> A[ij] B[ij]"))[1] != 0


@pytest.mark.parametrize("rank", [2, 3, 4])
def test_high_rank_symmetry_and_renaming(rank):
    rng = random.Random(810 + rank)
    occupied = [WickIndex(chr(105 + k), O) for k in range(rank)]
    virtual = [WickIndex(chr(97 + k), V) for k in range(rank)]
    indices = occupied + virtual
    # An ordered anchor distinguishes all slots; no sign-reversing automorphism.
    base = WickString(
        [WickTensor("anchor", indices), WickTensor("t", indices)], set(indices), 1
    )
    c = GraphCanonicalizer(cluster_symmetries(rank))
    key, sign = c.canonicalize(base)
    for _ in range(40):
        po = list(range(rank))
        pv = list(range(rank))
        rng.shuffle(po)
        rng.shuffle(pv)
        remap = {
            idx: WickIndex(f"x{rng.randrange(1000000)}", idx.space) for idx in indices
        }
        permuted = [occupied[k] for k in po] + [virtual[k] for k in pv]
        tensors = [
            WickTensor("t", [remap[x] for x in permuted]),
            WickTensor("anchor", [remap[x] for x in indices]),
        ]
        trial = WickString(tensors, set(remap.values()), 1)
        k, s = c.canonicalize(trial)
        assert k == key and s == sign * parity(po) * parity(pv)


def brute_key(term):
    """Small exact oracle: enumerate factor order and T2 slot permutations."""
    options = []
    for tensor in term.tensors:
        idx = tuple(tensor.indices)
        if tensor.name == "t" and len(idx) == 4:
            options.append(
                [
                    (tensor.name, tuple(idx[i] for i in p), s)
                    for p, s in (
                        ((0, 1, 2, 3), 1),
                        ((1, 0, 2, 3), -1),
                        ((0, 1, 3, 2), -1),
                        ((1, 0, 3, 2), 1),
                    )
                ]
            )
        else:
            options.append([(tensor.name, idx, 1)])
    best = None
    signs = set()
    for variant in product(*options):
        phase = np.prod([x[2] for x in variant])
        for factors in permutations(variant):
            renaming = {}
            counts = {}
            key = []
            for name, indices, _ in factors:
                new = []
                for idx in indices:
                    space = int(idx.space)
                    if idx in term.contractions:
                        if idx not in renaming:
                            renaming[idx] = counts.get(space, 0)
                            counts[space] = renaming[idx] + 1
                        new.append((space, "D", renaming[idx]))
                    else:
                        new.append((space, "F", idx.label))
                key.append((name, tuple(new)))
            key = tuple(key)
            if best is None or key < best:
                best = key
                signs = {phase}
            elif key == best:
                signs.add(phase)
    return best, (int(next(iter(signs))) if len(signs) == 1 else 0)


def test_partition_and_signs_against_small_enumeration():
    from ccsd import generate_equations

    equations = generate_equations(merge=False, verbose=False)
    graph_to_brute = {}
    brute_to_graph = {}
    relative = {}
    c = GraphCanonicalizer(cluster_symmetries(2))
    for expression in (equations.energy, equations.singles, equations.doubles):
        for term in expression.terms:
            g, gs = c.canonicalize(term)
            b, bs = brute_key(term)
            assert (gs == 0) == (bs == 0)
            if not gs:
                continue
            assert graph_to_brute.setdefault(g, b) == b
            assert brute_to_graph.setdefault(b, g) == g
            assert relative.setdefault(g, gs * bs) == gs * bs
    assert (
        len(graph_to_brute) == 238
    )  # 4 + 33 + 201 before zero coefficient removal (same here)


def test_exact_coefficient_collection():
    a = P("SUM <jb> h[jb] t[ijab]")
    a.factor = Fraction(1, 3)
    b = P("SUM <kc> t[kiac] h[kc]")
    b.factor = Fraction(1, 3)
    acc = TermAccumulator(cluster_symmetries())
    acc.add(a)
    acc.add(b)
    assert acc.expression().terms == []


def test_reject_noncommuting_operators():
    with pytest.raises(ValueError, match="fully contracted"):
        GraphCanonicalizer().canonicalize(P("C[i] D[j]"))


def test_streamed_ccsd_matches_existing_pipeline():
    from ccsd import generate_equations
    from derive import derive_equation

    expected = generate_equations(verbose=False)

    def collect(expr):
        c = GraphCanonicalizer(cluster_symmetries())
        return {
            c.canonicalize(t)[0]: t.factor * c.canonicalize(t)[1] for t in expr.terms
        }

    for rank, name in enumerate(("energy", "singles", "doubles")):
        actual, stats = derive_equation(2, rank)
        a, b = collect(actual), collect(getattr(expected, name))
        assert a.keys() == b.keys()
        for key in a:
            assert abs(float(a[key]) - b[key]) < 1e-12
        assert stats["connected_terms"] == expected.stats[name]["raw_terms"]
        compact, _ = derive_equation(2, rank, unordered=True)
        d = collect(compact)
        assert d.keys() == a.keys()
        for key in a:
            assert a[key] == d[key]


def test_disjoint_delta_map_matches_copy_and_fallback():
    from delta_reduce import reduce_disjoint_deltas
    from derive import products

    for rank in (2, 3, 4):
        checked = 0
        for _, _, term in products(rank, rank, unordered=True):
            for raw in term.iter_full_contractions():
                expected = raw.simplify_delta()
                actual = reduce_disjoint_deltas(raw)
                assert actual == expected
                checked += 1
                if checked >= 250:
                    break
            if checked >= 250:
                break
        assert checked == 250
    for text in (
        "SUM <ij> h[ij] delta[ij] delta[ji]",
        "delta[ij] h[ia]",
        "SUM <i> h[ia] delta[ij]",
    ):
        term = P(text)
        assert reduce_disjoint_deltas(term) == term.simplify_delta()


def test_scalar_and_operator_preservation():
    scalar = WickExpr(
        [WickString([], set(), Fraction(2, 3)), WickString([], set(), Fraction(1, 3))]
    ).combine_like_terms()
    assert len(scalar.terms) == 1 and scalar.terms[0].factor == 1
    assert scalar.terms[0].tensors == []
    # The non-full-contraction code path still retains the original operator algebra.
    expression = WickExpr.parse("C[i] D[j]", SPACES)
    assert any(
        t.type.value in (1, 2)
        for term in expression.wick_theorem().terms
        for t in term.tensors
    )


def test_high_rank_models_reduce_to_ccsd_when_t3_t4_are_zero():
    from ccsd import generate_equations
    from derive import derive_equation

    reference = generate_equations(verbose=False)
    c = GraphCanonicalizer(cluster_symmetries(4))

    def coefficients(expression):
        result = {}
        for term in expression.terms:
            if any(t.name == "t" and len(t.indices) > 4 for t in term.tensors):
                continue
            key, phase = c.canonicalize(term)
            result[key] = result.get(key, 0) + phase * term.factor
        return {k: v for k, v in result.items() if abs(v) > 1e-12}

    for max_rank in (3, 4):
        for projection, name in enumerate(("energy", "singles", "doubles")):
            high, _ = derive_equation(
                max_rank, projection, unordered=True, delta_mode="map"
            )
            actual, expected = coefficients(high), coefficients(
                getattr(reference, name)
            )
            assert actual.keys() == expected.keys()
            for key in actual:
                assert abs(float(actual[key]) - expected[key]) < 1e-12
