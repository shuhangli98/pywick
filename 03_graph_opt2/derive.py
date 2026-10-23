from fractions import Fraction
from itertools import combinations_with_replacement, product
from collections import Counter
from math import factorial
from time import perf_counter

from wick import WickIndex, WickTensor, WickString, WickExpr, WickSpaces
from graph_canonical import TermAccumulator, cluster_symmetries
from delta_reduce import reduce_disjoint_deltas
from orbit_contract import iter_orbit_contractions


def model(max_rank):
    if max_rank not in (2, 3, 4):
        raise ValueError("Supported benchmarks: CCSD, CCSDT, CCSDTQ")
    spaces = {WickSpaces.Occupied: "pqrsijkl", WickSpaces.Virtual: "pqrsabcd"}
    parse = lambda text: WickExpr.parse(text, spaces)
    h = (
        parse("SUM <pq> h[pq] C[p] D[q]")
        + 0.5 * parse("SUM <pqrs> v[pqrs] C[p] C[q] D[s] D[r]")
    ).wick_theorem(no_ctr=True)
    for term in h.terms:
        term.factor = Fraction(term.factor)
    clusters = {}
    projectors = {0: WickString([], set(), Fraction(1))}
    for rank in range(1, max_rank + 1):
        occupied = [WickIndex(label, WickSpaces.Occupied) for label in "ijkl"[:rank]]
        virtual = [WickIndex(label, WickSpaces.Virtual) for label in "abcd"[:rank]]
        amplitude = WickTensor("t", occupied + virtual)
        excitation = [WickTensor.cre([idx]) for idx in virtual] + [
            WickTensor.ann([idx]) for idx in reversed(occupied)
        ]
        clusters[rank] = WickString(
            [amplitude] + excitation,
            set(occupied + virtual),
            Fraction(1, factorial(rank) ** 2),
        )
        deexcitation = [WickTensor.cre([idx]) for idx in occupied] + [
            WickTensor.ann([idx]) for idx in reversed(virtual)
        ]
        projectors[rank] = WickString(deexcitation, set(), Fraction(1))
    return h, clusters, projectors


def products(max_rank, projection_rank, max_order=4, unordered=False):
    """Generate only rank-feasible products. Optional commuting-T multinomials.

    Sum of excitation ranks cannot exceed projection rank + 2: H supplies
    at most four quasiparticle annihilators. The bound only removes zeros.
    """
    h, clusters, projectors = model(max_rank)
    for order in range(max_order + 1):
        sequences = (
            combinations_with_replacement(range(1, max_rank + 1), order)
            if unordered
            else product(range(1, max_rank + 1), repeat=order)
        )
        for ranks in sequences:
            if sum(ranks) > projection_rank + 2:
                continue
            denominator = 1 if unordered else factorial(order)
            if unordered:
                for count in Counter(ranks).values():
                    denominator *= factorial(count)
            for hterm in h.terms:
                term = projectors[projection_rank] * hterm
                for rank in ranks:
                    term = term * clusters[rank]
                term.factor /= denominator
                yield order, ranks, term


def derive_equation(
    max_rank,
    projection_rank,
    *,
    max_order=4,
    unordered=False,
    progress=None,
    deadline=None,
    delta_mode="map",
    generation="orbit"
):
    if generation not in {"orbit", "slots", "labelled"}:
        raise ValueError("generation must be orbit, slots or labelled")
    if delta_mode not in {"copy", "map"}:
        raise ValueError("delta_mode must be copy or map")
    start = perf_counter()
    accumulator = TermAccumulator(cluster_symmetries(max_rank))
    stats = dict(
        rank=projection_rank,
        products=0,
        represented_contractions=0,
        connected_terms=0,
        generated_representatives=0,
        merge_calls=0,
        generation=generation,
        merged_terms=0,
        contraction_seconds=0.0,
        simplify_seconds=0.0,
        merge_seconds=0.0,
    )
    next_report = start + 10
    for order, ranks, term in products(max_rank, projection_rank, max_order, unordered):
        stats["products"] += 1
        iterator = (
            iter_orbit_contractions(term, exchange_clusters=generation == "orbit")
            if generation != "labelled"
            else ((t, 1) for t in term.iter_full_contractions())
        )
        while True:
            tick = perf_counter()
            if deadline is not None and tick > deadline:
                stats["seconds"] = tick - start
                stats["merged_terms"] = len(accumulator.expression().terms)
                if progress:
                    progress(dict(stats, status="timeout"))
                raise TimeoutError(stats)
            try:
                contracted, multiplicity = next(iterator)
            except StopIteration:
                stats["contraction_seconds"] += perf_counter() - tick
                break
            stats["contraction_seconds"] += perf_counter() - tick
            stats["represented_contractions"] += multiplicity
            stats["generated_representatives"] += 1
            tick = perf_counter()
            reduced = (
                contracted.simplify_delta()
                if delta_mode == "copy"
                else reduce_disjoint_deltas(contracted)
            )
            connected = reduced.factor and reduced.is_connected()
            stats["simplify_seconds"] += perf_counter() - tick
            if connected:
                stats["connected_terms"] += multiplicity
                stats["merge_calls"] += 1
                tick = perf_counter()
                accumulator.add(reduced)
                stats["merge_seconds"] += perf_counter() - tick
            now = perf_counter()
            if progress and now >= next_report:
                progress(
                    dict(
                        stats,
                        seconds=now - start,
                        order=order,
                        cluster_ranks=ranks,
                        accumulated_keys=len(accumulator.coefficients),
                        status="running",
                    )
                )
                next_report = now + 10
    result = accumulator.expression()
    stats.update(
        merged_terms=len(result.terms),
        seconds=perf_counter() - start,
        status="completed",
        zero_by_symmetry=accumulator.canonicalizer.zero_terms,
    )
    if progress:
        progress(stats)
    return result, stats
