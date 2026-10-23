from dataclasses import dataclass
from fractions import Fraction
from time import perf_counter

import igraph


@dataclass(frozen=True)
class SlotGroup:
    slots: tuple
    antisymmetric: bool = True


@dataclass(frozen=True)
class TensorSymmetry:
    groups: tuple = ()

    def validate(self, rank):
        slots = [slot for group in self.groups for slot in group.slots]
        if len(set(slots)) != len(slots) or any(s < 0 or s >= rank for s in slots):
            raise ValueError("Slot groups must be disjoint and within tensor rank")
        if any(not group.slots for group in self.groups):
            raise ValueError("Empty slot group")


def cluster_symmetries(max_rank=4):
    """t[i1...in,a1...an] is antisymmetric in each half, for any n."""
    return {
        ("t", 2 * n): TensorSymmetry(
            (SlotGroup(tuple(range(n))), SlotGroup(tuple(range(n, 2 * n))))
        )
        for n in range(1, max_rank + 1)
    }


def parity(sequence):
    """Sign of the ordering of distinct integer labels."""
    return (
        -1
        if sum(a > b for i, a in enumerate(sequence) for b in sequence[i + 1 :]) % 2
        else 1
    )


class GraphCanonicalizer:
    def __init__(self, symmetries=None):
        # No symmetry is assumed unless explicitly registered.
        self.symmetries = dict(symmetries or {})
        self.calls = 0
        self.zero_terms = 0
        self.seconds = 0.0

    def canonicalize(self, term):
        start = perf_counter()
        try:
            return self._canonicalize(term)
        finally:
            self.calls += 1
            self.seconds += perf_counter() - start

    def _canonicalize(self, term):
        colors, edges, index_nodes = [], [], {}
        antisymmetric_groups = []
        slot_positions = {}

        def node(color):
            colors.append(color)
            return len(colors) - 1

        used = {idx for tensor in term.tensors for idx in tensor.indices}
        if not term.contractions <= used:
            raise ValueError("Eliminate unused summed indices before canonicalization")
        for tensor in term.tensors:
            # TensorTypes.Tensor=5, KroneckerDelta=4; operators do not commute.
            if int(tensor.type) not in (4, 5):
                raise ValueError(
                    "Canonicalization requires a fully contracted tensor product"
                )
            rank = len(tensor.indices)
            symmetry = self.symmetries.get((tensor.name, rank), TensorSymmetry())
            symmetry.validate(rank)
            descriptor = tuple(
                (group.slots, group.antisymmetric) for group in symmetry.groups
            )
            root = node(("tensor", tensor.name, int(tensor.type), rank, descriptor))
            membership = {
                s: group_id
                for group_id, group in enumerate(symmetry.groups)
                for s in group.slots
            }
            slots = []
            for position, idx in enumerate(tensor.indices):
                if idx not in index_nodes:
                    color = (
                        ("index", int(idx.space), "dummy", "")
                        if idx in term.contractions
                        else ("index", int(idx.space), "free", idx.label)
                    )
                    index_nodes[idx] = node(color)
                slot_color = (
                    ("slot", "group", membership[position])
                    if position in membership
                    else ("slot", "fixed", position)
                )
                slot = node(slot_color)
                slots.append(slot)
                edges.extend(((root, slot), (slot, index_nodes[idx])))
            for group in symmetry.groups:
                if group.antisymmetric and len(group.slots) > 1:
                    group_nodes = tuple(slots[s] for s in group.slots)
                    antisymmetric_groups.append(group_nodes)
                    for position, vertex in enumerate(group_nodes):
                        slot_positions[vertex] = position

        palette = {color: i for i, color in enumerate(sorted(set(colors)))}
        color_ids = [palette[c] for c in colors]
        graph = igraph.Graph(n=len(colors), edges=edges, directed=False)
        graph.vs["original_vertex"] = list(range(len(colors)))
        canonical_graph = graph.permute_vertices(
            graph.canonical_permutation(color=color_ids)
        )
        # Read the actual relabeling through attributes. In igraph 1.0 the
        # permute_vertices vector is new->old (older versions used old->new).
        permutation = [0] * len(colors)
        for new, old in enumerate(canonical_graph.vs["original_vertex"]):
            permutation[old] = new

        # A sign-reversing automorphism means term = -term, hence zero.
        # Check generators only: phase is a group homomorphism to {+1, -1}.
        if antisymmetric_groups:
            for automorphism in graph.automorphism_group(color=color_ids):
                phase = 1
                for group in antisymmetric_groups:
                    phase *= parity([slot_positions[automorphism[v]] for v in group])
                if phase == -1:
                    self.zero_terms += 1
                    return (), 0

        phase = 1
        for group in antisymmetric_groups:
            phase *= parity([permutation[v] for v in group])
        canonical_colors = [None] * len(colors)
        for old, new in enumerate(permutation):
            canonical_colors[new] = colors[old]
        canonical_edges = tuple(
            sorted(tuple(sorted((permutation[a], permutation[b]))) for a, b in edges)
        )
        # Full exact encoding, not a potentially colliding graph hash.
        return (tuple(canonical_colors), canonical_edges), phase


class TermAccumulator:
    """Streaming signed collection, retaining one original representative/key."""

    def __init__(self, symmetries=None):
        self.canonicalizer = GraphCanonicalizer(symmetries)
        self.coefficients = {}
        self.representatives = {}
        self.input_terms = 0

    def add(self, term):
        self.input_terms += 1
        if not term.factor:
            return
        key, phase = self.canonicalizer.canonicalize(term)
        if not phase:
            return
        self.coefficients[key] = self.coefficients.get(key, 0) + phase * term.factor
        if key not in self.representatives:
            self.representatives[key] = (term, phase)

    def expression(self, tolerance=1e-12):
        from wick import WickExpr, WickString

        terms = []
        for key, coefficient in self.coefficients.items():
            # Fraction inputs cancel exactly. Float inputs retain the prior tolerance.
            keep = (
                coefficient != 0
                if isinstance(coefficient, Fraction)
                else abs(coefficient) > tolerance
            )
            if keep:
                representative, phase = self.representatives[key]
                terms.append(
                    WickString(
                        representative.tensors,
                        representative.contractions,
                        coefficient * phase,
                    )
                )
        return WickExpr(terms)
