from collections import Counter

from wick import WickString, WickTensor, TensorTypes, WickSpaces


def iter_orbit_contractions(term, *, exchange_clusters=True, connected_only=True):
    tensors = [t for t in term.tensors if t.type == TensorTypes.Tensor]
    ops = [
        t
        for t in term.tensors
        if t.type in (TensorTypes.Creation, TensorTypes.Annihilation)
    ]
    if len(tensors) + len(ops) != len(term.tensors):
        raise ValueError(
            "Orbit contraction requires ordinary tensors and C/D operators"
        )
    cores = [t for t in tensors if t.name in ("h", "v")]
    clusters = [t for t in tensors if t.name == "t"]
    if len(cores) != 1 or len(clusters) + 1 != len(tensors):
        raise ValueError(
            "Orbit contraction requires one h/v and pure excitation t factors"
        )
    if any(len(op.indices) != 1 for op in ops):
        raise ValueError("Each operator must have one index")
    indices = [op.indices[0] for op in ops]
    if len(set(indices)) != len(indices):
        raise ValueError("Operator indices must be distinct before contraction")
    if any(
        idx.space not in (WickSpaces.Occupied, WickSpaces.Virtual) for idx in indices
    ):
        raise ValueError("Split occupied/virtual spaces before orbit contraction")
    positions = {idx: i for i, idx in enumerate(indices)}
    scalar_indices = [idx for tensor in tensors for idx in tensor.indices]
    if (
        len(set(scalar_indices)) != len(scalar_indices)
        or set(scalar_indices) != term.contractions
        or not set(scalar_indices) <= positions.keys()
    ):
        raise ValueError("Each scalar slot must be a distinct dummy with one operator")

    masks, signatures, group_masks = [], [], []
    owner = [-1] * len(ops)
    for number, tensor in enumerate(clusters):
        rank = len(tensor.indices) // 2
        if not rank or len(tensor.indices) != 2 * rank:
            raise ValueError("Tn requires n occupied and n virtual slots")
        groups = (tensor.indices[:rank], tensor.indices[rank:])
        local_masks = []
        for group, space, op_type in zip(
            groups,
            (WickSpaces.Occupied, WickSpaces.Virtual),
            (TensorTypes.Annihilation, TensorTypes.Creation),
        ):
            if any(
                idx.space != space or ops[positions[idx]].type != op_type
                for idx in group
            ):
                raise ValueError(
                    "Only pure excitation amplitudes with occupied/virtual halves are supported"
                )
            mask = sum(1 << positions[idx] for idx in group)
            local_masks.append(mask)
            for idx in group:
                owner[positions[idx]] = number
        masks.append(local_masks[0] | local_masks[1])
        group_masks.extend(local_masks)
        signatures.append((tensor.name, rank))

    # Every T operator must lie after the projector and Hamiltonian operators.
    # Thus all slots in a group always have the same eligible left partners.
    if clusters:
        first_t = min(i for i, value in enumerate(owner) if value >= 0)
        if any(value < 0 for value in owner[first_t:]):
            raise ValueError(
                "Pure excitation factors must follow projector and Hamiltonian"
            )

    balance = Counter()
    for op in ops:
        balance[op.indices[0].space] += 1 if op.type == TensorTypes.Creation else -1
    if any(balance.values()) or len(ops) % 2:
        return
    partners = [0] * len(ops)
    for i, left in enumerate(ops):
        for j in range(i + 1, len(ops)):
            right = ops[j]
            if (
                left > right
                and left.type != right.type
                and indices[i].space == indices[j].space
            ):
                partners[i] |= 1 << j

    core_positions = {positions[idx] for idx in cores[0].indices}
    core_sources = sum(1 << i for i in core_positions if partners[i])
    source_targets = {i: partners[i] for i in core_positions if partners[i]}
    all_clusters = (1 << len(clusters)) - 1
    slot_group = [0] * len(ops)
    for mask in group_masks:
        for i in range(len(ops)):
            if mask & (1 << i):
                slot_group[i] = mask
    pairs = []

    def visit(remaining, sign, multiplicity, touched, linked):
        if connected_only:
            missing = all_clusters & ~linked
            sources = remaining & core_sources
            if missing.bit_count() > sources.bit_count():
                return
            targets = 0
            for i, candidates in source_targets.items():
                if sources & (1 << i):
                    targets |= candidates & remaining
            for cluster, mask in enumerate(masks):
                if missing & (1 << cluster) and not mask & targets:
                    return
        if not remaining:
            deltas = [
                WickTensor.kronecker_delta([indices[i], indices[j]]) for i, j in pairs
            ]
            yield WickString(
                tensors + deltas, term.contractions, sign * multiplicity * term.factor
            ), multiplicity
            return
        first = remaining & -remaining
        i = first.bit_length() - 1
        rest = remaining ^ first
        candidates = partners[i] & rest
        seen_unopened = set()
        while candidates:
            bit = candidates & -candidates
            j = bit.bit_length() - 1
            group = slot_group[j]
            alternatives = candidates & group if group else bit
            candidates &= ~alternatives
            weight = alternatives.bit_count()
            cluster = owner[j]
            new_touched, new_linked = touched, linked
            if cluster >= 0:
                cluster_bit = 1 << cluster
                if exchange_clusters and not touched & cluster_bit:
                    signature = signatures[cluster]
                    if signature in seen_unopened:
                        continue
                    seen_unopened.add(signature)
                    weight *= sum(
                        s == signature and not touched & (1 << k)
                        for k, s in enumerate(signatures)
                    )
                new_touched |= cluster_bit
                if i in core_positions:
                    new_linked |= cluster_bit
            parity = (rest & (bit - 1)).bit_count() % 2
            pairs.append((i, j))
            yield from visit(
                rest ^ bit,
                -sign if parity else sign,
                multiplicity * weight,
                new_touched,
                new_linked,
            )
            pairs.pop()

    yield from visit((1 << len(ops)) - 1, 1, 1, 0, 0)
