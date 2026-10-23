from wick import WickString, WickTensor, TensorTypes


def reduce_disjoint_deltas(term):
    mapping = {}
    seen = set()
    for tensor in term.tensors:
        if tensor.type != TensorTypes.KroneckerDelta:
            continue
        x, y = tensor.indices
        if (
            x == y
            or x in seen
            or y in seen
            or x.space != y.space
            or (x not in term.contractions and y not in term.contractions)
        ):
            return term.simplify_delta()
        seen.update((x, y))
        removed, kept = (x, y) if x in term.contractions else (y, x)
        mapping[removed] = kept
    tensors = [
        WickTensor(t.name, [mapping.get(idx, idx) for idx in t.indices], t.type)
        for t in term.tensors
        if t.type != TensorTypes.KroneckerDelta
    ]
    return WickString(tensors, term.contractions - mapping.keys(), term.factor)
