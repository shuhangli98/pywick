from enum import IntFlag
import atexit
from functools import reduce, wraps
from itertools import takewhile, accumulate
import os
from time import perf_counter

_WICK_PROFILE = os.environ.get("WICK_PROFILE", "1") != "0"
_WICK_PROFILE_DATA = {}


def _profile_start():
    return perf_counter() if _WICK_PROFILE else 0.0


def _profile_add(name, start):
    if not _WICK_PROFILE:
        return
    elapsed = perf_counter() - start
    total, count = _WICK_PROFILE_DATA.get(name, (0.0, 0))
    _WICK_PROFILE_DATA[name] = (total + elapsed, count + 1)


def _profiled(name):
    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = _profile_start()
            try:
                return func(*args, **kwargs)
            finally:
                _profile_add(name, start)

        return wrapper

    return decorate


def _print_profile():
    if not _WICK_PROFILE or not _WICK_PROFILE_DATA:
        return
    print("[wick timing] cumulative method timings")
    for name, (total, count) in sorted(
        _WICK_PROFILE_DATA.items(), key=lambda item: item[1][0], reverse=True
    ):
        print(f"[wick timing] {name:34s} {total:10.4f}s  calls={count}")


atexit.register(_print_profile)


class WickSpaces(IntFlag):
    """
    Types of orbital spaces.
    """

    # General = Occupied | Active | Virtual
    # Hole = Occupied | Active
    # Particle = Active | Virtual
    Nothing = 0
    Occupied = 1
    Active = 2
    Virtual = 4

    def short_repr(self) -> str:
        """
        Single-character shorthand: N, O(Occupied), A(Active), H(Hole), V(Virtual), P(Particle), or G(General).
        """
        return "NOAHVGPG"[int(self)]


class WickIndex:
    """
    An orbital index that stores its space type and label.
    space: space type, eg: WickSpaces.Occupied ...
    label: the label of the orbital, eg: i, j, a, b, p, q...
    """

    def __init__(self, label, space=None):
        self.label = label
        self.space = space or WickSpaces.Nothing

    def __eq__(self, other):
        if not isinstance(other, WickIndex):
            return NotImplemented
        return self.label == other.label and self.space == other.space

    def __hash__(self):
        return hash((self.space, self.label))

    def __lt__(self, index):
        if not isinstance(index, WickIndex):
            return NotImplemented
        return (self.space, self.label) < (index.space, index.label)

    def __repr__(self):
        return self.label

    @staticmethod
    def parse(label_str, orbital_space_info):
        """
        Parse a string of labels into a list of WickIndex objects, using the provided orbital space information.
        label_str: a string of labels, e.g. "ijabpq"
        orbital_space_info: a dict mapping labels to their space types, e.g. {WickSpaces.Occupied: "ij", WickSpaces.Virtual: "ab"}
        """

        def get_space_types(label):
            """
            Find all space types for a given label.
            """
            matching_types = [
                space_type
                for space_type, labels in orbital_space_info.items()
                if label in labels
            ]

            if matching_types:
                return reduce(lambda x, y: x | y, matching_types)
            else:
                raise ValueError(f"Label '{label}' not found in orbital space info.")

        return [WickIndex(label, get_space_types(label)) for label in label_str]


class TensorTypes(IntFlag):
    Creation = 1
    Annihilation = 2
    SpinFreeOperator = 3
    KroneckerDelta = 4
    Tensor = 5


class WickTensor:
    """
    A symbolic tensor object. Operator is also a tensor
    """

    def __init__(self, name, indices=None, operator_type=None):
        self.name = name
        self.indices = indices or []  # list of WickIndex objects
        self.type = operator_type or TensorTypes.Tensor

    def __eq__(self, other):
        if not isinstance(other, WickTensor):
            return NotImplemented
        return (self.name, self.indices, self.type) == (
            other.name,
            other.indices,
            other.type,
        )

    def __lt__(self, other):
        if not isinstance(other, WickTensor):
            return NotImplemented
        return self._fermi_ordering(self) < self._fermi_ordering(other)

    def __repr__(self):
        indices_str = "".join(repr(idx) for idx in self.indices)
        return f"{self.name}[{indices_str}]"

    @staticmethod
    def _fermi_ordering(op):
        """
        Encode the fermionic ordering for operators:
        D(vir) > C(occ) > D(occ) > C(vir)
        """
        is_ann = op.type == TensorTypes.Annihilation
        is_occ = op.indices[0].space == WickSpaces.Occupied
        order_map = {
            (False, False): 0,  # C(vir)
            (True, True): 1,  # D(occ)
            (False, True): 2,  # C(occ)
            (True, False): 3,  # D(vir)
        }
        return order_map[(is_ann, is_occ)]

    @staticmethod
    def cre(indices, name="C"):
        return WickTensor(name, indices, TensorTypes.Creation)

    @staticmethod
    def ann(indices, name="D"):
        return WickTensor(name, indices, TensorTypes.Annihilation)

    @staticmethod
    def kronecker_delta(indices):
        return WickTensor("delta", indices, TensorTypes.KroneckerDelta)

    @staticmethod
    def parse(tensor_str, orbital_space_info):
        """
        Given a string like "v[ijkl]", parse it to a Tensor.
        """
        types_map = {
            "C": TensorTypes.Creation,
            "D": TensorTypes.Annihilation,
            "delta": TensorTypes.KroneckerDelta,
        }
        name = tensor_str.split("[")[0].strip()
        type = types_map.get(name, None)
        indices = (
            ""
            if len(tensor_str.split("[")) == 1
            else tensor_str.split("[")[1][:-1].strip()
        )
        return WickTensor(name, WickIndex.parse(indices, orbital_space_info), type)


class WickString:
    """A string of WickTensor.

    Args:
        tensors (list): a list of WickTensor object.
        contractions (set): a set of contracted WickIndex.
        factor (float): the coefficient of the equation.
    """

    def __init__(self, tensors=None, contractions=None, factor=None):
        self.tensors = tensors or []
        self.contractions = contractions or set()
        self.factor = factor if factor is not None else (1.0 if tensors else 0.0)

    def __eq__(self, other):
        return (self.tensors, self.contractions, self.factor) == (
            other.tensors,
            other.contractions,
            other.factor,
        )

    def __lt__(self, other):
        return (
            len(self.tensors),
            len(self.contractions),
            self.tensors,
            self.contractions,
            self.factor,
        ) < (
            len(other.tensors),
            len(other.contractions),
            other.tensors,
            other.contractions,
            other.factor,
        )

    def __abs__(self):
        return WickString(self.tensors, self.contractions, 1.0)

    def used_index_labels(self):
        return set([idx.label for tensor in self.tensors for idx in tensor.indices])

    @staticmethod
    def parse(eq, orbital_space_info):
        """Given a string like "0.5 SUM <pq> h[pq] C[p] D[q]", parse it to a Equation.

        Args:
            eq (str)
            orbital_space_info (dict)
        """
        if any([i + j + k in eq for i in "Ee" for j in "+-" for k in "0123456789"]):
            raise ValueError("Scientific notation is not supported.")
        is_num = lambda i: i in "0123456789.+-"
        parse_factor = lambda s: float(s) if s not in "+-" else float(s + "1")

        if len(eq.strip()) == 0:
            return WickString()
        if "SUM" in eq:
            coeff, expr = eq.split("SUM")
            factor = None if len(coeff.strip()) == 0 else parse_factor(coeff.strip())
            contractions = set(
                WickIndex.parse(expr.split(">")[0].strip()[1:], orbital_space_info)
            )
            tensors_str = expr.split(">")[1].strip()
        elif is_num(eq.strip()[0]):
            factor_str = "".join(takewhile(is_num, eq.strip()))
            factor = parse_factor(factor_str)
            contractions = set()
            tensors_str = eq.strip()[len(factor_str) :].strip()
        else:
            factor = None
            contractions = set()
            tensors_str = eq.strip()
        tensors = [i for i in tensors_str.replace("]", "] ").split()]
        return WickString(
            [WickTensor.parse(i, orbital_space_info) for i in tensors],
            contractions,
            factor,
        )

    @staticmethod
    @_profiled("WickString.multiplication")
    def multiplication(a, b):
        import string

        a_labels = a.used_index_labels()
        b_labels = b.used_index_labels()
        a_contractions = set([idx.label for idx in a.contractions])
        b_contractions = set([idx.label for idx in b.contractions])
        unused_labels = sorted(list(set(string.ascii_letters) - a_labels - b_labels))

        # Build renaming maps
        # For indices that are contracted in a and also appear in b.
        a_map = {old: new for old, new in zip(a_contractions & b_labels, unused_labels)}
        # For indices that are contracted in b and also appear in a.
        b_map = {
            old: new
            for old, new in zip(
                b_contractions & (a_labels - a_contractions),
                unused_labels[len(a_map) :],
            )
        }

        apply_map = lambda indices, rename_map: [
            WickIndex(rename_map.get(idx.label, idx.label), idx.space)
            for idx in indices
        ]
        a_tensors = [
            WickTensor(i.name, apply_map(i.indices, a_map), i.type) for i in a.tensors
        ]
        b_tensors = [
            WickTensor(i.name, apply_map(i.indices, b_map), i.type) for i in b.tensors
        ]

        contractions = set(
            apply_map(a.contractions, a_map) + apply_map(b.contractions, b_map)
        )

        return WickString(a_tensors + b_tensors, contractions, a.factor * b.factor)

    def __mul__(self, other):
        return (
            WickString(self.tensors, self.contractions, self.factor * other)
            if isinstance(other, (float, int))
            else WickString.multiplication(self, other)
        )

    def __rmul__(self, other):
        return (
            WickString(self.tensors, self.contractions, self.factor * other)
            if isinstance(other, (float, int))
            else WickString.multiplication(other, self)
        )

    def __repr__(self):
        sum_expr = "SUM <%s|%s>" % (
            "".join(["%r" % x for x in self.contractions]),
            "".join([x.space.short_repr() for x in self.contractions]),
        )
        return "(%16.10f) %s %s" % (
            self.factor,
            sum_expr if len(self.contractions) != 0 else "",
            " ".join(["%r" % x for x in self.tensors]),
        )

    @_profiled("WickString.to_einsum")
    def to_einsum(self, x):
        """Print this object as np.einsum expression.
        x : WickTensor in the left hand side"""
        import string

        used_index = set([idx for itensor in self.tensors for idx in itensor.indices])
        used_labels = set([idx.label for idx in used_index])
        available_labels = set(string.ascii_letters) - used_labels
        letter_map = {}  # idx:new_label
        letter_map = {
            a: b if a.label in [key.label for key in letter_map] else a.label
            for a, b in zip(used_index, available_labels)
        }
        left_ctr = ",".join(
            [
                "".join([letter_map[idx] for idx in itensor.indices])
                for itensor in self.tensors
            ]
        )  # 'ijab,abcd'

        right_ctr = "".join([letter_map[idx] for idx in x.indices])  # 'ijcd'
        ctr = left_ctr + "->" + right_ctr

        left_tensors = ", ".join(
            [
                itensor.name
                + "".join([idx.space.short_repr() for idx in itensor.indices])
                for itensor in self.tensors
            ]
        )

        script = (
            x.name
            + " += "
            + str(self.factor)
            + ' * np.einsum("'
            + ctr
            + '", '
            + left_tensors
            + ", optimize=True)"
        )

        return script

    def sort(self):
        """Sort tensors (not operators) so that different WickString can be compared."""
        tensor_list = [
            x
            for x in self.tensors
            if x.type in {TensorTypes.Tensor, TensorTypes.KroneckerDelta}
        ]
        operator_list = [
            x
            for x in self.tensors
            if x.type not in {TensorTypes.Tensor, TensorTypes.KroneckerDelta}
        ]
        return WickString(
            tensors=sorted(tensor_list) + operator_list,
            contractions=self.contractions,
            factor=self.factor,
        )

    @_profiled("WickString.split_composite_space")
    def split_composite_space(self):
        """Break the general indices to several terms with only O/V-type indices.
        For example, SUM <pq|GG> h[pq] = SUM <pq|OO> h[pq] + SUM <pq|OV> h[pq] + SUM <pq|VO> h[pq] + SUM <pq|VV> h[pq].
        """

        from itertools import product

        available_spaces = {WickSpaces.Occupied, WickSpaces.Active, WickSpaces.Virtual}
        index_list = [
            [WickIndex(x.label, y) for y in available_spaces if y in x.space]
            for x in self.contractions
            if int(x.space).bit_count() >= 1
        ]
        terms = []
        for idx_prod in product(*index_list):
            index_map = {x.label: x for x in idx_prod}  # label: new_index
            tensors = [
                WickTensor(
                    x.name, [index_map.get(idx.label, idx) for idx in x.indices], x.type
                )
                for x in self.tensors
            ]
            contractions = set([index_map.get(x.label, x) for x in self.contractions])
            terms.append(
                WickString(
                    tensors=tensors, contractions=contractions, factor=self.factor
                )
            )

        return WickExpr(terms)

    def _fully_contract(self, ops, tensors):
        """Enumerate perfect matchings only, preserving the original pair rules.

        The leftmost remaining operator must be paired in a full contraction.
        Fixing it at each recursion level visits every perfect matching once.
        Moving its partner next to it contributes (-1)**(intervening operators).
        """
        if len(ops) % 2:
            return WickExpr([])

        balance = {}
        for op in ops:
            space = op.indices[0].space
            balance[space] = balance.get(space, 0) + (
                1 if op.type == TensorTypes.Creation else -1
            )
        if any(balance.values()):
            return WickExpr([])

        partners = [0] * len(ops)
        for i, left in enumerate(ops):
            for j in range(i + 1, len(ops)):
                right = ops[j]
                if (
                    left > right
                    and left.type != right.type
                    and left.indices[0].space == right.indices[0].space
                ):
                    partners[i] |= 1 << j

        terms = []
        pairs = []

        def visit(remaining, sign):
            if not remaining:
                deltas = [
                    WickTensor.kronecker_delta([ops[i].indices[0], ops[j].indices[0]])
                    for i, j in pairs
                ]
                terms.append(
                    WickString(tensors + deltas, self.contractions, sign * self.factor)
                )
                return

            first = remaining & -remaining
            i = first.bit_length() - 1
            rest = remaining ^ first
            candidates = partners[i] & rest
            while candidates:
                partner = candidates & -candidates
                candidates ^= partner
                j = partner.bit_length() - 1
                parity = (rest & (partner - 1)).bit_count() % 2
                pairs.append((i, j))
                visit(rest ^ partner, -sign if parity else sign)
                pairs.pop()

        visit((1 << len(ops)) - 1, 1)
        return WickExpr(terms)

    @_profiled("WickString.wick_theorem")
    def wick_theorem(self, max_unctr=-1, no_ctr=False):
        """Apply normal order to this object.
        Args: max_unctr : max number of uncontracted C/D operators.
                          -1 means all terms will be kept.
                          0 means only totally contracted terms (with no C/D) will be kept.
                          2 will keep terms with 0, 1, or 2 C/D operators.
                          Namely, with max_unctr == 2, C, CD, and DD will be kept,
                          CDC will not be kept.
              no_ctr : if True, will return the normal ordered expression, ignoring
                       any additional terms with contraction. Namely, this is simply add the
                       "{ }" to the given expression.
        """

        non_op_types = {TensorTypes.KroneckerDelta, TensorTypes.Tensor}
        ops = [x for x in self.tensors if x.type not in non_op_types]
        tensors = [x for x in self.tensors if x.type in non_op_types]
        if (
            max_unctr == 0
            and not no_ctr
            and all(
                op.type in {TensorTypes.Creation, TensorTypes.Annihilation}
                for op in ops
            )
        ):
            return self._fully_contract(ops, tensors)
        # find all possible pairs that can be contracted
        ctr_pairs_nested = [
            [
                (x, y)
                for y in range(x + 1, len(ops))
                if ops[x] > ops[y]
                and ops[x].type != ops[y].type
                and ops[x].indices[0].space == ops[y].indices[0].space
            ]
            for x in range(len(ops))
        ]
        # ctr_pairs = [x for x in ctr_pairs_nested for x in x]
        ctr_pairs = reduce(lambda x, y: x + y, ctr_pairs_nested, [])
        # ctr_start_idxs = [0]
        # for x in ctr_pairs_nested:
        #     ctr_start_idxs.append(ctr_start_idxs[-1] + len(x))

        # 0 = even, 1 = odd
        # op_idxs is the map from index in normal ordered to index in original string
        # op_idxs = sorted(range(len(ops)), key=lambda x: ops[x])
        # op_idxs = sorted(range(len(ops)), key=ops.__getitem__)
        # rev_idxs is the map from index in original to index in normal ordered string
        # op_rev_idxs = sorted(range(len(ops)), key=op_idxs.__getitem__)
        # for x in range(len(ops)):
        #     assert op_rev_idxs[op_idxs[x]] == x
        # sign for reordering operators to the normal order

        # init_sign = 0
        # tmp_ops = ops[:]
        # for x in range(len(tmp_ops)):
        #     min_idx = sorted(range(len(tmp_ops)), key=tmp_ops.__getitem__)[0]
        #     init_sign ^= min_idx % 2
        #     tmp_ops = tmp_ops[:min_idx] + tmp_ops[min_idx + 1 :]

        # init_sign = 0
        # for i, ri in enumerate(op_idxs):
        #     for rj in op_idxs[i + 1 :]:
        #         init_sign ^= rj < ri

        terms = []
        queue = [[]]
        if no_ctr:
            terms, queue = queue, terms
        while len(queue) != 0:
            q = queue.pop()
            if max_unctr == -1 or len(ops) - 2 * len(q) <= max_unctr:
                terms.append(q)
            for ctr in ctr_pairs:
                if all(ctr[k] != i[l] for i in q for k in [0, 1] for l in [0, 1]) and (
                    q == [] or ctr > q[-1]
                ):
                    queue.append(q + [ctr])
        final_terms = []

        for term in terms:
            sign = 0
            for x, y in term:
                sign ^= (y - x) % 2 == 0  # Single-contraction sign
            new_term = [x for x in term for x in x]
            for i, ri in enumerate(new_term):
                for rj in new_term[i + 1 :]:
                    sign ^= rj < ri  # Cross-point sign
            delta_tensors = [
                WickTensor.kronecker_delta([ops[x].indices[0], ops[y].indices[0]])
                for x, y in term
            ]

            mask = [True] * len(ops)
            for i in new_term:
                mask[i] = False

            rem_ops = [ops[x] for x in range(len(ops)) if mask[x]]
            for i, ri in enumerate(rem_ops):
                for rj in rem_ops[i + 1 :]:
                    sign ^= rj < ri  # To normal-ordering
            rem_ops = sorted(rem_ops)
            final_terms.append(
                WickString(
                    tensors + delta_tensors + rem_ops,
                    self.contractions,
                    [1.0, -1.0][sign] * self.factor,
                )
            )

        return WickExpr(final_terms)

    @_profiled("WickString.simplify_delta")
    def simplify_delta(self):
        """Remove delta tensors in this object and change the indices accordingly."""
        import copy

        new_wstr = copy.deepcopy(self)
        new_tensors = []
        for i in new_wstr.tensors:
            if i.type == TensorTypes.KroneckerDelta:
                x, y = i.indices
                if x.space & y.space == WickSpaces.Nothing:
                    new_wstr.factor = 0
                elif x not in new_wstr.contractions and y not in new_wstr.contractions:
                    found = False
                    for j in new_tensors:
                        if j.type == TensorTypes.KroneckerDelta:
                            if j.indices == (x, y) or j.indices == (y, x):
                                found = True
                                break
                    if not found:
                        new_tensors.append(i)
                else:
                    z = y if x in new_wstr.contractions else x
                    new_wstr.contractions -= set(
                        [x if x in new_wstr.contractions else y]
                    )
                    for k in new_wstr.tensors:
                        if k != i:
                            k.indices = [l if l not in (x, y) else z for l in k.indices]
            else:
                new_tensors.append(i)
        new_wstr.tensors = new_tensors
        return new_wstr

    @_profiled("WickString.is_connected")
    def is_connected(self, names=["h", "v"]):
        # Only for CC
        core_tensors = [x for x in self.tensors if x.name in names]
        assert len(core_tensors) == 1
        core_indices = core_tensors[0].indices
        other_tensors = [x for x in self.tensors if x.name not in names]
        for i in other_tensors:
            if not any(x in core_indices for x in i.indices):
                return False
        return True


class WickExpr:
    """A collection of WickString.
    terms : list of WickString.
    """

    def __init__(self, terms):
        self.terms = [terms] if isinstance(terms, WickString) else terms

    def __eq__(self, other):
        return self.terms == other.terms

    def __lt__(self, other):
        return self.terms < other.terms

    @_profiled("WickExpr.__mul__")
    def __mul__(self, other):
        if isinstance(other, (float, int)):
            return WickExpr([other * i for i in self.terms])
        else:
            return WickExpr([i * j for i in self.terms for j in other.terms])

    def __rmul__(self, other):
        if isinstance(other, (float, int)):
            return WickExpr([other * i for i in self.terms])
        else:
            return WickExpr([i * j for i in other.terms for j in self.terms])

    def __add__(self, other):
        return WickExpr(self.terms + other.terms)

    def __sub__(self, other):
        return self + ((-1.0) * other)

    def __xor__(self, other):  # Commutator
        return self * other - other * self

    def __repr__(self):
        return "\n".join("%r" % x for x in self.terms) + "\n"

    @_profiled("WickExpr.to_einsum")
    def to_einsum(self, x):
        """Print this object as np.einsum expression.
        x : WickTensor in the left hand side"""
        return "\n".join(i.to_einsum(x) for i in self.terms) + "\n"

    @staticmethod
    def parse(x, orbital_space_info):
        """
        Given a string like "SUM <pq> h[pq] C[p]D[q] + 0.5 SUM<pqrs> v[pqrs] C[p]C[q]D[s]D[r]",
        parse it to a WickExpr.
        Args: x : a string
              orbital_space_info : a dict like { WickIndexTypes.Inactive : "ijkl", ... }
        """
        return WickExpr(
            [
                WickString.parse(i, orbital_space_info)
                for i in x.replace("+", "\n+").replace("-", "\n-").split("\n")
                if i.strip() != ""
            ]
        )

    @_profiled("WickExpr.split_composite_space")
    def split_composite_space(self):
        """Break the P-type indices to several terms with only I/E-type indices.
        For example, SUM <pq|PP> = SUM <pq|II> + SUM <pq|IE> + SUM <pq|EI> + SUM <pq|EE>.
        """
        return WickExpr(
            [y for x in self.terms for y in x.split_composite_space().terms]
        )

    @_profiled("WickExpr.wick_theorem")
    def wick_theorem(self, max_unctr=-1, no_ctr=False):
        """Apply normal order to this object.
        Args: max_unctr : max number of uncontracted C/D operators.
                          -1 means all terms will be kept.
                          0 means only totally contracted terms (with no C/D) will be kept.
                          2 will keep terms with 0, 1, or 2 C/D operators.
                          Namely, with max_unctr == 2, C, CD, and DD will be kept,
                          CDC will not be kept.
              no_ctr : if True, will return the normal ordered expression, ignoring
                       any additional terms with contraction. Namely, this is simply add the
                       "{ }" to the given expression.
        """
        return WickExpr(
            [
                y
                for x in self.split_composite_space().terms
                for y in x.wick_theorem(max_unctr=max_unctr, no_ctr=no_ctr).terms
            ]
        )

    @_profiled("WickExpr.simplify_delta")
    def simplify_delta(self):
        """Remove delta tensors in this object and change the indices accordingly."""
        return WickExpr([t.simplify_delta() for t in self.terms])

    @_profiled("WickExpr.simplify_zero")
    def simplify_zero(self):
        """Remove terms in this object with a zero pre-factor."""
        return WickExpr([x for x in self.terms if abs(x.factor) > 1e-12])

    @_profiled("WickExpr.simplify")
    def simplify(self):
        """Simplify the expression."""
        return self.simplify_delta().simplify_zero()

    @_profiled("WickExpr.keep_connected")
    def keep_connected(self, names=["h", "v"]):
        return WickExpr([x for x in self.terms if x.is_connected(names=names)])
