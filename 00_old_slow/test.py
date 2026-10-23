from wick import *
from itertools import accumulate
from functools import reduce

# general = WickSpaces.Occupied | WickSpaces.Virtual
# print(int(general).bit_count())
# print(WickSpaces.Active in general)
# quit()

# Understand accumulate and reduce
print(list(accumulate([1, 2, 3, 4, 5], lambda x, y: x * y)))
print(hash("hello"))
print(reduce(lambda x, y: x | y, [1, 2, 3]))
print(1 | 2 | 4)

# Test WickIndex
orbital_space_info = {WickSpaces.Occupied: "ij", WickSpaces.Virtual: "kl"}
indices = WickIndex.parse("ijkl", orbital_space_info)
print(indices)  # [i, j, k, l]
print([idx.space for idx in indices])

# Test Tensor
orbital_space_info = {
    WickSpaces.Occupied: "pqrsijklmno",
    WickSpaces.Virtual: "pqrsabcdefg",
}
P = lambda x: WickTensor.parse(x, orbital_space_info)
print("Parsing h[pq] = %r" % P("h[pq]"))
print("Parsing t[iajb] = %r" % P("t[iajb]"))
print("Parsing C[i] = %r" % P("C[i]"))
print("Parsing ener = %r" % P("ener"))
print("Indices of h[pq] = %r" % P("h[pq]").indices)
print("Indices types of h[pq] = %r" % [x.space for x in P("h[pq]").indices])
print("Indices types of t[iajb] = %r" % [x.space for x in P("t[iajb]").indices])
print("D[i]  < C[j] = %r" % (P("D[i]") < P("C[j]")))
print("D[a]  < C[a] = %r" % (P("D[a]") < P("C[a]")))
print("D[i] == D[j] = %r" % (P("D[i]") == P("D[j]")))
print("D[i] == D[i] = %r" % (P("D[i]") == P("D[i]")))

# Test WickString
T = lambda x: WickTensor.parse(x, orbital_space_info)
P = lambda x: WickString.parse(x, orbital_space_info)
print("Parsing h[pq]       = %r" % P("h[pq]"))
print("Parsing 0.5t[iajb]  = %r" % P("0.5t[iajb]"))
print("Parsing +0.222 C[i] = %r" % P("+0.222 C[i]"))
print("Parsing -ener       = %r" % P("-ener"))
print("Parsing SUM <pq> h[pq] [p]D[q]    = %r" % P("SUM <pq> h[pq] C[p]D[q] "))
print("Parsing SUM<pq>h[pq]C[p]D[q]       = %r" % P("SUM<pq>h[pq]C[p]D[q]"))
print("Parsing SUM   <pq>h[pq]   C[p]D[q] = %r" % P("SUM   <pq>h[pq]   C[p]D[q]"))
print("Parsing -0.5SUM <ijab>t[ia]v[ijab] = %r" % P("-0.5SUM <ijab>t[ia]v[ijab]"))
print("Parsing -0.5SUM <ijab>t[ia]v[ijab] = %r" % P("-0.5SUM <ijab>t[ia]v[ijab]"))
print("Parsing 0.5 * (0.25 C[p] D[q])     = %r" % (0.5 * P("0.25 C[p] D[q]")))
print("Parsing C[p] v[pijk] * 0.5 D[q]    = %r" % (P("C[p] v[pijk]") * P("0.5 D[q]")))
print(
    "Parsing SUM <pq>C[pq] v[pqjk] * 0.5 SUM<q> a[q] D[q]    = %r"
    % (P("SUM <pq>C[pq] v[pqjk]") * P("0.5 SUM<q> a[q] D[q]"))
)
print(
    "EINSUM  SUM   <pq>h[pq]   C[p]D[q] = %s"
    % P("SUM   <pq>h[pq]   C[p]D[q]").to_einsum(T("E"))
)
print(
    "EINSUM  -0.5SUM <ijab>t[ia]v[ijab] = %s"
    % P("-0.5SUM <ijab>t[ia]v[ijab]").to_einsum(T("E"))
)
print(
    "EINSUM  0.5 * (0.25 C[p] D[q])     = %s"
    % (0.5 * P("0.25 C[p] D[q]")).to_einsum(T("g[pq]"))
)
print(
    "EINSUM  C[p] v[pijk] * 0.5 D[q]    = %s"
    % (P("C[p] v[pijk]") * P("0.5 D[q]")).to_einsum(T("x[ij]"))
)

print("SUM <pq|GG> h[pq]    = %s" % (P("SUM <pq> h[pq]").split_composite_space()))
print("SUM <ij|OO> h[ij]    = %s" % (P("SUM <ij> h[ij]").split_composite_space()))

print(P("SUM <pqrs> D[p]C[q]D[r]C[s]").split_composite_space().terms[-1])
print("====================================")
# print(P("SUM <pqrs> D[p]C[q]D[r]C[s]").split_composite_space()[-1].wick_theorem())
x = (
    P("SUM <pqrs> D[p]C[q]D[r]C[s]")
    .split_composite_space()
    .terms[-1]
    .wick_theorem()
    .terms
)
for i in x:
    print(i)
print("====================================")
f = P("SUM <pq> f[pq]C[p]D[q]").split_composite_space().terms
fn = []
for i, x in enumerate(f):
    fn.extend(x.wick_theorem().terms)
for x in fn:
    print(x)
fn = fn[:1] + fn[2:]

t1 = P("SUM <ia> t[ia]C[a]D[i]").split_composite_space().terms
print(t1)

terms = []
for x in fn:
    terms.append(x * t1[0])
print("====================================")
for x in terms:
    print(x)
print("====================================")
final_terms = []
for x in terms:
    final_terms.extend(x.wick_theorem().terms)
for x in final_terms:
    print(x)
    print(x.simplify_delta())
    print()

P = lambda x: WickExpr.parse(x, orbital_space_info)
expr = P("SUM<pq> h[pq] C[p] D[q] +0.5 SUM<pqrs> v[pqrs] C[p] C[q] D[s] D[r]")
print(expr)
expr_simplified = expr.wick_theorem().simplify()
print(expr_simplified)
