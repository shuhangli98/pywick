import numpy as np
from pyscf import ao2mo
import wick
from time import perf_counter
from dataclasses import dataclass, field
from math import factorial


def gccsd_ao2mo(mf):
    mo = mf.mo_coeff
    mol = mf.mol
    nmo = mo.shape[1]
    hcore_ao = mf.get_hcore()
    mo_a, mo_b = mo[: mol.nao], mo[mol.nao :]
    h1e = mo.T.conj() @ hcore_ao @ mo
    eri_or_mol = mol if mf._eri is None else mf._eri  # AO
    # Spin orbital ao2mo
    g2e = ao2mo.restore(4, ao2mo.general(eri_or_mol, (mo_a, mo_a, mo_b, mo_b)), nmo)
    g2e = g2e + g2e.transpose(1, 0)
    g2e += ao2mo.restore(4, ao2mo.full(eri_or_mol, mo_a), nmo)
    g2e += ao2mo.restore(4, ao2mo.full(eri_or_mol, mo_b), nmo)
    g2e = ao2mo.restore(1, g2e, nmo)
    nocc = mol.nelectron
    nvir = len(h1e) - nocc
    ecore = enuc = mol.energy_nuc()
    h1e = np.asarray(h1e, dtype=float)
    g2e = np.asarray(g2e, dtype=float)
    ecore += np.einsum("ii->", h1e[:nocc, :nocc], optimize="optimal")
    ecore += 0.5 * np.einsum(
        "iijj->", g2e[:nocc, :nocc, :nocc, :nocc], optimize="optimal"
    )
    ecore -= 0.5 * np.einsum(
        "ijji->", g2e[:nocc, :nocc, :nocc, :nocc], optimize="optimal"
    )
    h1e = h1e.copy()
    h1e += np.einsum("mnjj->mn", g2e[:, :, :nocc, :nocc], optimize="optimal")
    h1e -= np.einsum("mjjn->mn", g2e[:, :nocc, :nocc, :], optimize="optimal")
    # Chem to Phys
    g2e = g2e.transpose(0, 2, 1, 3)
    return ecore, h1e, g2e, nocc, nvir


orbital_space_info = {
    wick.WickSpaces.Occupied: "pqrsijklmno",
    wick.WickSpaces.Virtual: "pqrsabcdefg",
}


@dataclass
class CCEquations:
    energy: object
    singles: object
    doubles: object
    scripts: dict
    code: dict
    seconds: float
    max_order: int
    stats: dict = field(default_factory=dict)


def generate_equations(engine=wick, max_order=4, verbose=True, merge=True):
    """Derive (H_N exp(T1+T2))_C through fourth order by default.

    engine can be the archived implementation for like-for-like benchmarks.
    max_order=2 is retained for timing the former truncated example only.
    """
    if max_order not in (2, 3, 4):
        raise ValueError("max_order must be 2, 3, or 4")
    start = perf_counter()
    spaces = {
        engine.WickSpaces(int(space)): labels
        for space, labels in orbital_space_info.items()
    }
    parse = lambda text: engine.WickExpr.parse(text, spaces)
    normal = lambda expr: expr.wick_theorem(no_ctr=True).simplify()
    contract = lambda expr: expr.wick_theorem(0).simplify().keep_connected()
    h = normal(
        parse("SUM <pq> h[pq] C[p] D[q]")
        + 0.5 * parse("SUM <pqrs> v[pqrs] C[p] C[q] D[s] D[r]")
    )
    t = normal(
        parse("SUM <ai> t[ia] C[a] D[i]")
        + 0.25 * parse("SUM <abij> t[ijab] C[a] C[b] D[j] D[i]")
    )
    hbar = power = h
    for order in range(1, max_order + 1):
        power = power * t
        hbar = hbar + (1 / factorial(order)) * power
    from graph_canonical import TermAccumulator, cluster_symmetries

    expressions = {}
    stats = {}
    for name, expr in (
        ("energy", hbar),
        ("singles", parse("C[i] D[a]") * hbar),
        ("doubles", parse("C[i] C[j] D[b] D[a]") * hbar),
    ):
        stage = perf_counter()
        expressions[name] = contract(expr)
        contraction_seconds = perf_counter() - stage
        raw_terms = len(expressions[name].terms)
        merge_start = perf_counter()
        if merge:
            accumulator = TermAccumulator(cluster_symmetries(2))
            for term in expressions[name].terms:
                accumulator.add(term)
            expressions[name] = accumulator.expression()
        stats[name] = dict(
            raw_terms=raw_terms,
            merged_terms=len(expressions[name].terms),
            contraction_seconds=contraction_seconds,
            merge_seconds=perf_counter() - merge_start,
        )
        if verbose:
            print(
                f"[codegen] {name}: {perf_counter()-stage:.4f}s "
                f"terms={len(expressions[name].terms)}",
                flush=True,
            )
    lhs = {"energy": "E", "singles": "t1new[ia]", "doubles": "t2new[ijab]"}
    scripts = {
        name: expr.to_einsum(engine.WickTensor.parse(lhs[name], spaces))
        for name, expr in expressions.items()
    }
    code = {
        name: compile(script, f"<wick-{name}>", "exec")
        for name, script in scripts.items()
    }
    elapsed = perf_counter() - start
    if verbose:
        print(f"[codegen] total through T^{max_order}: {elapsed:.4f}s", flush=True)
    return CCEquations(
        **expressions,
        scripts=scripts,
        code=code,
        seconds=elapsed,
        max_order=max_order,
        stats=stats,
    )


class GCCSD:
    def __init__(self, mf, max_cycle=50, equations=None):
        self.equations = generate_equations() if equations is None else equations
        self.max_cycle = max_cycle
        self.ao2mo(mf)

    def ao2mo(self, mf):
        self.ecore, self.h1e, self.g2e, self.nocc, self.nvir = gccsd_ao2mo(mf)

    def init_amps(self):
        return np.zeros((self.nocc, self.nvir)), np.zeros(
            (self.nocc, self.nocc, self.nvir, self.nvir)
        )

    def update_amps(self, t1, t2):
        tOV = t1
        tOOVV = t2
        slices = {"O": slice(0, self.nocc), "V": slice(self.nocc, None)}
        h1 = {"h" + i + j: self.h1e[slices[i], slices[j]] for i in "OV" for j in "OV"}
        h2 = {
            "v" + i + j + k + l: self.g2e[slices[i], slices[j], slices[k], slices[l]]
            for i in "OV"
            for j in "OV"
            for k in "OV"
            for l in "OV"
        }
        t1new = np.zeros_like(t1)
        t2new = np.zeros_like(t2)
        exec(self.equations.code["singles"], globals(), h1 | h2 | locals())
        exec(self.equations.code["doubles"], globals(), h1 | h2 | locals())
        foo = np.diag(h1["hOO"])
        fvv = np.diag(h1["hVV"])
        t1new /= foo[:, None] - fvv[None, :]
        t2new /= (
            foo[:, None, None, None]
            + foo[None, :, None, None]
            - fvv[None, None, :, None]
            - fvv[None, None, None, :]
        )
        t1new += t1
        t2new += t2
        return t1new, t2new

    def energy(self, t1, t2):
        tOV = t1
        tOOVV = t2
        vOOVV = self.g2e[: self.nocc, : self.nocc, self.nocc :, self.nocc :]
        hOV = self.h1e[: self.nocc, self.nocc :]
        E = np.array(0.0)
        exec(self.equations.code["energy"], globals(), locals())
        return E

    def kernel(self, tol=1e-8, tolnormt=1e-6, max_cycle=None):
        max_cycle = self.max_cycle if max_cycle is None else max_cycle
        t1, t2 = self.init_amps()
        eold = self.energy(t1, t2)
        print("Init E_corr(GCCSD) = ", eold)
        converged = False
        for i in range(max_cycle):
            t1new, t2new = self.update_amps(t1, t2)
            normt = (
                np.linalg.norm(t1 - t1new) ** 2 + np.linalg.norm(t2 - t2new) ** 2
            ) ** 0.5
            t1, t2 = t1new, t2new
            eccsd = self.energy(t1, t2)
            print(
                "cycle = %d  E_corr(CCSD) = %.15g  dE = %.9g  norm(t1,t2) = %.6g"
                % (
                    i + 1,
                    eccsd,
                    eccsd - eold,
                    normt,
                )
            )

            if abs(eccsd - eold) < tol and normt < tolnormt:
                converged = True
                break
            eold = eccsd
        return converged, eccsd, t1, t2


if __name__ == "__main__":
    from pyscf import gto, scf, lib

    lib.num_threads(1)
    mol = gto.M(atom="O 0 0 0; H 0 1 0; H 0 0 1", basis="6-31g", verbose=4)
    mf = scf.GHF(mol).run(conv_tol=1e-14)
    result = GCCSD(mf).kernel()
    print("Converged:", result[0], "E_corr:", result[1])
