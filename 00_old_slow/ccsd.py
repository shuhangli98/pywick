import pyscf
import numpy as np
from pyscf import ao2mo
import wick
from time import perf_counter


def _term_count(x):
    return len(getattr(x, "terms", []))


def _timed_codegen(label, fn):
    print(f"[ccsd timing] begin {label}", flush=True)
    start = perf_counter()
    result = fn()
    elapsed = perf_counter() - start
    terms = _term_count(result)
    suffix = f" terms={terms}" if terms else ""
    print(f"[ccsd timing] end   {label}: {elapsed:.4f}s{suffix}", flush=True)
    return result


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

_codegen_start = perf_counter()

# x: WickExpr
Pt = lambda x: wick.WickTensor.parse(x, orbital_space_info)
P = lambda x: wick.WickExpr.parse(x, orbital_space_info)
NR = lambda x: x.wick_theorem(-1, no_ctr=True).simplify()
FC = lambda x: x.wick_theorem(0).simplify().keep_connected(names=["h", "v"])

h1 = _timed_codegen("parse h1", lambda: P("SUM <pq> h[pq] C[p] D[q]"))
h2 = _timed_codegen(
    "parse h2", lambda: 0.5 * P("SUM <pqrs> v[pqrs] C[p] C[q] D[s] D[r]")
)
t1 = _timed_codegen("parse t1", lambda: P("SUM <ai> t[ia] C[a] D[i]"))
t2 = _timed_codegen(
    "parse t2", lambda: 0.25 * P("SUM <abij> t[ijab] C[a] C[b] D[j] D[i]")
)
op1 = _timed_codegen("parse op1", lambda: P("C[i] D[a]"))
op2 = _timed_codegen("parse op2", lambda: P("C[i] C[j] D[b] D[a]"))

h = _timed_codegen("normal order h", lambda: NR(h1 + h2))
t = _timed_codegen("normal order t", lambda: NR(t1 + t2))

# # TODO
# hbar = h + (h ^ t)

hbar = _timed_codegen(
    "build hbar",
    lambda: h
    + h * t
    + 1 / 2 * h * t * t,  # + 1 / 6 * h * t * t * t + 1 / 24 * h * t * t * t * t
)

energy_eq = _timed_codegen("derive energy_eq", lambda: FC(hbar))
# print("start t1")
t1_eq = _timed_codegen("derive t1_eq", lambda: FC(op1 * hbar))
# print(t1_eq.to_einsum(Pt("t1new[ia]")))
# print("start t2")
t2_eq = _timed_codegen("derive t2_eq", lambda: FC(op2 * hbar))
# print(t2_eq.to_einsum(Pt("t2new[ijab]")))
_timed_codegen(
    "generate einsum scripts",
    lambda: (
        energy_eq.to_einsum(Pt("E")),
        t1_eq.to_einsum(Pt("t1new[ia]")),
        t2_eq.to_einsum(Pt("t2new[ijab]")),
    ),
)
print(
    f"[ccsd timing] symbolic code generation total: {perf_counter() - _codegen_start:.4f}s",
    flush=True,
)


class GCCSD:
    def __init__(self, mf, max_cycle=50):
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
        exec(t1_eq.to_einsum(Pt("t1new[ia]")), globals(), h1 | h2 | locals())
        exec(t2_eq.to_einsum(Pt("t2new[ijab]")), globals(), h1 | h2 | locals())
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
        exec(energy_eq.to_einsum(Pt("E")))
        return E

    def kernel(self, tol=1e-8, tolnormt=1e-6, max_cycle=50):
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


from pyscf import gto, scf

mol = gto.M(atom="O 0 0 0; H 0 1 0; H 0 0 1", basis="6-31g", verbose=4)
mf = scf.GHF(mol).run(conv_tol=1e-14)
# ccsd = gccsd.GCCSD(mf).run()
wccsd = GCCSD(mf).kernel()
