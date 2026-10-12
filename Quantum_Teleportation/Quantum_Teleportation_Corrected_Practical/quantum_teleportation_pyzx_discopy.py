"""
Quantum Teleportation: Formal Verification via ZX Calculus
Dissertation practical using DisCoPy + PyZX.

Recommended environment:
    Python 3.10 or 3.11
    discopy==1.2.2
    pyzx==0.7.3

The program performs:
1) Canonical branch-by-branch mathematical verification.
2) DisCoPy categorical circuit construction and graphical rendering.
3) PyZX conversion, graphical ZX rendering, full_reduce simplification,
   and semantic-equivalence checking.
4) Cross-framework PyZX <-> DisCoPy ZX round-trip verification.
5) 100 random-state teleportation tests.
6) CSV, JSON, TXT and PNG result generation.
"""

from __future__ import annotations

import csv
import importlib
import json
import platform
import sys
import time
import importlib
from pathlib import Path
from typing import Any

try:
    import numpy as np  # pyright: ignore[reportMissingImports]
except ModuleNotFoundError as exc:
    raise SystemExit(
        "NumPy is required. Install it in the active Python environment with:\n"
        "  python -m pip install numpy"
    ) from exc

# Use a non-interactive backend so figure generation does not depend on
# Tcl/Tk (Tkinter). This is important on Windows installations where Tcl/Tk
# is absent or incomplete. All dissertation figures are saved as PNG files.
import matplotlib  # pyright: ignore[reportMissingImports, reportMissingModuleSource]
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports, reportMissingModuleSource]

try:
    import pyzx as zx  # pyright: ignore[reportMissingImports, reportMissingModuleSource]
    # Import the submodule dynamically so static analysis does not reject
    # PyZX installations that omit submodule metadata.
    full_reduce = importlib.import_module("pyzx.simplify").full_reduce
    try:
        compare_tensors = importlib.import_module("pyzx.tensor").compare_tensors
    except ImportError:
        compare_tensors = zx.compare_tensors
except Exception as exc:
    raise SystemExit(
        "PyZX is required. Install the pinned environment first:\n"
        "  python -m pip install pyzx==0.7.3 discopy==1.2.2 numpy matplotlib\n"
        f"Original import error: {exc}"
    )

try:
    discopy = importlib.import_module("discopy")
    Circuit = importlib.import_module("discopy.quantum.circuit").Circuit
    # Import dynamically so static analysis does not flag optional DisCoPy
    # submodule metadata as unresolved.
    DisCoPyZXDiagram = importlib.import_module("discopy.quantum.zx").Diagram
except Exception as exc:
    raise SystemExit(
        "DisCoPy is required. Install the pinned environment first:\n"
        "  python -m pip install pyzx==0.7.3 discopy==1.2.2 numpy matplotlib\n"
        f"Original import error: {exc}"
    )


# ---------------------------------------------------------------------------
# 0. Output setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
TOL = 1e-12


# ---------------------------------------------------------------------------
# 1. Canonical mathematical specification
# ---------------------------------------------------------------------------
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
ket0 = np.array([1, 0], dtype=complex)
ket_plus = H @ ket0


def kron3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.kron(np.kron(a, b), c)


def one_qubit_gate(gate: np.ndarray, target: int) -> np.ndarray:
    mats = [I2, I2, I2]
    mats[target] = gate
    return kron3(*mats)


def cx3(control: int, target: int) -> np.ndarray:
    """3-qubit CNOT in q0,q1,q2 big-endian ordering."""
    U = np.zeros((8, 8), dtype=complex)
    for i in range(8):
        bits = [(i >> 2) & 1, (i >> 1) & 1, i & 1]
        out = bits.copy()
        if bits[control] == 1:
            out[target] ^= 1
        j = (out[0] << 2) | (out[1] << 1) | out[2]
        U[j, i] = 1
    return U


def cz3(control: int, target: int) -> np.ndarray:
    U = np.eye(8, dtype=complex)
    for i in range(8):
        bits = [(i >> 2) & 1, (i >> 1) & 1, i & 1]
        if bits[control] and bits[target]:
            U[i, i] = -1
    return U


# Bell preparation on q1,q2; Bell-basis transform on q0,q1.
U_bell = cx3(1, 2) @ one_qubit_gate(H, 1)
U_pre_measure = one_qubit_gate(H, 0) @ cx3(0, 1) @ U_bell

# Deferred-measurement/coherent-correction version of teleportation.
# X correction: CNOT q1 -> q2
# Z correction: CZ q0 -> q2
U_teleport_reference = cz3(0, 2) @ cx3(1, 2) @ U_pre_measure

ket00_anc = np.kron(ket0, ket0).reshape(4, 1)
bra_pp = np.kron(ket_plus, ket_plus).conj().reshape(1, 4)
input_embedding = np.kron(I2, ket00_anc)  # 8 x 2
output_projection = np.kron(bra_pp, I2)   # 2 x 8


def branch_map(m0: int, m1: int) -> np.ndarray:
    """Unnormalised map from Alice's unknown input to Bob for one branch."""
    bra = np.zeros((1, 4), dtype=complex)
    bra[0, 2 * m0 + m1] = 1
    projector = np.kron(bra, I2)
    return projector @ U_pre_measure @ input_embedding


def branch_correction(m0: int, m1: int) -> np.ndarray:
    """Bob correction C = Z^{m0} X^{m1}."""
    return np.linalg.matrix_power(Z, m0) @ np.linalg.matrix_power(X, m1)


def global_phase_error(a: np.ndarray, b: np.ndarray) -> float:
    """Frobenius error after aligning an irrelevant global phase."""
    av = a.reshape(-1)
    bv = b.reshape(-1)
    idx = int(np.argmax(np.abs(bv)))
    if abs(bv[idx]) < 1e-15:
        return float(np.linalg.norm(a - b))
    phase = av[idx] / bv[idx]
    if abs(phase) > 1e-15:
        phase /= abs(phase)
    return float(np.linalg.norm(a - phase * b))


def pyzx_version() -> str:
    return getattr(zx, "__version__", "unknown")


def discopy_version() -> str:
    return getattr(discopy, "__version__", "unknown")


# ---------------------------------------------------------------------------
# 2. Drawing helpers
# ---------------------------------------------------------------------------
def save_pyzx_graph(graph, path: Path, title: str) -> None:
    """Save a PyZX graph using PyZX's Matplotlib renderer."""
    plt.close("all")
    result = zx.draw_matplotlib(graph, labels=True, figsize=(12, 4))
    fig = result if hasattr(result, "savefig") else plt.gcf()
    if fig.axes:
        fig.axes[0].set_title(title)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def graph_size(graph) -> tuple[int, int]:
    return len(list(graph.vertices())), len(list(graph.edges()))


def discopy_tensor_to_matrix(tensor) -> np.ndarray:
    """Convert a DisCoPy tensor with dom+cod axes into a usual matrix.

    DisCoPy stores pure circuit tensors with domain axes followed by codomain
    axes. A standard linear-map matrix is indexed as output rows by input
    columns, so the axes must be reordered before reshaping.
    """
    array = np.asarray(tensor.array, dtype=complex)
    n_dom = len(tensor.dom)
    n_cod = len(tensor.cod)
    axes = list(range(n_dom, n_dom + n_cod)) + list(range(n_dom))
    return np.transpose(array, axes).reshape(2 ** n_cod, 2 ** n_dom)


# ---------------------------------------------------------------------------
# 3. DisCoPy implementation
# ---------------------------------------------------------------------------
def build_discopy_coherent_teleportation() -> Any:
    """Build the three-qubit coherent teleportation circuit in DisCoPy."""
    circuit = Circuit.id(3)
    circuit = circuit.H(1)          # Bell pair preparation
    circuit = circuit.CX(1, 2)
    circuit = circuit.CX(0, 1)      # Alice Bell-basis transform
    circuit = circuit.H(0)
    circuit = circuit.CX(1, 2)      # coherent X correction
    circuit = circuit.H(2)          # CZ(0,2) = H2 CNOT(0,2) H2
    circuit = circuit.CX(0, 2)
    circuit = circuit.H(2)
    return circuit


# ---------------------------------------------------------------------------
# 4. PyZX implementation
# ---------------------------------------------------------------------------
def build_pyzx_coherent_teleportation():
    """Build the same Clifford circuit in PyZX."""
    circuit = zx.Circuit(3, name="coherent_quantum_teleportation")
    circuit.add_gate("HAD", 1)
    circuit.add_gate("CNOT", 1, 2)
    circuit.add_gate("CNOT", 0, 1)
    circuit.add_gate("HAD", 0)
    circuit.add_gate("CNOT", 1, 2)
    circuit.add_gate("HAD", 2)
    circuit.add_gate("CNOT", 0, 2)
    circuit.add_gate("HAD", 2)
    return circuit


# ---------------------------------------------------------------------------
# 5. Formal verification and experiments
# ---------------------------------------------------------------------------
def canonical_branch_verification() -> list[dict]:
    rows = []
    for m0 in (0, 1):
        for m1 in (0, 1):
            B = branch_map(m0, m1)
            C = branch_correction(m0, m1)
            corrected = 2.0 * C @ B
            error = float(np.linalg.norm(corrected - I2))
            rows.append({
                "m0": m0,
                "m1": m1,
                "branch_probability": 0.25,
                "corrected_identity_error": error,
                "verified": bool(error < TOL),
            })
    return rows


def random_state_verification(U: np.ndarray, n: int = 100, seed: int = 20260901):
    rng = np.random.default_rng(seed)
    rows = []
    fixed_ancilla = np.kron(ket0, ket0)
    target_prefix = np.kron(ket_plus, ket_plus)

    for i in range(n):
        psi = rng.normal(size=2) + 1j * rng.normal(size=2)
        psi = psi / np.linalg.norm(psi)
        inp = np.kron(psi, fixed_ancilla)
        out = U @ inp
        target = np.kron(target_prefix, psi)
        fidelity = float(abs(np.vdot(target, out)) ** 2)
        state_error = float(np.linalg.norm(out - target))
        rows.append({
            "test": i + 1,
            "alpha_real": float(psi[0].real),
            "alpha_imag": float(psi[0].imag),
            "beta_real": float(psi[1].real),
            "beta_imag": float(psi[1].imag),
            "fidelity": fidelity,
            "state_vector_error": state_error,
        })
    return rows


def main() -> None:
    print("=" * 76)
    print("QUANTUM TELEPORTATION FORMAL VERIFICATION VIA ZX CALCULUS")
    print("DisCoPy + PyZX implementation")
    print("=" * 76)
    print(f"Python  : {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"Matplotlib backend: {matplotlib.get_backend()}")
    print(f"PyZX    : {pyzx_version()}")
    print(f"DisCoPy : {discopy_version()}")
    print()

    summary = {
        "python": sys.version.split()[0],
        "pyzx": pyzx_version(),
        "discopy": discopy_version(),
        "tolerance": TOL,
    }

    # A. Formal branch verification.
    print("[A] Canonical branch-by-branch verification")
    branch_rows = canonical_branch_verification()
    for row in branch_rows:
        print(
            f"  branch {row['m0']}{row['m1']} | p={row['branch_probability']:.3f} "
            f"| corrected error={row['corrected_identity_error']:.3e} "
            f"| PASS={row['verified']}"
        )
    summary["all_measurement_branches_verified"] = all(r["verified"] for r in branch_rows)

    with open(OUT / "branch_verification.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=branch_rows[0].keys())
        writer.writeheader()
        writer.writerows(branch_rows)

    # Plot branch verification.
    plt.figure(figsize=(7, 4))
    labels = [f"{r['m0']}{r['m1']}" for r in branch_rows]
    errors = [max(r["corrected_identity_error"], 1e-18) for r in branch_rows]
    plt.bar(labels, errors)
    plt.yscale("log")
    plt.axhline(TOL, linestyle="--", linewidth=1)
    plt.xlabel("Alice measurement branch (m0 m1)")
    plt.ylabel("Corrected identity error (log scale)")
    plt.title("Canonical Teleportation: Branch Verification")
    plt.tight_layout()
    plt.savefig(OUT / "05_branch_verification.png", dpi=220)
    plt.close()

    # B. DisCoPy circuit and matrix.
    print("\n[B] DisCoPy categorical circuit")
    d_circuit = build_discopy_coherent_teleportation()
    d_circuit.draw(
        path=str(OUT / "01_discopy_coherent_teleportation_circuit.png"),
        figsize=(13, 4),
    )
    d_tensor = d_circuit.eval()
    d_matrix = discopy_tensor_to_matrix(d_tensor)
    discopy_reference_error = global_phase_error(d_matrix, U_teleport_reference)
    print(f"  DisCoPy vs analytical U error (global-phase aligned): {discopy_reference_error:.3e}")
    summary["discopy_vs_analytical_unitary_error"] = discopy_reference_error
    summary["discopy_model_verified"] = bool(discopy_reference_error < TOL)

    # C. PyZX graph conversion + simplification.
    print("\n[C] PyZX ZX-calculus conversion and simplification")
    p_circuit = build_pyzx_coherent_teleportation()
    p_matrix = np.asarray(p_circuit.to_matrix(), dtype=complex).reshape(8, 8)
    pyzx_reference_error = global_phase_error(p_matrix, U_teleport_reference)
    print(f"  PyZX vs analytical U error (global-phase aligned): {pyzx_reference_error:.3e}")
    summary["pyzx_vs_analytical_unitary_error"] = pyzx_reference_error
    summary["pyzx_model_verified"] = bool(pyzx_reference_error < TOL)

    g_before = p_circuit.to_graph()
    save_pyzx_graph(
        g_before,
        OUT / "02_pyzx_zx_before_simplification.png",
        "PyZX ZX Diagram Before Simplification",
    )
    v_before, e_before = graph_size(g_before)

    g_after = g_before.copy()
    t0 = time.perf_counter()
    full_reduce(g_after)
    simplify_seconds = time.perf_counter() - t0
    try:
        g_after.normalize()
    except Exception:
        pass
    save_pyzx_graph(
        g_after,
        OUT / "03_pyzx_zx_after_full_reduce.png",
        "PyZX ZX Diagram After full_reduce",
    )
    v_after, e_after = graph_size(g_after)

    semantic_equal = bool(compare_tensors(g_before, g_after))
    print(f"  Semantic equality before/after full_reduce: {semantic_equal}")
    print(f"  Vertices: {v_before} -> {v_after}")
    print(f"  Edges   : {e_before} -> {e_after}")
    print(f"  Rewrite time: {simplify_seconds:.6f} s")

    summary.update({
        "pyzx_semantic_equal_before_after": semantic_equal,
        "zx_vertices_before": v_before,
        "zx_vertices_after": v_after,
        "zx_edges_before": e_before,
        "zx_edges_after": e_after,
        "pyzx_full_reduce_seconds": simplify_seconds,
    })

    # D. Formal induced teleportation map from PyZX unitary.
    print("\n[D] Teleportation theorem on the PyZX unitary")
    effective_map = output_projection @ p_matrix @ input_embedding
    induced_error = global_phase_error(effective_map, I2)
    print("  Effective map (<++| x I) U (I x |00>) =")
    print(np.round(effective_map, 12))
    print(f"  Effective-map identity error: {induced_error:.3e}")
    summary["effective_map_identity_error"] = induced_error
    summary["teleportation_theorem_verified"] = bool(induced_error < TOL)

    # E. Cross-tool PyZX -> DisCoPy ZX -> PyZX round trip.
    print("\n[E] DisCoPy ZX / PyZX interoperability")
    roundtrip_equal = False
    interoperability_error = None
    try:
        discopy_zx = DisCoPyZXDiagram.from_pyzx(g_before)
        discopy_zx.draw(
            path=str(OUT / "04_discopy_zx_imported_from_pyzx.png"),
            figsize=(13, 5),
            aspect="auto",
        )
        g_roundtrip = discopy_zx.to_pyzx()
        roundtrip_equal = bool(compare_tensors(g_before, g_roundtrip))
        print(f"  PyZX -> DisCoPy ZX -> PyZX semantic equality: {roundtrip_equal}")
    except Exception as exc:
        interoperability_error = str(exc)
        print(f"  Interoperability drawing/check failed: {exc}")
        print("  Main PyZX proof remains valid; verify the pinned package versions.")
    summary["pyzx_discopy_zx_roundtrip_equal"] = roundtrip_equal
    summary["interoperability_error"] = interoperability_error

    # F. Random-state verification.
    print("\n[F] 100 random-state teleportation tests")
    random_rows = random_state_verification(p_matrix)
    fidelities = np.array([r["fidelity"] for r in random_rows])
    state_errors = np.array([r["state_vector_error"] for r in random_rows])
    print(f"  Minimum fidelity : {fidelities.min():.15f}")
    print(f"  Mean fidelity    : {fidelities.mean():.15f}")
    print(f"  Maximum state error: {state_errors.max():.3e}")

    summary.update({
        "random_tests": len(random_rows),
        "minimum_fidelity": float(fidelities.min()),
        "mean_fidelity": float(fidelities.mean()),
        "maximum_state_vector_error": float(state_errors.max()),
        "all_random_tests_verified": bool(np.all(fidelities > 1 - TOL)),
    })

    with open(OUT / "random_state_verification.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=random_rows[0].keys())
        writer.writeheader()
        writer.writerows(random_rows)

    plt.figure(figsize=(8, 4))
    plt.plot([r["test"] for r in random_rows], fidelities, marker=".", linewidth=1)
    plt.ylim(0.999999999999, 1.000000000001)
    plt.xlabel("Random normalized input-state test")
    plt.ylabel("Teleportation fidelity")
    plt.title("Teleportation Fidelity Across 100 Random Input States")
    plt.tight_layout()
    plt.savefig(OUT / "06_random_state_fidelity.png", dpi=220)
    plt.close()

    # G. Save summaries.
    # Round-trip interoperability is reported separately because a version-specific
    # API issue should not invalidate the independent PyZX proof.
    summary["overall_pass"] = bool(
        summary["all_measurement_branches_verified"]
        and summary["discopy_model_verified"]
        and summary["pyzx_model_verified"]
        and summary["teleportation_theorem_verified"]
        and summary["pyzx_semantic_equal_before_after"]
        and summary["all_random_tests_verified"]
    )

    with open(OUT / "verification_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(OUT / "verification_summary.txt", "w", encoding="utf-8") as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")

    print("\n" + "=" * 76)
    print(f"OVERALL VERIFICATION PASS: {summary['overall_pass']}")
    print(f"Graphical and numeric results saved in: {OUT}")
    print("Next: run  python show_results.py  to open all generated figures.")
    print("=" * 76)


if __name__ == "__main__":
    main()
