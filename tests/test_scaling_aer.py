import sys
from pathlib import Path

import numpy as np
import pytest

# Make the repository root importable when pytest's import mode only adds tests/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qiskit_aer import AerSimulator

from scripts.run_scaling_aer import build_circuit, entropy_central_cut, simulate, tvd


def test_builder_rejects_odd_qubit_count():
    with pytest.raises(ValueError):
        build_circuit(3, 1, "local", 11)


def test_declared_gate_count_matches_topology():
    n, layers, seed = 8, 2, 11
    local = build_circuit(n, layers, "local", seed)
    nearest = build_circuit(n, layers, "nearest", seed)
    dense = build_circuit(n, layers, "all_to_all", seed)
    assert local.count_ops().get("cry", 0) == 0
    assert nearest.count_ops().get("cry", 0) == (n - 1) * layers
    assert dense.count_ops().get("cry", 0) == (n * (n - 1) // 2) * layers


def test_aer_norm_and_phase_readout_control():
    backend = AerSimulator(method="statevector", precision="double")
    state, _ = simulate(build_circuit(4, 1, "local", 11), backend, 11)
    assert np.vdot(state, state).real == pytest.approx(1.0, abs=1e-10)
    no_phase, _ = simulate(build_circuit(4, 1, "local", 11,
                              omit_final_rz_q0=True), backend, 11)
    assert tvd(np.abs(state)**2, np.abs(no_phase)**2) < 1e-10

    mixed, _ = simulate(build_circuit(4, 1, "local", 11,
                           hadamard_readout=True), backend, 11)
    mixed_no_phase, _ = simulate(build_circuit(4, 1, "local", 11,
        omit_final_rz_q0=True, hadamard_readout=True), backend, 11)
    assert tvd(np.abs(mixed)**2, np.abs(mixed_no_phase)**2) > 1e-4


def test_local_circuit_has_zero_central_cut_entropy():
    backend = AerSimulator(method="statevector", precision="double")
    state, _ = simulate(build_circuit(4, 2, "local", 29), backend, 29)
    entropy, rank = entropy_central_cut(state, 4)
    assert entropy == pytest.approx(0.0, abs=1e-9)
    assert rank == 1
