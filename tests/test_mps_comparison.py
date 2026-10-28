import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_mps_comparison import (  # noqa: E402
    evaluate,
    gate_only_circuit,
    observable_circuit,
)


def test_observable_circuit_contains_local_and_pairwise_expectations():
    gates = gate_only_circuit(4, 1, "nearest", 11)
    circuit, labels = observable_circuit(gates)
    assert len(labels) == 4 + 4  # four Z and four ZZ pairs (three edges plus wraparound)
    assert all(label in labels for label in ("z_0", "z_3", "zz_0_1", "zz_0_3"))
    assert sum(1 for inst in circuit.data if inst.operation.name.startswith("save_")) == 8


def test_mps_matches_statevector_for_small_low_entanglement_case():
    row = evaluate(4, "nearest", 1, 11, max_bond_dimension=128,
                   truncation_threshold=1e-12)
    assert row["status"] == "ok", row.get("error")
    assert row["observable_count"] == 8
    assert row["max_abs_observable_error"] < 1e-9
