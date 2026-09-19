"""Canonical compositional quantum-semantics experiment.

The implementation preserves the operator definitions in the recovered research
notebook while making execution deterministic and outputs machine-readable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from importlib.metadata import version
from typing import Mapping

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, entropy, partial_trace


@dataclass(frozen=True)
class ExperimentConfig:
    shots: int = 4096
    premise_steps: int = 5
    consequent_steps: int = 3
    theta_sun: float = 0.35
    theta_redden: float = 0.30
    concession_phase: float = 1.20
    factuality_probability: float = 0.80
    seed_simulator: int = 20260919
    seed_transpiler: int = 20260919
    optimization_level: int = 0

    def __post_init__(self) -> None:
        if self.shots <= 0:
            raise ValueError("shots must be positive")
        if self.premise_steps < 0 or self.consequent_steps < 0:
            raise ValueError("step counts must be non-negative")
        if not 0.0 <= self.factuality_probability <= 1.0:
            raise ValueError("factuality_probability must be in [0, 1]")


def probability_to_ry_angle(probability: float) -> float:
    """Map a Bernoulli probability to an RY preparation angle."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    return 2.0 * math.asin(math.sqrt(probability))


def build_circuit(config: ExperimentConfig, measure: bool = True) -> QuantumCircuit:
    """Build the nine-qubit canonical circuit.

    Layout:
      q0..q3 premise noun, verb, event, and projected truth;
      q4 factuality;
      q5..q7 consequent noun, color, and verb;
      q8 conjunction of projected premise truth and factuality.
    """
    qc = QuantumCircuit(9, 9 if measure else 0)
    premise_noun, premise_verb, premise_event, premise_truth = 0, 1, 2, 3
    factuality = 4
    tomato_noun, tomato_color, tomato_verb = 5, 6, 7
    inference = 8

    qc.x(premise_noun)
    qc.h(premise_verb)
    for _ in range(config.premise_steps):
        qc.h(premise_verb)
        qc.cx(premise_verb, premise_event)
        qc.rz(config.theta_sun, premise_event)
    qc.cx(premise_event, premise_truth)
    qc.x(premise_truth)

    qc.ry(probability_to_ry_angle(config.factuality_probability), factuality)
    qc.ccx(premise_truth, factuality, inference)

    qc.x(tomato_noun)
    qc.ch(inference, tomato_verb)
    for _ in range(config.consequent_steps):
        qc.ch(inference, tomato_verb)
        qc.ccx(inference, tomato_verb, tomato_color)
        qc.crz(config.theta_redden, inference, tomato_color)
        qc.crz(-config.concession_phase, inference, tomato_color)

    if measure:
        qc.measure(range(9), range(9))
    return qc


def bit_at(bitstring: str, qubit_index: int) -> int:
    """Return qubit value for Qiskit's displayed classical order c8...c0."""
    compact = bitstring.replace(" ", "")
    return int(compact[-(qubit_index + 1)])


def summarize_counts(counts: Mapping[str, int]) -> dict[str, object]:
    """Compute registered and conditional probabilities from raw counts."""
    shots = sum(counts.values())
    if shots <= 0:
        raise ValueError("counts must contain at least one shot")

    totals = {"premise_truth": [0, 0], "factuality": [0, 0], "inference": [0, 0]}
    joint_inference_red = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
    joint_premise_factuality = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}

    for bitstring, count in counts.items():
        premise = bit_at(bitstring, 3)
        factuality = bit_at(bitstring, 4)
        inference = bit_at(bitstring, 8)
        red = bit_at(bitstring, 6)
        totals["premise_truth"][premise] += count
        totals["factuality"][factuality] += count
        totals["inference"][inference] += count
        joint_premise_factuality[(premise, factuality)] += count
        joint_inference_red[(inference, red)] += count

    inference_one = totals["inference"][1]
    inference_zero = totals["inference"][0]
    return {
        "shots": shots,
        "p_premise_truth": totals["premise_truth"][1] / shots,
        "p_factuality": totals["factuality"][1] / shots,
        "p_inference": inference_one / shots,
        "p_red": (joint_inference_red[(0, 1)] + joint_inference_red[(1, 1)]) / shots,
        "p_red_given_inference": (
            joint_inference_red[(1, 1)] / inference_one if inference_one else None
        ),
        "p_red_given_no_inference": (
            joint_inference_red[(0, 1)] / inference_zero if inference_zero else None
        ),
        "joint_premise_factuality": {
            f"{p}{f}": count / shots
            for (p, f), count in sorted(joint_premise_factuality.items())
        },
        "joint_inference_red": {
            f"{i}{r}": count / shots
            for (i, r), count in sorted(joint_inference_red.items())
        },
    }


def premise_entropy(config: ExperimentConfig) -> float:
    """Von Neumann entropy of the four-qubit premise subsystem."""
    state = Statevector.from_instruction(build_circuit(config, measure=False))
    reduced = partial_trace(state, [4, 5, 6, 7, 8])
    return float(entropy(reduced, base=2))


def run_experiment(config: ExperimentConfig | None = None) -> dict[str, object]:
    """Execute a deterministic Aer reference run and return a JSON-ready record."""
    cfg = config or ExperimentConfig()
    simulator = AerSimulator(seed_simulator=cfg.seed_simulator)
    circuit = build_circuit(cfg, measure=True)
    compiled = transpile(
        circuit,
        simulator,
        optimization_level=cfg.optimization_level,
        seed_transpiler=cfg.seed_transpiler,
    )
    result = simulator.run(
        compiled,
        shots=cfg.shots,
        seed_simulator=cfg.seed_simulator,
    ).result()
    counts = result.get_counts(compiled)
    metrics = summarize_counts(counts)
    metrics["premise_entropy_bits"] = premise_entropy(cfg)
    return {
        "schema_version": "1.0",
        "experiment_id": "concession-polarity-factuality-v1",
        "sentence_es": "Aunque no salga el sol lentamente, es muy probable (80%) que el tomate enrojezca rápidamente.",
        "config": asdict(cfg),
        "software": {
            "qiskit": version("qiskit"),
            "qiskit_aer": version("qiskit-aer"),
        },
        "circuit": {
            "num_qubits": compiled.num_qubits,
            "depth": compiled.depth(),
            "size": compiled.size(),
            "operation_counts": {str(k): int(v) for k, v in compiled.count_ops().items()},
        },
        "counts": {key: int(value) for key, value in sorted(counts.items())},
        "metrics": metrics,
    }

