# Hisab Qiskit Aer scaling protocol

## Purpose

Measure circuit construction, transpiled circuit resources, statevector-simulation runtime, state normalization, central-cut entanglement diagnostics, and phase-sensitive readout for the current synthetic Hisab operator alphabet.

This is a circuit-engineering experiment, not linguistic validation. It does not establish quantum advantage, classical intractability, or BQP-hardness of natural-language question answering.

## Circuit family

Each logical layer applies signed rotations RY(±π/4) and RZ(±π/4) to all qubits. The entangling operation is controlled-RY(±π/4), with three coupling patterns:

- local: no two-qubit gates; a product-state negative control.
- nearest: nearest-neighbour chain using an alternating brick-wall schedule.
- all_to_all: round-robin pairings intended to apply each unordered qubit pair once per logical layer, with direction/sign selected deterministically.

Every circuit saves its final statevector and is run with Qiskit Aer CPU statevector mode in double precision. Qiskit transpilation uses optimization level zero and deterministic transpiler/simulator seeds. This gives transparent resource counts, but it does not model noisy hardware.

## Progressive sweep

The runner includes replicated cases at 4, 8, 12, and 16 qubits, with 1, 2, and 4 logical layers, and two deterministic seeds. At 20 qubits it runs one seed; all-to-all coupling is capped at two layers. At 22 and 24 qubits, local and nearest-neighbour cases go through four layers, while all-to-all is capped at one layer. Use --max-qubits 16 or 20 to run a smaller sweep.

The exact central-cut singular-value decomposition is computed through 20 qubits only. The largest frontier statevector is 24 qubits (about 268 MB for the raw complex128 state array, excluding simulator/workspace overhead). Runs are written to CSV after each circuit so partial results survive a failed or cancelled run.

## Metrics

- Logical gate counts and transpiled circuit depth/gate counts, including CX count after decomposition.
- Statevector normalization error.
- CPU simulator runtime.
- Von Neumann entropy, normalized cut entropy and numerical Schmidt rank for the central bipartition through 20 qubits.
- As a readout control, omission of the last RZ on qubit 0 should not change Z-basis probabilities in a local circuit, while a Hadamard before readout can make the phase difference observable.

The readout control tests the chosen measurement basis, not a natural-language property. Controlled-RY is not itself a universal hardware-efficient choice; this benchmark tests the current operator alphabet rather than selecting an optimal ansatz.

## Interpretation and stop conditions

If a configuration errors or exceeds the CI wall time, the CSV preserves previous cases and the summary lists failures. First compare circuits at identical qubit counts and logical depths; do not compare different tasks or gate families as if they were the same benchmark.

High central-cut entropy or high Schmidt rank describes the prepared state. Neither metric alone proves that the family is hard for every classical algorithm. A statevector simulator is intentionally exponential in qubit count; its runtime is not a fair proxy for best-possible classical algorithms, nor does a successful 24-qubit run demonstrate useful quantum computational advantage.

No IBM hardware job is submitted by this workflow. Hardware testing, noisy simulation, comparison to optimized tensor-network methods, and any natural-language task remain separate milestones.
