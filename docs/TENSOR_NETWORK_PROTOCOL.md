# Statevector versus tensor-network comparison protocol

## Purpose

Compare two **classical** simulation methods on the same deterministic synthetic Hisab operator circuits:

- Qiskit Aer `statevector`, used as the dense reference;
- Qiskit Aer `matrix_product_state` (MPS), a matrix-product/tensor-network representation with a configurable bond-dimension cap.

This is a controlled simulator comparison, not quantum hardware execution. It is deliberately separate from the existing full-statevector scaling sweep.

## Matched circuit and observables

The runner builds the same circuit family and seeds as `scripts/run_scaling_aer.py`, strips the statevector-only save instruction, and transpiles once into the shared MPS-compatible gate basis `{ry, rz, cx, h}`. This decomposes the original controlled-RY gates into supported one- and two-qubit gates; both backends then receive that identical circuit.

Rather than exporting a (2^n)-amplitude vector from MPS (which would erase much of the memory advantage being tested), both methods calculate the same set of expectation values:

- (langle Z_i angle) for every qubit;
- (langle Z_i Z_{i+1} angle) for every adjacent pair;
- (langle Z_0 Z_{n-1} angle) as one long-range correlation.

The runner records simulation wall time and the mean/max absolute difference between the MPS expectations and the statevector expectations. It does **not** claim that agreement on these selected observables is equivalent to full-state fidelity.

## Truncation and reproducibility

Defaults: maximum MPS bond dimension 128; truncation threshold (10^{-12}); double precision; fixed seeds 11, 29, and 47. A bond-dimension cap or truncation can introduce approximation error. The chosen cap is reported in all outputs; the default run must be interpreted as a bounded-MPS experiment, not guaranteed exact MPS evolution.

The current frontier includes 20, 22, 24, and 26 qubits; local and nearest-neighbour circuits at 1, 2, and 4 layers; and all-to-all circuits at one layer. It uses three deterministic seeds per configuration. Individual and aggregate results are exported to CSV and JSON, with a Markdown report.

## Metrics and interpretation

- Statevector and MPS simulation wall time (per run).
- MPS speedup ratio, defined as (	ext{statevector time}/	ext{MPS time}); greater than 1 means MPS was faster on that runner.
- Maximum and mean absolute errors over the selected Pauli expectation values.
- Failed configurations, retained rather than silently discarded.

Runtime depends on the CI machine, software versions, transpilation, topology, qubit ordering and bond cap. Dense all-to-all coupling can force larger intermediate bond dimensions than local circuits. Results from this one MPS implementation do not establish scaling limits for all tensor-network contraction algorithms.

## Explicit claim boundary

These experiments do not establish:
- a quantum speedup or quantum advantage;
- hardness against all classical algorithms or all tensor-network methods;
- a BQP-hard natural-language task;
- linguistic or cognitive validity;
- performance on real hardware or a natural-language application.

The benchmark is useful for identifying which circuit structures are classically tractable under this specific MPS configuration and how much accuracy is retained at a selected bond-dimension cap.
