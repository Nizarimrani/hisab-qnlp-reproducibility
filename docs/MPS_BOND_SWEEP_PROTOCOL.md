# MPS bond-dimension sensitivity protocol

## Objective

Test whether increasing the maximum MPS bond dimension reduces the discrepancy observed on dense all-to-all circuits, and measure the additional runtime. This is a follow-up to the completed 84-circuit comparison at `chi <= 128`.

## Initial resource-bounded sweep

- 24 qubits, all-to-all topology, one logical layer: bond dimensions 128, 256 and 512; seeds 11, 29 and 47 at 128 and 256; one seed (11) at 512 as an initial feasibility probe.
- 26 qubits, all-to-all topology, one logical layer: bond dimensions 128 and 256; seeds 11, 29 and 47.
- The 128-cap runs are repeated as a controlled anchor using the same runner and code path.

The 26-qubit, 512-bond-dimension configuration is intentionally excluded from this first sweep. It can be considered only after timing and memory implications of the smaller staged runs are reviewed.

## Matched comparison

Each run builds one deterministic synthetic circuit, transpires it once to the shared MPS-compatible basis `{ry, rz, cx, h}`, then computes identical expectation values using Aer statevector and MPS backends. The observables are every local `Z_i`, every adjacent `Z_i Z_{i+1}`, and the end-to-end `Z_0 Z_{n-1}` correlation.

Metrics include statevector and MPS runtime, their ratio, mean absolute expectation error and maximum absolute expectation error. The statevector reference is computed once per run; each GitHub Actions matrix job is one (qubit count, bond dimension, seed) tuple, allowing errors/timeouts to be isolated rather than lose the entire sweep.

## Interpretation

Increasing `chi` may reduce truncation error but can substantially increase runtime and memory. A decrease in these selected-observable errors does not prove convergence of the full state. A slower MPS simulation only compares this MPS configuration to dense statevector on these synthetic circuits. Neither result demonstrates quantum advantage, all-classical hardness, or linguistic validity.

Do not launch 26 qubits at `chi=512` automatically. First inspect whether `chi=256` reduces error enough to justify additional cost.
