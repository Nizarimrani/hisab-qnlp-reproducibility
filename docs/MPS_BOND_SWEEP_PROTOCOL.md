# MPS bond-dimension sensitivity protocol

## Objective

Test whether increasing the maximum MPS bond dimension reduces the discrepancy observed on dense all-to-all circuits, and measure the extra runtime. This follows the completed comparison at `chi <= 128` and the staged tests at `chi = 256`.

## Matched comparison

Each run builds one deterministic synthetic circuit, transpiles it to the shared MPS-compatible basis `{ry, rz, cx, h}`, then computes identical expectation values using Aer statevector and MPS backends. The observables are all local `Z_i`, all adjacent `Z_i Z_{i+1}`, and the end-to-end `Z_0 Z_{n-1}` correlation.

The statevector stage completes first. Its runtime, observable values, and a machine-readable checkpoint are written before MPS starts. The MPS simulation runs in a child process with a configurable deadline. If the child exceeds that deadline, the runner records `status=mps_timeout`, writes the wall time consumed, keeps the statevector measurements, and exits normally so the artifact can upload. MPS runtime and approximation-error fields stay blank unless MPS actually returns.

## Checkpoint files

Each run directory can contain:

- `result.csv`: one row, overwritten atomically by stage with the latest known status and metrics;
- `checkpoint.json` and `summary.json`: stage, status, versions, configuration, and interpretation boundary;
- `statevector_expectations.json`: the reference local Pauli expectations saved before MPS starts;
- `mps_worker_input.json` and `mps_worker_output.json`: worker handoff and output when available;
- `REPORT.md`: a human-readable report that marks missing MPS measurements as missing rather than zero.

The output checkpoint is refreshed after circuit transpilation, after statevector, immediately before MPS, and after MPS success, error, or timeout.

## Resource-bounded checkpoint probe

The dedicated checkpoint-recovery workflow tests the 24-qubit, all-to-all, one-layer circuit at `chi=512`, seed 11, with a five-minute MPS deadline. This is an operational recovery probe, not a claim that five minutes suffices to characterize the approximation at `chi=512`. If the deadline is reached, the experiment should deliver the dense statevector reference and a marked timeout, which is still a useful reproducibility result.

The original 13-cell bond sweep is manual-only, to prevent pushes to the runner from unintentionally relaunching a costly matrix. Each high-cost run should be explicitly bounded.

## Interpretation limits

- Both statevector and MPS are classical simulators.
- Selected Pauli expectation errors are not full-state fidelity.
- A timeout gives no MPS approximation-error estimate; it only bounds elapsed time on the current runner.
- Runtime and memory depend on circuit, gate ordering, Qiskit/Aer versions, and runner resources.
- These synthetic circuits do not establish natural-language validity, classical intractability, or quantum advantage.
