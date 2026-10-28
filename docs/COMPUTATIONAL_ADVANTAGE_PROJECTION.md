# Computational-complexity route for Hisab: evidence and limits

## Executive result

The current Qiskit experiments support a serious **scaling warning** for exact classical simulation of dense Hisab-style circuits. They do not, by themselves, prove that Hisab semantic inference has a quantum computational advantage.

The research target should be split into two propositions:

1. **Circuit-simulation proposition:** this specified family becomes expensive for exact statevector simulation, with entanglement that makes exact low-bond-dimension MPS representation costly.
2. **Task-advantage proposition:** a well-defined linguistic inference problem encoded by the operator family can be solved with fewer resources on quantum hardware than by strong classical methods.

The first proposition has empirical and algorithmic support. The second remains unproved. The current RY/RZ/CRY rotation ensemble has not been shown to satisfy a published random-circuit-sampling hardness theorem; therefore that theorem must not be attached directly to the current circuit.

## 1. Observed scaling in the current synthetic family

The reproducible scaling runner sweeps a deterministic circuit family with signed RY(±π/4), RZ(±π/4), and controlled-RY(±π/4) gates. The all-to-all topology applies a dense set of pairwise couplings. A projection script now reads its actual CSV output and emits:

- medians across seeds at each qubit count;
- fitted exponential growth over the measured range;
- a separate structural estimate using the exact statevector work model;
- raw statevector memory, gate-count and depth proxies for larger registers;
- the numerical central-cut Schmidt ranks and entropies that were actually computed;
- prominent claim boundaries in both JSON and Markdown.

The fitted curve and the structural model are deliberately separate. For one dense layer with \(G=\Theta(n^2)\) two-qubit gates, a standard full-statevector simulation uses \(\Theta(G\,2^n)=\Theta(n^2 2^n)\) work and \(\Theta(2^n)\) complex amplitudes. A complex128 amplitude takes 16 bytes, so the raw statevector alone uses \(16\,2^n\) bytes; this excludes simulator workspace and does not estimate peak process memory.

For an increase of two qubits, raw statevector memory grows by exactly 4×. In the existing data, dense all-to-all statevector time also grows by roughly 5× per two qubits across the measured frontier. The generated projection reports the actual measured ratios rather than hard-coding that description.

### Statevector growth is not the same as computational advantage

Statevector cost tells us how one exact classical simulation method scales. It is not a lower bound on tensor networks, stabilizer methods, problem-specific classical algorithms, or approximate methods. It also does not show that an actual quantum processor can run the same circuit with sufficient fidelity, or that the output solves a linguistically valuable task.

## 2. Tensor-network cross-check

A matched Qiskit Aer study used the same circuits to calculate the same local-Z, adjacent-ZZ and endpoint-ZZ expectations with statevector and MPS methods. For all-to-all one-layer circuits, the maximum absolute observable error at a fixed MPS bond-dimension cap of 128 rose as follows:

| Qubits | Maximum selected-observable error |
|---:|---:|
| 20 | 0.0243 |
| 22 | 0.0477 |
| 24 | 0.0875 |
| 26 | 0.1384 |

These are errors in selected Pauli expectations, not full-state fidelity or semantic accuracy. For the same dense family, the numerical Schmidt rank across the balanced cut reached 256 at 16 qubits and 1,024 at 20 qubits at the SVD threshold used in the experiment. Exact MPS representation requires a bond dimension at least as large as the Schmidt rank across that cut. These sampled numerical ranks are circuit-specific evidence; they do not prove that every approximation must fail at the same rate.

The takeaway is narrower and useful: this dense circuit family is less friendly to a fixed-low-bond-dimension MPS approximation than the local/nearest-neighbour controls in the same experiment. Any stronger claim needs larger-bond runs, accuracy-vs-cost curves and additional classical methods.

## 3. What complexity theory permits us to say

### Exact probability evaluation

For sufficiently general universal quantum circuits, exact output-probability/amplitude calculations can encode #P-hard computations. The 2023 Nature Physics result, *The hardness of random quantum circuits*, proves #P-hardness of estimating output probabilities for random quantum circuits in its formal setting and extends the result to IQP circuits. This does not establish that every gate family is hard, or that every output probability of the current Hisab circuit is hard.

### Approximate sampling

Hardness of approximate sampling is a stronger and more conditional statement than high statevector runtime. The established random-circuit-sampling line of work relies on properties of a specified circuit ensemble, including average-case hardness and anti-concentration, and on complexity-theoretic assumptions to turn those properties into a classical hardness argument. A finite noisy circuit benchmark, by itself, does not prove those premises.

### The formal route for Hisab

A credible extension is to specify a typed operator compiler in which:

- semantic alternatives/state preparation map to superposition operations;
- phase-sensitive operators (including a formally specified concessive/interference action) map to non-Clifford phase operations;
- dependencies between semantic elements map to controlled two-qubit interactions;
- evolution is an explicit sequence of typed operators rather than an opaque random layer;
- output semantics are fixed before benchmarking.

If that compiler can efficiently express a universal gate set (for example H, T and CNOT) over its semantic registers, the *worst-case circuit expressivity* can inherit the known hardness of general quantum circuit computations. If an approximate-sampling claim is sought, the sampled circuit family must instead meet the conditions used by the relevant random-circuit or IQP theorem.

This is a conditional route to a hard subfamily—not proof of a useful semantic advantage. An arbitrary universal circuit can be encoded as an artificial operator sequence; that alone does not show that ordinary sentences induce hard instances or that the circuit outperforms NLP baselines.

## 4. A second, more language-adjacent route: quantum oracle sketching

A recent preprint by Zhao et al., arXiv:2604.07639 (v2 submitted 1 October 2026), claims an exponential separation in model size for specific large-classical-data classification and dimension-reduction settings. Its real-data experiments include IMDb movie-review sentiment. The proposed mechanism, quantum oracle sketching (QOS), constructs an approximate phase oracle from random classical samples and combines it with quantum query algorithms and classical-shadow readout.

This is worth evaluating because sentiment is a natural-language task. It is not, however, the same as Hisab's current operator circuit. The proof and algorithm depend on a specific access model: the target is represented as a function/oracle over a large domain, classical samples are streamed to approximate that oracle, and a specified quantum query algorithm operates on it. Ordinary text inputs do not automatically satisfy those assumptions, and fewer qubits alone do not mean less total computation or data access.

There is also a version-control caution. A public mathematical audit by Vellón Gascón (July 2026) reports counterexamples and proof/interface gaps in the source paper's v1, including a finite counterexample to one printed QOS error threshold. That audit explicitly scopes its findings to arXiv:2604.07639v1; the preprint's current v2 was submitted later and should be audited separately before its theorem is relied upon. The correct conclusion is neither to discard all QOS results nor to take the abstract claim as independently verified for Hisab.

**Recommended parallel experiment:** freeze the current v2 and code commit of the QOS authors' public repository; reproduce their IMDb sentiment benchmark under its documented access model; audit the theorem assumptions relevant to that exact code path; then test whether Hisab's structured linguistic operators can be used as a feature/oracle representation without losing the theorem's input-access and sample-complexity guarantees.

References:
- Zhao et al. (2026), “Exponential quantum advantage in processing massive classical data,” arXiv:2604.07639, version history and abstract: https://arxiv.org/abs/2604.07639
- Authors' QOS code and IMDb experiment instructions: https://github.com/haimengzhao/quantum-oracle-sketching
- Vellón Gascón (2026), version-specific audit of source v1, DOI: https://doi.org/10.5281/zenodo.21704142 ; audit repository states that it examines v1, not the later source v2: https://github.com/gatephys/quantum-oracle-sketching

## 5. Why Qiskit simulation still helps

Qiskit Aer is useful for a projection even though it is itself classical:

- the statevector run measures an exact method's finite-size cost;
- entanglement/Schmidt-rank diagnostics help identify when a tensor-network approximation may become expensive;
- matched MPS runs quantify the accuracy/cost trade-off;
- the analytical \(n^2 2^n\) work model and \(16\,2^n\)-byte memory formula extrapolate beyond the exact simulator's practical range.

The projection file is intentionally explicit about which sizes were measured and which are extrapolated. A curve fit over 20–26 qubits must not be called a measured result at 32 or 40 qubits.

## 6. Criteria for an advantage claim

A serious claim about Hisab itself needs all of the following:

1. A fixed, nontrivial semantic task with a target output and scoring rule agreed in advance.
2. A deterministic compiler from labelled input structures to a documented gate family, with size polynomial in input length.
3. A classical reference that verifies small cases and competitive classical baselines for the task.
4. Scaling of output quality against total resource cost: preprocessing/input encoding, circuit gates/depth, shots, readout, mitigation and noise.
5. An applicable complexity argument for the precise gate ensemble and output task—not an analogy to random circuits.
6. Repeated sizes/seeds, uncertainty estimates, and a statement of the assumptions needed for extrapolation.

Until then, the strongest responsible conclusion is: **Hisab's current dense circuit family displays exponential growth for exact statevector simulation and increasing fixed-bond MPS error in the tested range. That is a reason to investigate a complexity-grounded operator extension, not proof of a quantum advantage in language processing.**

## References

1. Ramis Movassagh (2023), “The hardness of random quantum circuits,” *Nature Physics*. https://www.nature.com/articles/s41567-023-02131-2
2. A. Bouland, W. Fefferman, C. Nirkhe and U. Vazirani (2019), “On the complexity and verification of quantum random circuit sampling,” *Nature Physics*. https://www.nature.com/articles/s41567-018-0318-2
3. A. Bouland, B. Fefferman, C. Nirkhe and U. Vazirani (2018/2019), “The Complexity and Verification of Quantum Random Circuit Sampling,” NIST record and citation. https://www.nist.gov/publications/complexity-and-verification-quantum-random-circuit-sampling
4. “Boundaries of quantum supremacy via random circuit sampling,” *npj Quantum Information* (2023), discusses the finite width/depth region in which classical simulation is challenged. https://www.nature.com/articles/s41534-023-00703-x
5. “Does provable absence of barren plateaus imply classical simulability?” *Nature Communications* (2025), reviews why structures making variational circuits trainable may also make them classically simulable in many cases. https://www.nature.com/articles/s41467-025-63099-6
