# Structured semantic task benchmark v0

## Purpose

This is the first task-level validation step beyond circuit-scaling measurements. It defines an exhaustive, small, structured microtask inspired by Hisab's premise polarity, factuality threshold, conditional activation, concession, and event/aspect stages. It checks that a gate-level Qiskit implementation matches an independently written symbolic rule, then compares simple supervised classifiers on a frozen split.

This is **not** an open-domain NLP benchmark and does not test a text parser. The Spanish sentences are illustrative realizations; structured feature annotations are supplied directly.

## Dataset

The complete factorial design contains 72 cases:

- underlying premise truth: false or true;
- polarity: positive or negated;
- factuality probability: 0.20, 0.50, or 0.80;
- concession: absent or present;
- consequent aspect: onset, ongoing, or completed.

Labels are inactive, not_started, ongoing, and completed.

## Predeclared target rule

1. Negation toggles the truth value of the underlying proposition: effective premise = premise_truth XOR negated.
2. The inference branch activates only if the effective premise is true and the factuality probability reaches the high threshold of 0.80.
3. The concessive marker does not, by itself, cancel consequent activation.
4. If activation occurs, the aspect input selects the consequent state: onset, ongoing, or completed. Otherwise the output is inactive.

These choices define an intentionally simplified task. They are hypotheses for this benchmark, not claims that all natural concessive constructions behave this way or that the target rule is linguistically complete.

## Quantum-circuit encoding

The circuit uses 11 qubits. It encodes the structured input, calculates effective premise truth with two CNOT gates, computes conditional activation with a Toffoli gate, and routes the one-hot aspect into two output bits. Inputs are computational-basis values, so the circuit is a reversible encoding of a deterministic rule. No hardware execution, superposition-based algorithmic benefit, or quantum advantage is claimed.

The circuit is evaluated with Qiskit's exact statevector simulator. Every one of the 72 possible inputs is compared with the independent symbolic oracle. This agreement is a correctness result, not evidence that the circuit is faster or more expressive than a classical implementation.

## Classical baselines

A reproducible stratified split uses 54 training examples and 18 test examples, with seed 20261028. The supervised baselines are:

- most-frequent-class dummy classifier;
- logistic regression;
- decision tree;
- random forest (200 trees, maximum depth 8).

Metrics include test accuracy, macro-F1, fit time, and prediction time. The symbolic oracle is also timed, but it defines the target rule and therefore is a reference implementation rather than a learned competitor.

## Reproduce

Install the project and benchmark extra:

    python -m pip install -e '.[test,benchmark]'
    pytest
    python scripts/run_semantic_task_benchmark.py --output-dir results/semantic_task

The run produces semantic_task_dataset.csv, semantic_task_metrics.json, and REPORT.md. CI uploads these as a workflow artifact.

## Interpretation limits

- The task is synthetic, exhaustively enumerable, and uses supplied feature annotations.
- The example split is small and single-shot; results do not establish generalization to real text.
- The target rule is an engineered specification, not an independently annotated linguistic ground truth.
- Both the circuit and its statevector simulation are classical in execution for this experiment.
- Runtime measurements on the simulator cannot be described as quantum speedup.
- A stronger next phase needs independently annotated natural-language data, predeclared metrics, robust repeated splits, competitive classical language baselines, and full end-to-end resource accounting.
