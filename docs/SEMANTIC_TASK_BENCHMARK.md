# Structured semantic task benchmark v0.2

## Purpose

This is the first task-level validation step beyond circuit-scaling measurements. It defines an exhaustive, small, structured microtask inspired by Hisab's premise polarity, factuality, conditional activation, concession, and event/aspect stages. It checks that a gate-level Qiskit implementation matches an independently written symbolic probability rule, then compares simple supervised classifiers on a frozen split.

This is **not** an open-domain NLP benchmark and does not test a text parser. The Spanish sentences are illustrative realizations; structured feature annotations are supplied directly.

## Dataset

The complete factorial design contains 72 cases:

- underlying proposition truth: false or true;
- polarity: positive or negated;
- factuality probability: 0.20, 0.50, or 0.80;
- concession: absent or present;
- consequent aspect: onset, ongoing, or completed.

The categorical semantic-state labels are inactive, not_started, ongoing, and completed. With the balanced factorial design, the label distribution is 36 inactive and 12 for each active aspect.

## Predeclared target rule

1. Negation toggles the truth value of the underlying proposition: effective premise = premise_truth XOR negated.
2. If the effective premise is false, the output is inactive with probability 1.
3. If the effective premise is true, the factuality probability p is encoded as P(q=1)=p. The output is inactive with probability 1-p and the selected consequent aspect has probability p.
4. The concessive marker does not, by itself, cancel consequent activation.

This is a deliberately simplified probabilistic rule inspired by the premise/factuality-controlled inference in the reference circuit. It is not a claim that all concessive constructions behave this way or that the target rule is a complete linguistic theory. The categorical target label is the licensed semantic state; the probability output separately represents factuality.

## Quantum-circuit encoding

The circuit uses 11 qubits. It encodes the structured input, prepares the factuality qubit with an RY angle whose probability of measuring 1 is p, calculates effective premise truth with two CNOT gates, computes conditional activation with a Toffoli gate, and routes the one-hot aspect into two output bits.

The circuit is evaluated with Qiskit's exact statevector simulator. For every one of the 72 inputs, the four output-status probabilities are compared with the independent symbolic distribution. Agreement is a correctness result, not evidence that the circuit is faster or more expressive than a classical implementation. No hardware execution or quantum advantage is claimed.

## Classical baselines

A reproducible stratified split uses 54 training examples and 18 test examples, with seed 20261028. The supervised baselines are:

- most-frequent-class dummy classifier;
- logistic regression;
- decision tree;
- random forest (200 trees, maximum depth 8).

Metrics include test accuracy, macro-F1, fit time, and prediction time. The symbolic probability rule is also timed, but it defines the target and therefore is a reference implementation rather than a learned competitor.

## Reproduce

Install the project and benchmark extra:

    python -m pip install -e '.[test,benchmark]'
    pytest
    python scripts/run_semantic_task_benchmark.py --output-dir results/semantic_task

The run produces semantic_task_dataset.csv, semantic_task_metrics.json, and REPORT.md. CI uploads these as a workflow artifact.

## Interpretation limits

- The task is synthetic, exhaustively enumerable, and uses supplied feature annotations.
- The supervised test set is small and comes from one fixed split; it does not establish generalization to real text.
- The target rule is engineered, not independent annotated linguistic ground truth.
- The circuit is evaluated by a classical statevector simulator.
- Runtime measurements on the simulator cannot be described as quantum speedup.
- The next phase needs independently annotated natural-language examples, predeclared metrics, repeated splits, competitive classical language baselines, and end-to-end resource accounting.
