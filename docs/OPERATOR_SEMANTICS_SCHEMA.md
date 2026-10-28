# Hisab operator schema and compositional circuit v1

## Objective

Separate three failure surfaces that were previously mixed together:

1. **Extraction:** raw Spanish text to predicted operator cues and scope token indices.
2. **Composition:** typed operator records to an interpretable semantic frame.
3. **Circuit realization:** the same operator records to a Qiskit circuit whose decoded output is checked against an independent symbolic composition function.

This is a minimal implementation contract for research experiments, not a complete linguistic ontology or an assertion that the existing nine-qubit circuit accepts arbitrary text.

## Typed record

```json
{
  "operator_id": "op-0000",
  "kind": "negation",
  "cue_token_indices": [0],
  "scope_token_indices": [1, 2],
  "source": "predicted"
}
```

The kind must be negation or uncertainty. Token indices are zero-based, sorted, unique, and local to the tokenized text unit. The source records whether the frame was built from predicted tags, gold annotations, or a test fixture; gold annotations must not be consumed by inference.

The current adapter converts the rule pipeline's predicted cue tags and scope tags to typed records. It cannot recover every original BRAT relation, particularly discontinuous or overlapping scopes; that limitation is explicit.

## Composition semantics v1

For each target token, start from a declared base truth value named base_truth.

- Each applicable negation operator toggles truth. Two negations therefore cancel by parity in this deliberately minimal algebra.
- Any applicable uncertainty operator sets uncertain=true.
- No numeric probability is inferred from the uncertainty tag because the corpus annotates cues and scopes, not calibrated probabilities.

This is an operational test rule, not a universal account of natural-language negation or modality. Uncertainty and truth are kept as distinct state dimensions.

## Variable-length Qiskit encoding

A circuit is constructed for a target token and the applicable operator list:

- q0 stores the proposition truth bit.
- One X gate on q0 per applicable negation toggles truth.
- Each applicable uncertainty operator receives its own flag qubit set to 1, preserving provenance rather than letting multiple uncertainty operators cancel.
- The statevector output is decoded and compared with the reference composition function.

This circuit is deterministic, uses computational-basis states and X gates only, and is simulated on a classical computer. It validates circuit/reference consistency, not a quantum speedup, quantum advantage, or semantic accuracy against clinical ground truth.

## Reproduction and diagnostics

The NUBes benchmark reports:

- token-label and contiguous exact-span metrics for four systems, plus exact-match scores against original BRAT scope targets;
- number and type of typed operators generated from held-out text using predictions only;
- five-seed, document-grouped holdout means and sample standard deviations;
- original BRAT target exact-match metrics with discontinuous segments retained;
- number of target tokens tested through the variable-length circuit;
- symbolic/circuit agreement, circuit width and gate-count summaries;
- aggregate runtime for schema conversion and statevector evaluation.

The declared base truth used for this consistency check is true for each target token. It is a test convention, not a label found in NUBes. A successful match means that circuit and symbolic implementations agree for the operators produced; it does not mean those operators are linguistically correct.

## Next acceptance criteria

- [x] Add tests for multi-negation, overlapping negation + uncertainty, and punctuation/line scope boundaries.
- [x] Report strict exact-boundary cue/scope span metrics alongside token-label metrics.
- [x] Add a hybrid baseline that uses contextual cue predictions but replaces learned scope labels with deterministic scope propagation.
- [x] Parse linked BRAT target entities with all offset segments retained and report exact-match metrics against original entities.
- [x] Add an end-to-end test showing two predicted contiguous runs do not exactly match one discontinuous BRAT target.
- [ ] Add a relation-aware prediction representation that can emit linked multi-part scope targets; current systems predict contiguous token-tag runs only.
- Define event-level proposition values before interpreting truth-state outputs on real text.
- Compare the circuit and symbolic implementation on identical inputs with end-to-end resource accounting.
- Keep quantum-advantage claims blocked until a real, competitive quantum/classical task comparison exists.
