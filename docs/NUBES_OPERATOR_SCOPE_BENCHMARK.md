# NUBes operator/scope benchmark

## Why this is the next empirical step

The 72-example structured microtask checks the fidelity of a specified circuit-to-rule mapping. It does not process free text. This companion benchmark tests a narrower but real language-processing component: identifying negation and uncertainty cues and the spans linked to them in Spanish text.

The benchmark uses NUBes SAMPLE-001. The upstream repository documents 29,682 sentences in the full corpus and states that SAMPLE-001 was annotated by two annotators and a referee; other samples were annotated by one annotator. NUBes has negation and uncertainty cues, events, and annotation relations that define their scope.

## Task formulation

The original BRAT annotations are converted to token labels:

- O: token is not inside an annotated cue or scope;
- NEG_CUE / UNCERT_CUE: token overlaps a negation or uncertainty marker;
- NEG_SCOPE / UNCERT_SCOPE: token is inside a target span attached by Scope or DiscScope to that type of cue;
- BOTH_CUE / MIXED_SCOPE: overlapping annotation of both types.

Cue labels take priority if the same token also overlaps a scope. The conversion is computed from BRAT entity offsets and relations, not from a cue lexicon or model predictions.

A text-only classical baseline uses current-token, local-context, character prefix/suffix and token-shape features. A majority-class dummy is reported as a no-skill baseline. The `hisab_inspired_lexicon_scope_rules` system applies an explicit cue lexicon and propagates scope to punctuation, line end, or the next cue. A fourth, hybrid system (`contextual_cues_plus_rule_scope`) uses only cue labels predicted by contextual LinearSVC, discards that model's predicted scope labels, and applies deterministic punctuation-bounded scope propagation. It uses token strings and boundaries only; gold annotation labels are not read during inference. Its outputs are converted to typed records by `hisab_qnlp.operator_semantics` and passed to a variable-length Qiskit circuit. The test split holds out whole text/annotation files to avoid putting tokens from the same file in both train and test.

## Reproduce

The workflow fetches the upstream repository at run time without redistributing the corpus:

    git clone --depth 1 https://github.com/Vicomtech/NUBes-negation-uncertainty-biomedical-corpus.git external/NUBes-negation-uncertainty-biomedical-corpus
    python -m pip install -e '.[test,benchmark]'
    python scripts/run_nubes_operator_scope_benchmark.py --corpus-dir external/NUBes-negation-uncertainty-biomedical-corpus/NUBes/SAMPLE-001 --output-dir results/nubes_operator_scope

The output contains aggregate counts, token-level class metrics for four systems (majority dummy, contextual LinearSVC, lexical/scope rules, and hybrid contextual-cue/rule-scope), strict exact-boundary cue/scope span metrics, timings, software versions, held-out split details, and a consistency audit comparing typed operator composition with a variable-length Qiskit statevector. Exact-span scoring uses maximal contiguous token-tag runs: a one-token boundary error fails the exact match. It does not yet score a discontinuous BRAT scope as one linked multi-part span. The circuit audit uses a declared base truth of true; agreement validates implementation consistency, not semantic correctness. It deliberately excludes source sentences and per-token predictions. See `docs/OPERATOR_SEMANTICS_SCHEMA.md` for the typed interface and gate-level interpretation.

## Attribution and license

Source: https://github.com/Vicomtech/NUBes-negation-uncertainty-biomedical-corpus

Lima Lopez, S.; Perez, N.; Cuadros, M.; Rigau, G. (2020). NUBes: A Corpus of Negation and Uncertainty in Spanish Clinical Texts. LREC 2020.

The upstream repository states that NUBes, IULA+, their experiment splits, and annotation guidelines are licensed under Creative Commons Attribution-ShareAlike 3.0 Spain (CC BY-SA 3.0 ES). Respect attribution and ShareAlike conditions for any redistributed adaptations. The workflow does not store source texts or predictions in this repository or its result artifact. This note is not legal advice.

## Limitations and claim boundaries

- NUBes is biomedical Spanish, not balanced general-domain Spanish.
- This measures token-level cue/scope detection, not event-level inference, concession semantics, or Hisab's entire architecture.
- Both the rule-based pipeline and the hybrid system use punctuation-bounded scope heuristics; they do not reproduce every long-distance or discontinuous BRAT relation.
- The variable-length circuit uses deterministic X gates and a classical statevector simulator. Its agreement with the symbolic reference is a code-consistency check, not a task-level accuracy measure or a quantum advantage result.
- The existing fixed nine-qubit circuit is not applied to arbitrary documents by this pipeline.
- No quantum advantage, quantum speedup, or end-to-end Hisab win is demonstrated.
