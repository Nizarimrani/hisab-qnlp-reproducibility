# Compositional quantum semantics: reproducibility note

## Scope

This release isolates one restricted compositional-semantic experiment from a larger programme of geometric language representation. The aim is to make the implemented mechanism inspectable and rerunnable without extending the evidence beyond the experiment.

## Target construction

The target sentence combines a negated premise, slow premise evolution, an 80% factuality value, controlled inference, fast consequent evolution, and concessive phase modulation.

The nine-qubit register assigns four qubits to the premise, one to factuality, three to the consequent, and one to the conjunction controlling consequent activation. Factuality probability is prepared with

```text
theta = 2 * asin(sqrt(p)).
```

The consequent dynamics execute only when the projected premise truth and factuality controls are both active. The concession term is implemented as an additional negative controlled phase during consequent evolution.

## Reference parameters

| Parameter | Value |
|---|---:|
| Premise steps | 5 |
| Consequent steps | 3 |
| Premise phase | 0.35 |
| Consequent phase | 0.30 |
| Concession phase | 1.20 |
| Factuality | 0.80 |
| Shots | 4,096 |
| Simulator seed | 20,260,919 |
| Transpiler seed | 20,260,919 |

## Historical result

The recovered research notebook reported premise-subsystem entropy `0.189674` bits, inference-branch probability `0.0170898`, global redness probability `0.0090332`, conditional redness `0.528571`, and zero redness outside the inference branch. The new reference run is deliberately seeded; finite-shot counts may therefore differ from the historical unseeded sample while preserving the tested invariants.

## Hardware record

Four earlier circuits were executed on `ibm_kyiv`. Their job identifiers and counts are preserved under `results/historical_ibm/`. These runs establish hardware execution of selected exploratory circuits, not execution of the complete Wild Card operator stack on hardware.

## Interpretation

The circuit demonstrates compositional behavior inside the implemented mapping. It does not show that the mapping is uniquely correct, linguistically complete, computationally superior, or asymptotically advantageous. Such claims require independently specified semantic tasks, strong resource-matched baselines, repeated experiments, and hardware-aware cost accounting.

