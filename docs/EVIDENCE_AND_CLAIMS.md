# Evidence and claim boundaries

## Demonstrated

1. A nine-qubit circuit composes operators assigned to premise dynamics, negation, factuality, inference, consequent dynamics, and concession.
2. The reference implementation runs deterministically under fixed Aer and transpiler seeds.
3. The inference-off branch has zero consequent-red activation by circuit construction and test.
4. Four historical notebook records contain completed `ibm_kyiv` jobs with counts.
5. The canonical Wild Card notebook produced the reported lead metrics for the selected sentence.

## Supported as research hypotheses

1. Typed operator composition may preserve distinctions that are obscured by early aggregation.
2. Hilbert-space representations may provide a useful formal language for graded and partially active semantic states.
3. Concession can be investigated as an interference-like operator rather than a Boolean override.

## Not demonstrated

1. Quantum computational advantage or speedup.
2. A general representational advantage over all classical alternatives.
3. Better task accuracy than strong, fairly trained classical language models.
4. Scalability to open-domain natural language.
5. Cognitive, neurological, biological, or clinical validity.

## Terminology

**Quantum implementation** refers to a circuit formulated with quantum gates and executed on a simulator or quantum processor.

**Quantum-inspired** refers to an operator or state-space construction evaluated classically.

**Hardware-traceable** means that a source record retains backend, job identifier, shot count, completion output, and counts. It does not imply that all calibration and transpilation metadata have been archived.

**Reproducible reference** means that the repository fixes software dependencies, seeds, parameters, and output schema so the simulator experiment can be rerun independently.

