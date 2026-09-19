# Hisab QNLP Reproducibility

Reproducible reference implementation for a compositional quantum-semantics experiment developed by Team Neuronova and continued at Hisab Labs.

The experiment represents a restricted linguistic composition involving:

- conditional inference;
- polarity (negation);
- epistemic factuality;
- concessive interference;
- adverbial event duration;
- a projected consequent property.

The canonical Spanish sentence is:

> Aunque no salga el sol lentamente, es muy probable (80%) que el tomate enrojezca rápidamente.

English gloss:

> Although the sun does not slowly come out, it is very likely (80%) that the tomato will redden quickly.

## Evidence status

This repository provides:

- a deterministic Qiskit Aer reference implementation;
- semantic invariant tests;
- fixed simulator and transpiler seeds;
- machine-readable output generation;
- a provenance record for four historical `ibm_kyiv` executions;
- explicit boundaries between demonstrated results and open hypotheses.

The historical hardware records show that selected circuits executed successfully on IBM Quantum hardware. They do not establish quantum computational advantage or validate the linguistic interpretation independently.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .[test]
python scripts/run_reference.py --output results/reference/aer_reference.json
pytest
```

## Reproducibility contract

The reference run fixes:

- Qiskit and Aer dependency ranges;
- circuit parameters;
- 4,096 shots;
- simulator seed `20260919`;
- transpiler seed `20260919`;
- optimization level `0`;
- classical-bit ordering.

The generated JSON contains parameters, package versions, circuit resources, counts, and derived metrics.

## Claim boundary

Supported:

- the operator stack can be encoded and executed as a quantum circuit;
- the inference gate prevents consequent activation when its control branch is inactive;
- the supplied historical records contain traceable IBM job identifiers and results;
- the implementation supports controlled simulator reproduction.

Not established:

- quantum computational advantage;
- superiority over all classical semantic models;
- linguistic or cognitive validity beyond the tested construction;
- scalability to unrestricted natural language;
- biological or clinical validity.

See [Evidence and claims](docs/EVIDENCE_AND_CLAIMS.md) and the [technical note](docs/TECHNICAL_NOTE.md).

## Repository layout

```text
src/hisab_qnlp/                 reference implementation
scripts/run_reference.py       reproducible command-line run
tests/                         semantic and numerical invariants
results/historical_ibm/        hardware provenance records
results/reference/             generated Aer output
docs/                          technical and evidence notes
```

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff).

## License

Code and documentation are released under the MIT License.
