# Proposal: Pythonic Circuit-Builder API for the QDK

## Summary

[qdk-pythonic](https://github.com/brycewestheimer/qdk-pythonic) is a standalone Python
package that provides a native circuit-builder API for the QDK. It generates
valid Q# under the hood and passes it to `qsharp.eval()`, `qsharp.run()`, and
`qsharp.estimate()` — zero QDK modifications required, full resource estimation
fidelity preserved.
```python
from qdk_pythonic import Circuit

circ = Circuit()
q = circ.allocate(2)
circ.h(q[0]).cx(q[0], q[1]).measure_all()

results = circ.run(shots=1000)
estimate = circ.estimate()
```

The package also includes domain modules for condensed matter (Ising,
Heisenberg, Hubbard on lattice geometries), combinatorial optimization
(MaxCut, QUBO, TSP via QAOA), quantum finance (amplitude estimation for
option pricing), and quantum ML (feature encoding, quantum kernels). Each
domain object produces a standard `Circuit`, so code generation, simulation,
and resource estimation work uniformly across all domains.

See [qdk_pythonic_api_demo.ipynb](qdk_pythonic_api_demo.ipynb) for a side-by-side
comparison with native `qsharp`.

## Motivation

The QDK's resource estimator is its primary competitive advantage. Resource
estimation workflows involve constructing parameterized circuits across problem
sizes and hardware configurations — exactly the kind of work where Python's
composability matters most and string interpolation is most painful. A Pythonic
API keeps the entire workflow in Python while preserving full fidelity with the
Q# compilation and estimation pipeline.

Every major quantum framework (Qiskit, Cirq, CUDA-Q) provides a Pythonic
circuit-builder API. The QDK is the notable exception.

## Architecture

The current implementation (Approach A) generates Q# strings from a Python-level
circuit IR. This is a zero-modification approach — it works today as a standalone
`pip install` with no changes to the QDK codebase.

A natural next step (Approach B) would be to expose the simulator's internal
gate operations directly via new `#[pyfunction]` exports in the Rust/PyO3
bindings, bypassing the Q# string compiler entirely. This would eliminate
serialization overhead and enable tighter integration for high-gate-count
circuits. The Pythonic API layer designed in Approach A would remain the same —
only the backend would change from code generation to direct FFI.

## Status / Maturity

The package has full unit test coverage for the core circuit builder, code
generators (Q# and OpenQASM 3.0), parsers, analysis metrics, and all four
domain modules. Integration tests verify end-to-end execution through the
`qsharp` runtime. The supported gate set includes H, X, Y, Z, S, T, Rx,
Ry, Rz, R1, CNOT, CZ, SWAP, and CCNOT, with `Controlled` and `Adjoint`
modifiers. Both code generation and parsing support OpenQASM 3.0.

## Links

- **Repository:** https://github.com/brycewestheimer/qdk-pythonic
- **Demo notebook:** [qdk_pythonic_api_demo.ipynb](qdk_pythonic_api_demo.ipynb)
- **Domain deep-dive:** [qdk_pythonic_domains.ipynb](qdk_pythonic_domains.ipynb)
- **Interop & roundtripping:** [qdk_pythonic_interop.ipynb](qdk_pythonic_interop.ipynb)