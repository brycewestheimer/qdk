# Pythonic circuit-builder API for the QDK

This sample demonstrates [qdk-pythonic](https://github.com/brycewestheimer/qdk-pythonic), a standalone Python package that provides a native circuit-builder API for the Microsoft Quantum Development Kit. Instead of writing Q# strings directly, users construct circuits with Python classes, and the package generates valid Q# under the hood for simulation and resource estimation via the standard `qsharp` pipeline.

The sample includes three notebooks. [qdk_pythonic_api_demo.ipynb](qdk_pythonic_api_demo.ipynb) introduces the core API with side-by-side comparisons against native `qsharp`. [qdk_pythonic_domains.ipynb](qdk_pythonic_domains.ipynb) showcases the domain-specific modules for condensed matter physics, combinatorial optimization, quantum finance, and quantum machine learning. [qdk_pythonic_interop.ipynb](qdk_pythonic_interop.ipynb) demonstrates cross-format code generation (Q# and OpenQASM 3.0), parsing, roundtripping, JSON serialization, parameterized circuits, and circuit composition. The [proposal](qdk_pythonic_api_proposal.md) describes the motivation and architecture.

Install with `pip install git+https://github.com/brycewestheimer/qdk-pythonic.git`. The package requires `qsharp>=1.25` for simulation and resource estimation.
