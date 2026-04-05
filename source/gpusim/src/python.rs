// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// Python-style docstrings use unbackticked identifiers (RuntimeError, sim.allocate(), etc.)
// that are for Python's help() system, not rustdoc.
#![allow(clippy::doc_markdown)]

//! PyO3 bindings for the GPU quantum simulator.
//!
//! This module exposes `GpuQuantumSim` to Python as the `qdk_gpu_sim` package.
//! All methods delegate directly to the underlying Rust struct -- no business
//! logic lives in the Python layer.
//!
//! Gate methods on the Rust side do NOT return `Result` -- they panic on invalid
//! qubit IDs. The Python wrapper catches these panics using `std::panic::catch_unwind`
//! and converts them to Python `RuntimeError` exceptions.

use std::panic::{self, AssertUnwindSafe};

use num_complex::Complex64;
use pyo3::exceptions::{PyOSError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyComplex;

use crate::GpuQuantumSim;
use crate::error::GpuSimError;

/// Convert a [`GpuSimError`] into a typed Python exception.
///
/// - Hardware/driver errors -> `OSError`
/// - Invalid qubit IDs -> `ValueError`
/// - GPU runtime errors -> `RuntimeError`
// PyO3 map_err callback takes owned GpuSimError for conversion.
#[allow(clippy::needless_pass_by_value)]
fn to_py_err(e: GpuSimError) -> PyErr {
    match &e {
        GpuSimError::NoAdapter
        | GpuSimError::DeviceRequest(_)
        | GpuSimError::TooManyQubits { .. }
        | GpuSimError::BufferTooLarge { .. } => PyErr::new::<PyOSError, _>(e.to_string()),
        GpuSimError::QubitNotFound(_) | GpuSimError::DuplicateQubit(_) => {
            PyErr::new::<PyValueError, _>(e.to_string())
        }
        #[cfg(feature = "f64_emulation")]
        GpuSimError::FmaNotFused => PyErr::new::<PyOSError, _>(e.to_string()),
        GpuSimError::DevicePollFailed(_)
        | GpuSimError::BufferMapRejected(_)
        | GpuSimError::ChannelDisconnected
        | GpuSimError::DeviceError(_) => PyErr::new::<PyRuntimeError, _>(e.to_string()),
    }
}

/// Catch a Rust panic and convert it to a Python `RuntimeError`.
///
/// Gate methods on `GpuQuantumSim` panic (via `expect()`) on invalid qubit IDs
/// rather than returning `Result`. This wrapper catches those panics at the
/// Python FFI boundary and converts them to exceptions.
fn catch_panic<F, R>(f: F) -> PyResult<R>
where
    F: FnOnce() -> R + panic::UnwindSafe,
{
    match panic::catch_unwind(f) {
        Ok(val) => Ok(val),
        Err(payload) => {
            let msg = if let Some(s) = payload.downcast_ref::<&str>() {
                (*s).to_string()
            } else if let Some(s) = payload.downcast_ref::<String>() {
                s.clone()
            } else {
                "unknown panic in GPU simulator".to_string()
            };
            Err(PyErr::new::<PyRuntimeError, _>(msg))
        }
    }
}

/// GPU-accelerated dense state vector quantum simulator.
///
/// This simulator uses wgpu compute shaders to parallelize quantum gate
/// application across GPU threads. It supports up to ~28 qubits on
/// consumer GPUs (limited by VRAM).
///
/// Example:
///     sim = GpuQuantumSim(seed=42)
///     q0 = sim.allocate()
///     q1 = sim.allocate()
///     sim.h(q0)
///     sim.mcx([q0], q1)  # CNOT -> Bell state
///     state = sim.get_state()
///     sim.release(q0)
///     sim.release(q1)
#[pyclass(name = "GpuQuantumSim")]
pub struct PyGpuQuantumSim {
    inner: GpuQuantumSim,
}

#[pymethods]
impl PyGpuQuantumSim {
    /// Create a new GPU quantum simulator.
    ///
    /// Args:
    ///     seed: Optional RNG seed for reproducible measurement results.
    ///         If None, a random seed is used.
    ///
    /// Raises:
    ///     RuntimeError: If no compatible GPU adapter is found or device
    ///         creation fails.
    #[new]
    #[pyo3(signature = (seed=None))]
    fn new(seed: Option<u64>) -> PyResult<Self> {
        let inner = GpuQuantumSim::new(seed).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    /// Allocate a new qubit, returning its integer ID.
    ///
    /// The qubit is initialized to the |0> state. Qubit IDs are assigned
    /// sequentially starting from 0 and may be recycled after release.
    ///
    /// Returns:
    ///     int: The allocated qubit's ID.
    ///
    /// Raises:
    ///     OSError: If the GPU cannot hold the required state vector.
    fn allocate(&mut self) -> PyResult<usize> {
        self.inner.allocate().map_err(to_py_err)
    }

    /// Release a qubit by ID.
    ///
    /// The qubit is measured (collapsing its state), and its ID becomes
    /// available for reuse. Releasing all qubits resets the simulator.
    ///
    /// Args:
    ///     id: The qubit ID to release.
    ///
    /// Raises:
    ///     RuntimeError: If the qubit ID is invalid or already released.
    fn release(&mut self, id: usize) -> PyResult<()> {
        catch_panic(AssertUnwindSafe(|| self.inner.release(id)))
    }

    // ---- Single-qubit gates ----

    /// Apply the Hadamard gate to a qubit.
    ///
    /// Args:
    ///     target: Target qubit ID.
    ///
    /// Raises:
    ///     RuntimeError: If the qubit ID is invalid.
    fn h(&mut self, target: usize) -> PyResult<()> {
        catch_panic(AssertUnwindSafe(|| self.inner.h(target)))
    }

    /// Apply the Pauli-X (NOT) gate to a qubit.
    ///
    /// Args:
    ///     target: Target qubit ID.
    ///
    /// Raises:
    ///     RuntimeError: If the qubit ID is invalid.
    fn x(&mut self, target: usize) -> PyResult<()> {
        catch_panic(AssertUnwindSafe(|| self.inner.x(target)))
    }

    /// Apply the Pauli-Y gate to a qubit.
    ///
    /// Args:
    ///     target: Target qubit ID.
    ///
    /// Raises:
    ///     RuntimeError: If the qubit ID is invalid.
    fn y(&mut self, target: usize) -> PyResult<()> {
        catch_panic(AssertUnwindSafe(|| self.inner.y(target)))
    }

    /// Apply the Pauli-Z gate to a qubit.
    ///
    /// Args:
    ///     target: Target qubit ID.
    ///
    /// Raises:
    ///     RuntimeError: If the qubit ID is invalid.
    fn z(&mut self, target: usize) -> PyResult<()> {
        catch_panic(AssertUnwindSafe(|| self.inner.z(target)))
    }

    /// Apply the S (phase) gate to a qubit.
    ///
    /// Args:
    ///     target: Target qubit ID.
    ///
    /// Raises:
    ///     RuntimeError: If the qubit ID is invalid.
    fn s(&mut self, target: usize) -> PyResult<()> {
        catch_panic(AssertUnwindSafe(|| self.inner.s(target)))
    }

    /// Apply the S-adjoint gate to a qubit.
    ///
    /// Args:
    ///     target: Target qubit ID.
    ///
    /// Raises:
    ///     RuntimeError: If the qubit ID is invalid.
    fn sadj(&mut self, target: usize) -> PyResult<()> {
        catch_panic(AssertUnwindSafe(|| self.inner.sadj(target)))
    }

    /// Apply the T (pi/8) gate to a qubit.
    ///
    /// Args:
    ///     target: Target qubit ID.
    ///
    /// Raises:
    ///     RuntimeError: If the qubit ID is invalid.
    fn t(&mut self, target: usize) -> PyResult<()> {
        catch_panic(AssertUnwindSafe(|| self.inner.t(target)))
    }

    /// Apply the T-adjoint gate to a qubit.
    ///
    /// Args:
    ///     target: Target qubit ID.
    ///
    /// Raises:
    ///     RuntimeError: If the qubit ID is invalid.
    fn tadj(&mut self, target: usize) -> PyResult<()> {
        catch_panic(AssertUnwindSafe(|| self.inner.tadj(target)))
    }

    /// Apply the sqrt-X gate to a qubit.
    ///
    /// Args:
    ///     target: Target qubit ID.
    ///
    /// Raises:
    ///     RuntimeError: If the qubit ID is invalid.
    fn sx(&mut self, target: usize) -> PyResult<()> {
        catch_panic(AssertUnwindSafe(|| self.inner.sx(target)))
    }

    /// Apply the sqrt-X-adjoint gate to a qubit.
    ///
    /// Args:
    ///     target: Target qubit ID.
    ///
    /// Raises:
    ///     RuntimeError: If the qubit ID is invalid.
    fn sxadj(&mut self, target: usize) -> PyResult<()> {
        catch_panic(AssertUnwindSafe(|| self.inner.sxadj(target)))
    }

    // ---- Rotation gates ----

    /// Apply RX rotation gate.
    ///
    /// Args:
    ///     theta: Rotation angle in radians.
    ///     target: Target qubit ID.
    ///
    /// Raises:
    ///     RuntimeError: If the qubit ID is invalid.
    fn rx(&mut self, theta: f64, target: usize) -> PyResult<()> {
        catch_panic(AssertUnwindSafe(|| self.inner.rx(theta, target)))
    }

    /// Apply RY rotation gate.
    ///
    /// Args:
    ///     theta: Rotation angle in radians.
    ///     target: Target qubit ID.
    ///
    /// Raises:
    ///     RuntimeError: If the qubit ID is invalid.
    fn ry(&mut self, theta: f64, target: usize) -> PyResult<()> {
        catch_panic(AssertUnwindSafe(|| self.inner.ry(theta, target)))
    }

    /// Apply RZ rotation gate.
    ///
    /// Args:
    ///     theta: Rotation angle in radians.
    ///     target: Target qubit ID.
    ///
    /// Raises:
    ///     RuntimeError: If the qubit ID is invalid.
    fn rz(&mut self, theta: f64, target: usize) -> PyResult<()> {
        catch_panic(AssertUnwindSafe(|| self.inner.rz(theta, target)))
    }

    // ---- Multi-controlled gates ----

    /// Multi-controlled X (Toffoli generalization).
    ///
    /// Applies X to the target qubit conditioned on all control qubits
    /// being in the |1> state. With one control, this is CNOT.
    ///
    /// Args:
    ///     controls: List of control qubit IDs.
    ///     target: Target qubit ID.
    ///
    /// Raises:
    ///     RuntimeError: If any qubit ID is invalid.
    // PyO3 extracts Python lists into owned Vec; a reference parameter is not supported at the FFI boundary.
    #[allow(clippy::needless_pass_by_value)]
    fn mcx(&mut self, controls: Vec<usize>, target: usize) -> PyResult<()> {
        catch_panic(AssertUnwindSafe(|| self.inner.mcx(&controls, target)))
    }

    /// Multi-controlled Y gate.
    ///
    /// Args:
    ///     controls: List of control qubit IDs.
    ///     target: Target qubit ID.
    ///
    /// Raises:
    ///     RuntimeError: If any qubit ID is invalid.
    // PyO3 extracts Python lists into owned Vec; a reference parameter is not supported at the FFI boundary.
    #[allow(clippy::needless_pass_by_value)]
    fn mcy(&mut self, controls: Vec<usize>, target: usize) -> PyResult<()> {
        catch_panic(AssertUnwindSafe(|| self.inner.mcy(&controls, target)))
    }

    /// Multi-controlled Z gate.
    ///
    /// Args:
    ///     controls: List of control qubit IDs.
    ///     target: Target qubit ID.
    ///
    /// Raises:
    ///     RuntimeError: If any qubit ID is invalid.
    // PyO3 extracts Python lists into owned Vec; a reference parameter is not supported at the FFI boundary.
    #[allow(clippy::needless_pass_by_value)]
    fn mcz(&mut self, controls: Vec<usize>, target: usize) -> PyResult<()> {
        catch_panic(AssertUnwindSafe(|| self.inner.mcz(&controls, target)))
    }

    /// Multi-controlled Hadamard gate.
    ///
    /// Args:
    ///     controls: List of control qubit IDs.
    ///     target: Target qubit ID.
    ///
    /// Raises:
    ///     RuntimeError: If any qubit ID is invalid.
    // PyO3 extracts Python lists into owned Vec; a reference parameter is not supported at the FFI boundary.
    #[allow(clippy::needless_pass_by_value)]
    fn mch(&mut self, controls: Vec<usize>, target: usize) -> PyResult<()> {
        catch_panic(AssertUnwindSafe(|| self.inner.mch(&controls, target)))
    }

    /// Multi-controlled S gate.
    ///
    /// Args:
    ///     controls: List of control qubit IDs.
    ///     target: Target qubit ID.
    ///
    /// Raises:
    ///     RuntimeError: If any qubit ID is invalid.
    // PyO3 extracts Python lists into owned Vec; a reference parameter is not supported at the FFI boundary.
    #[allow(clippy::needless_pass_by_value)]
    fn mcs(&mut self, controls: Vec<usize>, target: usize) -> PyResult<()> {
        catch_panic(AssertUnwindSafe(|| self.inner.mcs(&controls, target)))
    }

    /// Multi-controlled S-adjoint gate.
    ///
    /// Args:
    ///     controls: List of control qubit IDs.
    ///     target: Target qubit ID.
    ///
    /// Raises:
    ///     RuntimeError: If any qubit ID is invalid.
    // PyO3 extracts Python lists into owned Vec; a reference parameter is not supported at the FFI boundary.
    #[allow(clippy::needless_pass_by_value)]
    fn mcsadj(&mut self, controls: Vec<usize>, target: usize) -> PyResult<()> {
        catch_panic(AssertUnwindSafe(|| self.inner.mcsadj(&controls, target)))
    }

    /// Multi-controlled T gate.
    ///
    /// Args:
    ///     controls: List of control qubit IDs.
    ///     target: Target qubit ID.
    ///
    /// Raises:
    ///     RuntimeError: If any qubit ID is invalid.
    // PyO3 extracts Python lists into owned Vec; a reference parameter is not supported at the FFI boundary.
    #[allow(clippy::needless_pass_by_value)]
    fn mct(&mut self, controls: Vec<usize>, target: usize) -> PyResult<()> {
        catch_panic(AssertUnwindSafe(|| self.inner.mct(&controls, target)))
    }

    /// Multi-controlled T-adjoint gate.
    ///
    /// Args:
    ///     controls: List of control qubit IDs.
    ///     target: Target qubit ID.
    ///
    /// Raises:
    ///     RuntimeError: If any qubit ID is invalid.
    // PyO3 extracts Python lists into owned Vec; a reference parameter is not supported at the FFI boundary.
    #[allow(clippy::needless_pass_by_value)]
    fn mctadj(&mut self, controls: Vec<usize>, target: usize) -> PyResult<()> {
        catch_panic(AssertUnwindSafe(|| self.inner.mctadj(&controls, target)))
    }

    /// Multi-controlled RZ rotation gate.
    ///
    /// Args:
    ///     controls: List of control qubit IDs.
    ///     theta: Rotation angle in radians.
    ///     target: Target qubit ID.
    ///
    /// Raises:
    ///     RuntimeError: If any qubit ID is invalid.
    // PyO3 extracts Python lists into owned Vec; a reference parameter is not supported at the FFI boundary.
    #[allow(clippy::needless_pass_by_value)]
    fn mcrz(&mut self, controls: Vec<usize>, theta: f64, target: usize) -> PyResult<()> {
        catch_panic(AssertUnwindSafe(|| {
            self.inner.mcrz(&controls, theta, target);
        }))
    }

    /// Multi-controlled Phase gate.
    ///
    /// Applies a controlled phase rotation. The phase is specified as
    /// separate real and imaginary parts.
    ///
    /// Args:
    ///     controls: List of control qubit IDs.
    ///     phase_re: Real part of the complex phase value.
    ///     phase_im: Imaginary part of the complex phase value.
    ///     target: Target qubit ID.
    ///
    /// Raises:
    ///     RuntimeError: If any qubit ID is invalid.
    // PyO3 extracts Python lists into owned Vec; a reference parameter is not supported at the FFI boundary.
    #[allow(clippy::needless_pass_by_value)]
    fn mcphase(
        &mut self,
        controls: Vec<usize>,
        phase_re: f64,
        phase_im: f64,
        target: usize,
    ) -> PyResult<()> {
        let phase = Complex64::new(phase_re, phase_im);
        catch_panic(AssertUnwindSafe(|| {
            self.inner.mcphase(&controls, phase, target);
        }))
    }

    /// Swap the logical IDs of two qubits.
    ///
    /// This is a metadata operation -- it permutes the qubit ID mapping
    /// without performing any GPU computation.
    ///
    /// Args:
    ///     q1: First qubit ID.
    ///     q2: Second qubit ID.
    ///
    /// Raises:
    ///     RuntimeError: If either qubit ID is invalid.
    fn swap_qubit_ids(&mut self, q1: usize, q2: usize) -> PyResult<()> {
        catch_panic(AssertUnwindSafe(|| self.inner.swap_qubit_ids(q1, q2)))
    }

    /// Apply the physical SWAP gate to two qubits.
    ///
    /// This applies the 4x4 SWAP unitary to the state vector on the GPU.
    /// It is NOT a qubit ID relabeling -- see swap_qubit_ids for that.
    ///
    /// Args:
    ///     q1: First qubit ID.
    ///     q2: Second qubit ID.
    ///
    /// Raises:
    ///     RuntimeError: If either qubit ID is invalid.
    fn swap(&mut self, q1: usize, q2: usize) -> PyResult<()> {
        catch_panic(AssertUnwindSafe(|| self.inner.swap(q1, q2)))
    }

    // ---- Measurement ----

    /// Measure a single qubit, collapsing its state.
    ///
    /// Args:
    ///     id: Qubit ID to measure.
    ///
    /// Returns:
    ///     bool: False for |0>, True for |1>.
    ///
    /// Raises:
    ///     RuntimeError: If the qubit ID is invalid.
    fn measure(&mut self, id: usize) -> PyResult<bool> {
        self.inner.measure(id).map_err(to_py_err)
    }

    /// Joint measurement on multiple qubits.
    ///
    /// Measures the parity of the given qubits.
    ///
    /// Args:
    ///     ids: List of qubit IDs to measure jointly.
    ///
    /// Returns:
    ///     bool: False for even parity, True for odd parity.
    ///
    /// Raises:
    ///     RuntimeError: If any qubit ID is invalid.
    // PyO3 extracts Python lists into owned Vec; a reference parameter is not supported at the FFI boundary.
    #[allow(clippy::needless_pass_by_value)]
    fn joint_measure(&mut self, ids: Vec<usize>) -> PyResult<bool> {
        self.inner.joint_measure(&ids).map_err(to_py_err)
    }

    /// Compute the probability of odd parity for the given qubits.
    ///
    /// This does not collapse the state.
    ///
    /// Args:
    ///     ids: List of qubit IDs.
    ///
    /// Returns:
    ///     float: Probability in [0.0, 1.0].
    ///
    /// Raises:
    ///     RuntimeError: If any qubit ID is invalid.
    // PyO3 extracts Python lists into owned Vec; a reference parameter is not supported at the FFI boundary.
    #[allow(clippy::needless_pass_by_value)]
    fn joint_probability(&self, ids: Vec<usize>) -> PyResult<f64> {
        self.inner.joint_probability(&ids).map_err(to_py_err)
    }

    /// Check if a qubit is in the |0> state (within floating-point tolerance).
    ///
    /// Args:
    ///     id: Qubit ID.
    ///
    /// Returns:
    ///     bool: True if the qubit's |1> probability is effectively zero.
    ///
    /// Raises:
    ///     RuntimeError: If the qubit ID is invalid.
    fn qubit_is_zero(&self, id: usize) -> PyResult<bool> {
        self.inner.qubit_is_zero(id).map_err(to_py_err)
    }

    // ---- State inspection ----

    /// Return the non-zero amplitudes of the state vector.
    ///
    /// Returns a list of (basis_state, amplitude) tuples where basis_state
    /// is a Python int and amplitude is a Python complex number.
    ///
    /// Returns:
    ///     list[tuple[int, complex]]: Non-zero (basis_state, amplitude) pairs.
    ///
    /// Raises:
    ///     RuntimeError: If GPU readback fails.
    fn get_state<'py>(&self, py: Python<'py>) -> PyResult<Vec<(Py<PyAny>, Bound<'py, PyComplex>)>> {
        let (state, _num_qubits) = self.inner.get_state().map_err(to_py_err)?;
        state
            .into_iter()
            .map(|(index, amplitude)| {
                let py_index = index.into_pyobject(py)?.into_any().unbind();
                let py_complex = PyComplex::from_doubles(py, amplitude.re, amplitude.im);
                Ok((py_index, py_complex))
            })
            .collect()
    }

    /// Return a human-readable string representation of the quantum state.
    ///
    /// Returns:
    ///     str: Formatted state description.
    ///
    /// Raises:
    ///     RuntimeError: If GPU readback fails.
    fn dump(&self) -> PyResult<String> {
        self.inner.dump().map_err(to_py_err)
    }

    /// Set the RNG seed for subsequent measurement outcomes.
    ///
    /// Args:
    ///     seed: 64-bit unsigned integer seed value.
    fn set_rng_seed(&mut self, seed: u64) {
        self.inner.set_rng_seed(seed);
    }

    /// Report the maximum number of qubits supported by the GPU.
    ///
    /// Returns:
    ///     int: Maximum qubit count.
    // PyO3 methods return to Python; #[must_use] is meaningless at the FFI boundary.
    #[allow(clippy::must_use_candidate)]
    fn max_qubits(&self) -> u32 {
        self.inner.max_qubits()
    }
}

/// The `qdk_gpu_sim` Python module.
#[pymodule]
fn qdk_gpu_sim(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGpuQuantumSim>()?;
    Ok(())
}
