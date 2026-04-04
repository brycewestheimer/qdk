/// A single gate operation in a test circuit.
#[derive(Clone, Debug)]
pub enum TestGate {
    // Single-qubit Clifford
    H(usize),
    X(usize),
    Y(usize),
    Z(usize),
    S(usize),
    Sadj(usize),
    SX(usize),
    SXadj(usize),

    // Single-qubit non-Clifford
    T(usize),
    Tadj(usize),

    // Parameterized rotations
    Rx(f64, usize),
    Ry(f64, usize),
    Rz(f64, usize),

    // Two-qubit gates
    Cx(usize, usize), // control, target
    Cy(usize, usize),
    Cz(usize, usize),
    Swap(usize, usize),

    // Multi-controlled
    Mcx(Vec<usize>, usize), // controls, target
    Mcy(Vec<usize>, usize),
    Mcz(Vec<usize>, usize),

    // State management
    Allocate,       // allocates next qubit, appending to the qubit_ids list
    Release(usize), // releases the qubit at the given logical index
}

/// A test circuit: a fixed number of initial qubits plus a sequence of gates.
#[derive(Clone, Debug)]
pub struct TestCircuit {
    pub num_qubits: usize,
    pub gates: Vec<TestGate>,
}
