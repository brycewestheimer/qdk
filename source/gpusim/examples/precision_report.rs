// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// Run with: cargo run -p qdk-gpu-sim --features gpu-tests --example precision_report
//
// Produces a formatted precision characterization report including GPU hardware
// information gathered at runtime.

#[cfg(feature = "gpu-tests")]
mod report {
    #![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

    use num_bigint::BigUint;
    use num_complex::Complex64;
    use qdk_gpu_sim::GpuQuantumSim;
    use quantum_sparse_sim::QuantumSim;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    const PRECISION_SEED: u64 = 0xF32_0EC1_510F_ACE0;

    #[derive(Clone)]
    enum Gate {
        H(usize),
        T(usize),
        S(usize),
        Rx(f64, usize),
        Ry(f64, usize),
        Rz(f64, usize),
        Cx(usize, usize),
    }

    fn random_universal_circuit(num_qubits: usize, depth: usize, seed: u64) -> Vec<Gate> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut gates = Vec::new();
        for _ in 0..depth {
            for q in 0..num_qubits {
                let gate = match rng.gen_range(0..6u32) {
                    0 => Gate::H(q),
                    1 => Gate::T(q),
                    2 => Gate::Rx(rng.r#gen::<f64>() * std::f64::consts::TAU, q),
                    3 => Gate::Ry(rng.r#gen::<f64>() * std::f64::consts::TAU, q),
                    4 => Gate::Rz(rng.r#gen::<f64>() * std::f64::consts::TAU, q),
                    _ => Gate::S(q),
                };
                gates.push(gate);
            }
            for q in 0..num_qubits.saturating_sub(1) {
                if rng.gen_bool(0.4) {
                    gates.push(Gate::Cx(q, q + 1));
                }
            }
        }
        gates
    }

    fn run_gpu(circuit: &[Gate], n: usize) -> Vec<Complex64> {
        let mut sim =
            GpuQuantumSim::new(Some(PRECISION_SEED)).expect("GPU simulator should initialize");
        for _ in 0..n {
            sim.allocate();
        }
        for g in circuit {
            match g {
                Gate::H(q) => sim.h(*q),
                Gate::T(q) => sim.t(*q),
                Gate::S(q) => sim.s(*q),
                Gate::Rx(t, q) => sim.rx(*t, *q),
                Gate::Ry(t, q) => sim.ry(*t, *q),
                Gate::Rz(t, q) => sim.rz(*t, *q),
                Gate::Cx(c, t) => sim.mcx(&[*c], *t),
            }
        }
        let (state, _) = sim.get_state().expect("GPU get_state should succeed");
        to_dense(&state, n)
    }

    fn run_sparse(circuit: &[Gate], n: usize) -> Vec<Complex64> {
        let rng = StdRng::seed_from_u64(PRECISION_SEED);
        let mut sim = QuantumSim::new(Some(rng));
        let ids: Vec<usize> = (0..n).map(|_| sim.allocate()).collect();
        for g in circuit {
            match g {
                Gate::H(q) => sim.h(ids[*q]),
                Gate::T(q) => sim.t(ids[*q]),
                Gate::S(q) => sim.s(ids[*q]),
                Gate::Rx(t, q) => sim.rx(*t, ids[*q]),
                Gate::Ry(t, q) => sim.ry(*t, ids[*q]),
                Gate::Rz(t, q) => sim.rz(*t, ids[*q]),
                Gate::Cx(c, t) => sim.mcx(&[ids[*c]], ids[*t]),
            }
        }
        let (state, _) = sim.get_state();
        to_dense(&state, n)
    }

    fn to_dense(sparse: &[(BigUint, Complex64)], num_qubits: usize) -> Vec<Complex64> {
        let dim = 1usize << num_qubits;
        let mut dense = vec![Complex64::new(0.0, 0.0); dim];
        for (idx, amp) in sparse {
            let i = idx.to_u64_digits().first().copied().unwrap_or(0) as usize;
            if i < dim {
                dense[i] = *amp;
            }
        }
        dense
    }

    struct Metrics {
        max_error: f64,
        rms_error: f64,
        fidelity: f64,
        trace_distance: f64,
    }

    fn compute_metrics(gpu: &[Complex64], reference: &[Complex64]) -> Metrics {
        let n = reference.len();
        let mut max_err = 0.0_f64;
        let mut sum_sq = 0.0_f64;
        let mut inner = Complex64::new(0.0, 0.0);

        for i in 0..n {
            let diff = gpu[i] - reference[i];
            let err = diff.norm();
            max_err = max_err.max(err);
            sum_sq += err * err;
            inner += reference[i].conj() * gpu[i];
        }

        let rms = (sum_sq / n as f64).sqrt();
        let fid = inner.norm_sqr();
        let td = (1.0 - fid).max(0.0).sqrt();

        Metrics {
            max_error: max_err,
            rms_error: rms,
            fidelity: fid,
            trace_distance: td,
        }
    }

    pub fn run() {
        // Print GPU hardware information
        let sim = GpuQuantumSim::new(None).expect("GPU simulator should initialize");
        let info = sim.adapter_info();

        println!("GPU Precision Characterization Report");
        println!("=====================================");
        println!();
        println!("GPU: {}", info.name);
        println!("Backend: {:?}", info.backend);
        println!("Driver: {}", info.driver_info);
        println!();

        let qubit_counts = [4, 8, 12, 16, 20];
        let depths = [10, 50, 100, 500, 1000, 5000];

        println!(
            "{:>6} | {:>5} | {:>11} | {:>11} | {:>15} | {:>13}",
            "Qubits", "Depth", "Max Error", "RMS Error", "Fidelity", "Trace Dist"
        );
        println!("{}", "-".repeat(78));

        let mut worst_fidelity = 1.0_f64;
        let mut worst_params = (0, 0);

        for &nq in &qubit_counts {
            for &d in &depths {
                let seed = PRECISION_SEED
                    .wrapping_add(nq as u64 * 1_000_000)
                    .wrapping_add(d as u64);
                let circuit = random_universal_circuit(nq, d, seed);

                let gpu_dense = run_gpu(&circuit, nq);
                let ref_dense = run_sparse(&circuit, nq);
                let m = compute_metrics(&gpu_dense, &ref_dense);

                println!(
                    "{:>6} | {:>5} | {:>11.3e} | {:>11.3e} | {:>15.12} | {:>13.3e}",
                    nq, d, m.max_error, m.rms_error, m.fidelity, m.trace_distance,
                );

                if m.fidelity < worst_fidelity {
                    worst_fidelity = m.fidelity;
                    worst_params = (nq, d);
                }
            }
        }

        println!();
        println!("Analysis:");
        println!(
            "- Worst fidelity: {:.12} at {} qubits, depth {}",
            worst_fidelity, worst_params.0, worst_params.1
        );
        println!(
            "- f32 precision: ~7 decimal digits; error grows as O(depth * eps) where eps ~ 1e-7"
        );
        println!(
            "- For chemistry QPE circuits (depth < 1000, < 20 qubits), f32 is expected to be sufficient"
        );
        println!(
            "- Max component error is the most sensitive metric; fidelity degrades more slowly"
        );
    }
}

#[cfg(feature = "gpu-tests")]
fn main() {
    report::run();
}

#[cfg(not(feature = "gpu-tests"))]
fn main() {
    eprintln!("example 'precision_report' requires --features gpu-tests; skipping.");
}
