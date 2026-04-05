// Shader sources: conditionally compiled for f32 or f64 emulation mode.
// In f64 mode, ds_math.wgsl is prepended to each f64 shader variant.
#[cfg(not(feature = "f64_emulation"))]
const SINGLE_QUBIT_SHADER: &str = include_str!("../shaders/single_qubit_gate.wgsl");
#[cfg(feature = "f64_emulation")]
const SINGLE_QUBIT_SHADER: &str = concat!(
    include_str!("../shaders/ds_math.wgsl"),
    "\n",
    include_str!("../shaders/single_qubit_gate_f64.wgsl"),
);

#[cfg(not(feature = "f64_emulation"))]
const TWO_QUBIT_SHADER: &str = include_str!("../shaders/two_qubit_gate.wgsl");
#[cfg(feature = "f64_emulation")]
const TWO_QUBIT_SHADER: &str = concat!(
    include_str!("../shaders/ds_math.wgsl"),
    "\n",
    include_str!("../shaders/two_qubit_gate_f64.wgsl"),
);

#[cfg(not(feature = "f64_emulation"))]
const MULTI_CONTROLLED_SHADER: &str = include_str!("../shaders/multi_controlled_gate.wgsl");
#[cfg(feature = "f64_emulation")]
const MULTI_CONTROLLED_SHADER: &str = concat!(
    include_str!("../shaders/ds_math.wgsl"),
    "\n",
    include_str!("../shaders/multi_controlled_gate_f64.wgsl"),
);

#[cfg(not(feature = "f64_emulation"))]
const MEASUREMENT_SHADER: &str = include_str!("../shaders/measurement.wgsl");
#[cfg(feature = "f64_emulation")]
const MEASUREMENT_SHADER: &str = concat!(
    include_str!("../shaders/ds_math.wgsl"),
    "\n",
    include_str!("../shaders/measurement_f64.wgsl"),
);

#[cfg(not(feature = "f64_emulation"))]
const COLLAPSE_SHADER: &str = include_str!("../shaders/collapse.wgsl");
#[cfg(feature = "f64_emulation")]
const COLLAPSE_SHADER: &str = concat!(
    include_str!("../shaders/ds_math.wgsl"),
    "\n",
    include_str!("../shaders/collapse_f64.wgsl"),
);

/// Cached compute pipelines for gate and measurement operations.
///
/// Gate kernels (single-qubit, two-qubit, multi-controlled) and the collapse
/// shader share a binding pattern:
///   `@binding(0)` = storage `read_write` (state vector)
///   `@binding(1)` = uniform (parameters)
///
/// The measurement shader uses a different layout with three bindings:
///   `@binding(0)` = storage `read` (state vector, read-only)
///   `@binding(1)` = uniform (parameters)
///   `@binding(2)` = storage `read_write` (partial sums output)
pub struct PipelineCache {
    /// Pipeline for single-qubit gate application.
    single_qubit: wgpu::ComputePipeline,
    /// Pipeline for two-qubit (4x4 unitary) gate application.
    two_qubit: wgpu::ComputePipeline,
    /// Pipeline for multi-controlled gate application.
    multi_controlled: wgpu::ComputePipeline,
    /// Pipeline for measurement probability reduction.
    measurement: wgpu::ComputePipeline,
    /// Pipeline for post-measurement state collapse.
    collapse: wgpu::ComputePipeline,
    /// Bind group layout shared by gate and collapse pipelines (2 bindings).
    gate_layout: wgpu::BindGroupLayout,
    /// Bind group layout for the measurement pipeline (3 bindings).
    measurement_layout: wgpu::BindGroupLayout,
}

impl PipelineCache {
    /// Compiles shaders and creates compute pipelines.
    ///
    /// This is called once during simulator initialization. Shader compilation
    /// can take 10-100ms, so it must not be on the hot path.
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn new(device: &wgpu::Device) -> Self {
        // Compile WGSL shader modules from the module-level constants
        // (which are conditionally defined for f32 or f64 emulation mode).
        let single_qubit_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("single_qubit_gate"),
            source: wgpu::ShaderSource::Wgsl(SINGLE_QUBIT_SHADER.into()),
        });
        let two_qubit_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("two_qubit_gate"),
            source: wgpu::ShaderSource::Wgsl(TWO_QUBIT_SHADER.into()),
        });
        let multi_controlled_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("multi_controlled_gate"),
            source: wgpu::ShaderSource::Wgsl(MULTI_CONTROLLED_SHADER.into()),
        });
        let measurement_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("measurement"),
            source: wgpu::ShaderSource::Wgsl(MEASUREMENT_SHADER.into()),
        });
        let collapse_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("collapse"),
            source: wgpu::ShaderSource::Wgsl(COLLAPSE_SHADER.into()),
        });

        // Gate/collapse layout: 2 bindings (rw storage + uniform)
        let gate_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gate_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Measurement layout: 3 bindings (read-only storage + uniform + rw storage)
        let measurement_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("measurement_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let gate_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gate_pipeline_layout"),
            bind_group_layouts: &[&gate_layout],
            push_constant_ranges: &[],
        });

        let measurement_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("measurement_pipeline_layout"),
                bind_group_layouts: &[&measurement_layout],
                push_constant_ranges: &[],
            });

        let create_gate_pipeline = |module: &wgpu::ShaderModule, label: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&gate_pipeline_layout),
                module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let measurement = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("measurement_pipeline"),
            layout: Some(&measurement_pipeline_layout),
            module: &measurement_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            single_qubit: create_gate_pipeline(&single_qubit_module, "single_qubit_gate_pipeline"),
            two_qubit: create_gate_pipeline(&two_qubit_module, "two_qubit_gate_pipeline"),
            multi_controlled: create_gate_pipeline(
                &multi_controlled_module,
                "multi_controlled_gate_pipeline",
            ),
            measurement,
            collapse: create_gate_pipeline(&collapse_module, "collapse_pipeline"),
            gate_layout,
            measurement_layout,
        }
    }

    /// Returns the single-qubit gate compute pipeline.
    #[must_use]
    pub fn single_qubit_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.single_qubit
    }

    /// Returns the two-qubit gate compute pipeline.
    #[must_use]
    pub fn two_qubit_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.two_qubit
    }

    /// Returns the multi-controlled gate compute pipeline.
    #[must_use]
    pub fn multi_controlled_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.multi_controlled
    }

    /// Returns the measurement probability reduction pipeline.
    #[must_use]
    pub fn measurement_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.measurement
    }

    /// Returns the post-measurement state collapse pipeline.
    #[must_use]
    pub fn collapse_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.collapse
    }

    /// Returns the bind group layout for gate and collapse pipelines.
    #[must_use]
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.gate_layout
    }

    /// Returns the bind group layout for the measurement pipeline.
    #[must_use]
    pub fn measurement_layout(&self) -> &wgpu::BindGroupLayout {
        &self.measurement_layout
    }
}
