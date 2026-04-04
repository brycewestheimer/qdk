/// Cached compute pipelines for gate operations.
///
/// All three gate kernels (single-qubit, two-qubit, multi-controlled) use
/// the same binding pattern:
///   `@binding(0)` = storage `read_write` (state vector)
///   `@binding(1)` = uniform (gate parameters)
///
/// The bind group layout is shared; only the uniform buffer size differs
/// per dispatch.
pub struct PipelineCache {
    /// Pipeline for single-qubit gate application.
    single_qubit: wgpu::ComputePipeline,
    /// Pipeline for two-qubit (4x4 unitary) gate application.
    two_qubit: wgpu::ComputePipeline,
    /// Pipeline for multi-controlled gate application.
    multi_controlled: wgpu::ComputePipeline,
    /// Bind group layout shared by all gate pipelines.
    bind_group_layout: wgpu::BindGroupLayout,
}

impl PipelineCache {
    /// Compiles shaders and creates compute pipelines.
    ///
    /// This is called once during simulator initialization. Shader compilation
    /// can take 10-100ms, so it must not be on the hot path.
    #[must_use]
    pub fn new(device: &wgpu::Device) -> Self {
        // Load all WGSL shader sources (embedded at compile time)
        let single_qubit_source = include_str!("../shaders/single_qubit_gate.wgsl");
        let two_qubit_source = include_str!("../shaders/two_qubit_gate.wgsl");
        let multi_controlled_source = include_str!("../shaders/multi_controlled_gate.wgsl");

        let single_qubit_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("single_qubit_gate"),
            source: wgpu::ShaderSource::Wgsl(single_qubit_source.into()),
        });
        let two_qubit_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("two_qubit_gate"),
            source: wgpu::ShaderSource::Wgsl(two_qubit_source.into()),
        });
        let multi_controlled_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("multi_controlled_gate"),
            source: wgpu::ShaderSource::Wgsl(multi_controlled_source.into()),
        });

        // Create the bind group layout with two entries:
        //   binding 0: state vector (read-write storage buffer)
        //   binding 1: gate parameters (uniform buffer)
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gate_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let create_pipeline = |module: &wgpu::ShaderModule, label: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        Self {
            single_qubit: create_pipeline(&single_qubit_module, "single_qubit_gate_pipeline"),
            two_qubit: create_pipeline(&two_qubit_module, "two_qubit_gate_pipeline"),
            multi_controlled: create_pipeline(
                &multi_controlled_module,
                "multi_controlled_gate_pipeline",
            ),
            bind_group_layout,
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

    /// Returns the bind group layout for gate pipelines.
    #[must_use]
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }
}
