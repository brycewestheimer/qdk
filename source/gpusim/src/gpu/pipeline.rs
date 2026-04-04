/// Cached compute pipelines for gate operations.
pub struct PipelineCache {
    /// Pipeline for single-qubit gate application.
    single_qubit: wgpu::ComputePipeline,
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
        // Load the WGSL shader source (embedded at compile time)
        let shader_source = include_str!("../shaders/single_qubit_gate.wgsl");
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("single_qubit_gate"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
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

        let single_qubit = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("single_qubit_gate_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            single_qubit,
            bind_group_layout,
        }
    }

    /// Returns the single-qubit gate compute pipeline.
    #[must_use]
    pub fn single_qubit_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.single_qubit
    }

    /// Returns the bind group layout for gate pipelines.
    #[must_use]
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }
}
