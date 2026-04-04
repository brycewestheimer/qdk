use thiserror::Error;

/// Errors that can occur during GPU simulator operation.
#[derive(Debug, Error)]
pub enum GpuSimError {
    /// No suitable wgpu adapter found on this system.
    #[error("no suitable GPU adapter found")]
    NoAdapter,

    /// The wgpu device request failed.
    #[error("GPU device request failed: {0}")]
    DeviceRequest(#[from] wgpu::RequestDeviceError),

    /// The requested state buffer exceeds the GPU's maximum buffer size.
    #[error("state buffer size {requested} bytes exceeds GPU maximum of {max} bytes")]
    BufferTooLarge { requested: u64, max: u64 },

    /// The specified qubit ID is not currently allocated.
    #[error("qubit ID {0} is not allocated")]
    QubitNotFound(usize),

    /// Attempted to allocate a qubit ID that is already in use.
    #[error("qubit ID {0} is already allocated")]
    DuplicateQubit(usize),

    /// Allocating the requested number of qubits would exceed GPU memory.
    #[error("requested {requested} qubits but GPU supports at most {max}")]
    TooManyQubits { requested: u32, max: u32 },

    /// GPU buffer readback (map + copy) failed.
    #[error("GPU buffer readback failed")]
    BufferMapFailed,
}
