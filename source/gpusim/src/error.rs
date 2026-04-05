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

    /// `device.poll()` failed during buffer readback.
    #[error("GPU device poll failed: {0}")]
    DevicePollFailed(String),

    /// The buffer mapping request was rejected by the GPU driver.
    #[error("GPU buffer map rejected: {0}")]
    BufferMapRejected(String),

    /// The channel used to receive the buffer mapping result was disconnected.
    #[error("buffer map notification channel disconnected")]
    ChannelDisconnected,

    /// A GPU device operation failed.
    #[error("device error: {0}")]
    DeviceError(String),

    /// The GPU's `fma()` intrinsic is not a true fused multiply-add.
    ///
    /// f64 emulation via double-single arithmetic requires hardware FMA to
    /// compute rounding errors. Without it, precision degrades to f32 levels.
    #[cfg(feature = "f64_emulation")]
    #[error(
        "GPU fma() is not a true fused multiply-add; \
         f64 emulation would degrade to f32 precision"
    )]
    FmaNotFused,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_messages_are_descriptive() {
        let e = GpuSimError::DevicePollFailed("timeout".to_string());
        assert!(e.to_string().contains("timeout"));
        let e = GpuSimError::BufferMapRejected("validation error".to_string());
        assert!(e.to_string().contains("validation error"));
        let e = GpuSimError::ChannelDisconnected;
        assert!(!e.to_string().is_empty());
    }

    #[cfg(feature = "f64_emulation")]
    #[test]
    fn fma_not_fused_error_message() {
        let e = GpuSimError::FmaNotFused;
        assert!(e.to_string().contains("fma"));
    }
}
